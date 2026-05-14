"""MediaPipe Tasks Pose Landmarker — frame별 랜드마크 추출.

본 모듈은 두 path를 지원한다 (호환 모드 A):

**Live path (docs/2-3-3 본 구현)**:
- ProcessedFrame 입력 (2-3-2 video_preprocessor 출력).
- 출력: ExtractedFrame (ProcessedFrame + PoseLandmarks 6종 + pose_quality_flags).
- MediaPipe Pose Tasks API, **LIVE_STREAM mode** 운영 (§3-3).
- 비동기 callback 기반 — `detect_async` + `result_callback`. callback 순서가
  frame 도착 순서와 다를 수 있으므로 `dict[frame_index]` 보관 + thread-safe.
- 6 landmark 종 (좌우 12점) — shoulder/hip/knee/ankle/heel/foot_index (§4-1).
- 디버그 모드 시 33 landmark 전체 보존 (`landmarks_full`).
- Phase 1: dataclass + Literal 토대 / Phase 2-2a: __init__ + 모델 로딩 /
  Phase 2-2b: result_callback 본체 / Phase 2-2c: process_frame 동기 wrapper.

**File path (demo path, Vertical Slice 임시)**:
- Iterable[np.ndarray] 입력. MediaPipe 단발 호출 (VIDEO mode).
- 출력: list[FramePose] (33×4 numpy).
- 함수: `extract_poses_from_frames`.
- 본 함수는 demo_trunk.py 호환 + Vertical Slice 회의 자산 보존 목적.
  docs/2-3-3 live path 완성 후 점진 deprecate 또는 file→live adapter로 재구성.

좌표계 (양쪽 path 공통):
- **x, y**: 이미지 대비 **정규화** 좌표 [0, 1]. 픽셀 아님.
- **z**: MediaPipe 깊이 스케일(상대값). live path는 사용 안 함 (docs/2-3-3 §3-4
  world landmark 미사용).
- **visibility**: [0, 1] 가시성 점수.

랜드마크 좌/우는 **피사체의 해부학적 좌/우** (시청자 화면 기준 아님).
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Optional

import cv2
import mediapipe as mp
import numpy as np

from choborunner_ai.config import MediaPipePoseConfig
from choborunner_ai.video_preprocessor import ProcessedFrame

logger = logging.getLogger(__name__)


# ============================================================
# Live path (docs/2-3-3) — Phase 1 토대 (dataclass + Literal)
# ============================================================


PoseQualityFlag = Literal[
    "low_pose_visibility",
    "no_pose_detected",
    "multi_pose_detected",
]
"""Pose 단계 품질 플래그 (docs/2-3-3 §4-2 추출 실패 + 추후 확장).

video_preprocessor의 `QualityFlag` (frame-level)와 분리 — pose 단계 신호는
별도 Literal로 모듈 결합 최소화. ExtractedFrame은 둘 다 보존:
- `processed_frame.quality_flags`: frame-level (2-3-2 신호 passthrough)
- `pose_quality_flags`: pose-level (본 Literal)
"""


@dataclass
class Landmark:
    """단일 landmark — normalized 좌표 + visibility (docs/2-3-3 §3-4).

    z 좌표는 본 v1에서 사용 안 함 (world landmark 미사용 정책).
    """

    x: float
    y: float
    visibility: float

    def to_numpy(self) -> np.ndarray:
        """(3,) np.ndarray [x, y, visibility]."""
        return np.array([self.x, self.y, self.visibility], dtype=np.float64)


@dataclass
class LandmarkPair:
    """좌·우 한 쌍 landmark — docs/2-3-3 §4 출력 구조 정합."""

    left: Landmark
    right: Landmark

    def to_numpy(self) -> np.ndarray:
        """(2, 3) np.ndarray — [[left x, y, vis], [right x, y, vis]]."""
        return np.stack([self.left.to_numpy(), self.right.to_numpy()], axis=0)


@dataclass
class PoseLandmarks:
    """운영 모드 6종 landmark — docs/2-3-3 §4-1.

    자세 지표 계산에 필요한 6종만 후속 단계로 전달:
    - shoulder: Trunk Lean
    - hip: Trunk Lean, Initial Knee Flexion, IC 검출 (pelvis_x)
    - knee: Initial Knee Flexion
    - ankle: Initial Knee Flexion, Foot Strike Pattern
    - heel: Foot Strike Pattern, IC 검출
    - foot_index: Foot Strike Pattern, IC 검출

    `to_numpy()` 반환: (12, 3) — 6 종 × 2 (좌/우) × 3 (x, y, vis).
    행 순서: shoulder L/R, hip L/R, knee L/R, ankle L/R, heel L/R, foot_index L/R.
    """

    shoulder: LandmarkPair
    hip: LandmarkPair
    knee: LandmarkPair
    ankle: LandmarkPair
    heel: LandmarkPair
    foot_index: LandmarkPair

    def to_numpy(self) -> np.ndarray:
        """(12, 3) np.ndarray — 6 pair × 2 (L/R) × 3 (x, y, vis)."""
        return np.concatenate(
            [
                self.shoulder.to_numpy(),
                self.hip.to_numpy(),
                self.knee.to_numpy(),
                self.ankle.to_numpy(),
                self.heel.to_numpy(),
                self.foot_index.to_numpy(),
            ],
            axis=0,
        )


@dataclass(eq=False)
class ExtractedFrame:
    """Pose 추출 결과 (docs/2-3-3 §4 출력 구조).

    `eq=False` 이유: `landmarks_full` 필드가 `np.ndarray`라 dataclass 자동
    `__eq__`가 element-wise 비교를 생성해 bool ambiguous error 발생.

    Live stream mode callback 순서 보장 X — `frame_index` + `timestamp_sec`
    둘 다 보존하여 후속 모듈이 정렬 가능 (§3-5).

    Attributes:
        processed_frame: 2-3-2 ProcessedFrame 통째 보존 (image, frame_quality_flags,
            fps_actual_recent 포함). `frame_quality_flags`는 frame-level 신호,
            본 `pose_quality_flags`와 분리.
        pose_detected: MediaPipe 추출 성공 여부 (§4-2).
        landmarks: 6종 PoseLandmarks (운영 모드). pose_detected=False면 None.
        landmarks_full: 33점 전체 `np.ndarray (33, 4)` [x, y, z, visibility].
            `MediaPipePoseConfig.debug_mode=True`일 때만 채움, 그 외 None
            (§4-1 메모리 절약).
        pose_quality_flags: pose 단계 품질 신호. video_preprocessor의 frame-level
            플래그와 분리.
        frame_index: 도착 순서 (callback 순서 보장 X 대비, §3-5).
    """

    processed_frame: ProcessedFrame
    pose_detected: bool
    landmarks: Optional[PoseLandmarks] = None
    landmarks_full: Optional[np.ndarray] = None
    pose_quality_flags: list[PoseQualityFlag] = field(default_factory=list)
    frame_index: int = 0


# ============================================================
# Live path — Phase 2-2a PoseExtractor __init__ + 모델 로딩
# ============================================================


class PoseExtractor:
    """MediaPipe Pose Landmarker LIVE_STREAM mode wrapper (docs/2-3-3).

    Live stream mode 비동기 callback 기반 — `detect_async()` + `result_callback`
    패턴. callback이 도착하는 순서가 frame 도착 순서와 다를 수 있으므로
    (docs/2-3-3 §3-5), `self._results dict[frame_index]` 보관 + `threading.Lock`
    으로 thread-safe 처리.

    Stateful — 모델 인스턴스를 `__init__`에서 한 번만 로딩 후 재사용. 매 frame
    초기화 반복 X (legacy file path의 함수 호출 패턴과 대비).

    Phase 단계:
    - Phase 2-2a (본 단계): `__init__` + LIVE_STREAM options + 모델 로딩
    - Phase 2-2b: `_on_result` 본체 (callback 시 dict 갱신, thread safety)
    - Phase 2-2c: `process_frame` 동기 wrapper (detect_async + polling)
    """

    def __init__(self, cfg: MediaPipePoseConfig) -> None:
        """LIVE_STREAM mode landmarker 초기화.

        Args:
            cfg: MediaPipePoseConfig (DI). model_path / num_poses /
                min_*_confidence / output_segmentation_masks 사용.

        Raises:
            FileNotFoundError: `cfg.model_path` 파일 부재.
            RuntimeError: `PoseLandmarker.create_from_options` 실패.
        """
        if not cfg.model_path.is_file():
            raise FileNotFoundError(
                f"Pose Landmarker 모델 파일이 없음: {cfg.model_path.resolve()}\n"
                f"  assets/models/ 경로 확인. cfg.model_path={cfg.model_path}"
            )

        self._cfg = cfg
        # frame_index → PoseLandmarkerResult (callback에서 채움)
        self._results: dict[int, object] = {}
        # timestamp_ms → frame_index (process_frame에서 등록, callback에서 pop)
        self._pending_timestamps: dict[int, int] = {}
        self._lock = threading.Lock()
        self._last_timestamp_ms: int = -1
        self._printed_first_result: bool = False

        # MediaPipe LIVE_STREAM options 구성
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(cfg.model_path.resolve())),
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=cfg.num_poses,
            min_pose_detection_confidence=cfg.min_pose_detection_confidence,
            min_pose_presence_confidence=cfg.min_pose_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            output_segmentation_masks=cfg.output_segmentation_masks,
            result_callback=self._on_result,
        )

        load_start = time.perf_counter()
        try:
            self._landmarker = PoseLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(
                f"PoseLandmarker.create_from_options 실패: "
                f"cfg.model_path={cfg.model_path}, cause: {e}"
            ) from e
        load_elapsed_ms = (time.perf_counter() - load_start) * 1000.0
        logger.info(
            "PoseExtractor 초기화 완료 — model=%s (%.1f KB), 로딩 %.1f ms",
            cfg.model_path.name,
            cfg.model_path.stat().st_size / 1024,
            load_elapsed_ms,
        )

    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        """LIVE_STREAM result callback — Phase 2-2b에서 본체 채움.

        현재 Phase 2-2a placeholder. Phase 2-2b에서:
        - self._pending_timestamps에서 timestamp_ms로 frame_index 조회
        - self._results[frame_index] = result 저장
        - threading.Lock으로 thread-safe 보호
        - 예외 swallow + logger.exception
        """
        pass


# ============================================================
# File path (demo path, Vertical Slice 임시 — docs/2-3-3 본 구현 외)
# ============================================================


class LM:
    """MediaPipe Pose 33점 인덱스 (Tasks `PoseLandmark` 열거와 동일)."""

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# 러닝 지표에 쓰는 핵심 12점 (양측 shoulder, hip, knee, ankle, heel, foot_index)
KEY_RUNNING_LANDMARK_INDICES: tuple[int, ...] = (
    LM.LEFT_SHOULDER,
    LM.RIGHT_SHOULDER,
    LM.LEFT_HIP,
    LM.RIGHT_HIP,
    LM.LEFT_KNEE,
    LM.RIGHT_KNEE,
    LM.LEFT_ANKLE,
    LM.RIGHT_ANKLE,
    LM.LEFT_HEEL,
    LM.RIGHT_HEEL,
    LM.LEFT_FOOT_INDEX,
    LM.RIGHT_FOOT_INDEX,
)


@dataclass
class FramePose:
    """프레임별 포즈 추출 결과 — **demo path / Vertical Slice 임시**."""

    frame_index: int
    # (33, 4) — [x, y, z, visibility] 정규화 좌표. 포즈 미검출 시 None.
    landmarks: Optional[np.ndarray]


DEFAULT_MODEL_PATH = Path("legacy/demo_02/models/pose_landmarker_lite.task")


def _normalized_landmarks_to_array(pose_lm) -> np.ndarray:
    """Tasks API 포즈 랜드마크 리스트 → (33, 4) numpy 정규화 좌표."""
    out = np.zeros((33, 4), dtype=np.float64)
    for i, p in enumerate(pose_lm):
        out[i, 0] = 0.0 if p.x is None else float(p.x)
        out[i, 1] = 0.0 if p.y is None else float(p.y)
        out[i, 2] = 0.0 if p.z is None else float(p.z)
        out[i, 3] = 0.0 if p.visibility is None else float(p.visibility)
    return out


def extract_poses_from_frames(
    frames: Iterable[np.ndarray],
    fps: float,
    model_path: Path = DEFAULT_MODEL_PATH,
    min_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> list[FramePose]:
    """BGR frame Iterable에서 프레임별 FramePose list 추출 — **demo path / Vertical Slice 임시**.

    docs/2-3-3 live path와 별개. demo_trunk.py가 영상 파일 처리 시 사용.

    Args:
        frames: BGR np.ndarray Iterable (예: `video_preprocessor.iter_frames`).
        fps: 영상 FPS — MediaPipe VIDEO 모드 timestamp 계산용. ≤0이면 33ms/frame 가정.
        model_path: PoseLandmarker `.task` 모델 파일 경로.
        min_detection_confidence: 첫 포즈 검출 신뢰 임계.
        min_pose_presence_confidence: 매 프레임 포즈 존재 신뢰 임계.
        min_tracking_confidence: 추적 신뢰 임계.

    Returns:
        프레임별 FramePose list. 포즈 미검출 frame은 `landmarks=None`.

    Raises:
        FileNotFoundError: 모델 파일 부재.
    """
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Pose Landmarker 모델 파일이 없음: {model_path.resolve()}\n"
            "  legacy/demo_02/models/ 경로 확인 또는 --model 인자로 경로 지정."
        )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )

    results: list[FramePose] = []
    fps_safe = fps if fps > 1e-6 else 30.0
    with PoseLandmarker.create_from_options(options) as landmarker:
        for idx, image in enumerate(frames):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if not rgb.flags["C_CONTIGUOUS"]:
                rgb = np.ascontiguousarray(rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(idx * 1000.0 / fps_safe)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            lm = (
                _normalized_landmarks_to_array(result.pose_landmarks[0])
                if result.pose_landmarks
                else None
            )
            results.append(FramePose(frame_index=idx, landmarks=lm))

    detected = sum(1 for f in results if f.landmarks is not None)
    total = len(results)
    pct = 100.0 * detected / total if total else 0.0
    logger.info(
        "포즈 추출 완료: %d / %d frame (%.1f%%) 검출",
        detected,
        total,
        pct,
    )
    return results
