"""MediaPipe Tasks Pose Landmarker — frame별 랜드마크 추출.

본 모듈은 두 path를 지원한다 (호환 모드 A):

**Live path (docs/2-3-3 본 구현)**:
- ProcessedFrame 입력 (2-3-2 video_preprocessor 출력).
- **process_frame 반환은 `PoseLandmarks | None` 단독**. ExtractedFrame
  조립(ProcessedFrame + PoseLandmarks + flags)은 호출자(pipeline.py) 책임 —
  모듈 경계 분리.
- MediaPipe Pose Tasks API, **LIVE_STREAM mode** 운영 (§3-3).
- 비동기 callback 기반 — `detect_async` + `result_callback`. callback 순서가
  frame 도착 순서와 다를 수 있으므로 `dict[frame_index]` 보관 + thread-safe.
- 6 landmark 종 (좌우 12점) — shoulder/hip/knee/ankle/heel/foot_index (§4-1).
- 디버그 모드 시 33 landmark 전체 보존 (`PoseLandmarks.landmarks_full`).
- Phase 1: dataclass + Literal 토대 / Phase 2-2a: __init__ + 모델 로딩 /
  Phase 2-2b: result_callback 본체 / Phase 2-2c-1: process_frame 골격
  (detect_async + polling) / Phase 2-2c-2: _convert_result + process_frame
  변환 통합 (PoseLandmarks | None 반환).

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


@dataclass(eq=False)
class PoseLandmarks:
    """운영 모드 6종 landmark + 디버그 모드 33점 raw — docs/2-3-3 §4-1.

    `eq=False` 이유: `landmarks_full` 필드가 `np.ndarray`라 dataclass 자동
    `__eq__`가 element-wise 비교를 생성해 bool ambiguous error 발생
    (ExtractedFrame과 동일 사유).

    자세 지표 계산에 필요한 6종만 후속 단계로 전달:
    - shoulder: Trunk Lean
    - hip: Trunk Lean, Initial Knee Flexion, IC 검출 (pelvis_x)
    - knee: Initial Knee Flexion
    - ankle: Initial Knee Flexion, Foot Strike Pattern
    - heel: Foot Strike Pattern, IC 검출
    - foot_index: Foot Strike Pattern, IC 검출

    Attributes:
        shoulder/hip/knee/ankle/heel/foot_index: 6종 LandmarkPair (좌/우).
            각 Landmark는 (x, y, visibility) 3필드 — docs §3-4 normalized
            좌표 정책 (z 미사용).
        landmarks_full: 디버그 모드 raw 33점 `np.ndarray (33, 4)` [x, y, z,
            visibility]. `MediaPipePoseConfig.debug_mode=True`일 때만 채움,
            그 외 None (§4-1 메모리 절약). **z는 디버그 raw 자산만** — 운영
            6종은 z 미보존.

    `to_numpy()` 반환: (12, 3) — 6 종 × 2 (좌/우) × 3 (x, y, vis). landmarks_full
    포함 안 함 (디버그 자산 분리).
    행 순서: shoulder L/R, hip L/R, knee L/R, ankle L/R, heel L/R, foot_index L/R.
    """

    shoulder: LandmarkPair
    hip: LandmarkPair
    knee: LandmarkPair
    ankle: LandmarkPair
    heel: LandmarkPair
    foot_index: LandmarkPair
    landmarks_full: Optional[np.ndarray] = None

    def to_numpy(self) -> np.ndarray:
        """(12, 3) np.ndarray — 6 pair × 2 (L/R) × 3 (x, y, vis). landmarks_full 미포함."""
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
    """Pose 추출 + frame 메타 합성 — pipeline.py 조립 단위 (docs/2-3-3 §4 출력 구조).

    `eq=False` 이유: `landmarks` 내부 `landmarks_full` 필드가 `np.ndarray`라
    dataclass 자동 `__eq__`가 element-wise 비교를 생성해 bool ambiguous error 발생.

    Live stream mode callback 순서 보장 X — `frame_index` + `timestamp_sec`
    둘 다 보존하여 후속 모듈이 정렬 가능 (§3-5).

    조립 위치: `PoseExtractor.process_frame`은 `PoseLandmarks` 단독 반환,
    `pipeline.py`가 ProcessedFrame + flags를 합쳐 본 dataclass로 조립. 본
    모듈은 정의만 제공.

    Attributes:
        processed_frame: 2-3-2 ProcessedFrame 통째 보존 (image, frame_quality_flags,
            fps_actual_recent 포함). `frame_quality_flags`는 frame-level 신호,
            본 `pose_quality_flags`와 분리.
        pose_detected: MediaPipe 추출 성공 여부 (§4-2).
        landmarks: 6종 PoseLandmarks (운영 모드). pose_detected=False면 None.
            `landmarks_full` (디버그 raw 33점)은 `PoseLandmarks` 내부에 보관 —
            본 dataclass는 별도 필드 미보유.
        pose_quality_flags: pose 단계 품질 신호. video_preprocessor의 frame-level
            플래그와 분리.
        frame_index: 도착 순서 (callback 순서 보장 X 대비, §3-5).
    """

    processed_frame: ProcessedFrame
    pose_detected: bool
    landmarks: Optional[PoseLandmarks] = None
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
    - Phase 2-2a: `__init__` + LIVE_STREAM options + 모델 로딩
    - Phase 2-2b: `_on_result` 본체 (callback 시 dict 갱신, thread safety)
    - Phase 2-2c-1: `process_frame` 골격 — detect_async + polling
    - Phase 2-2c-2 (본 단계): `_convert_result` + process_frame 통합 —
      `PoseLandmarks | None` 반환
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
        """LIVE_STREAM result callback 본체 (Phase 2-2b).

        비동기 callback — frame 도착 순서와 다를 수 있음 (docs/2-3-3 §3-5).
        timestamp_ms 기준으로 `_pending_timestamps`에서 frame_index 조회하여
        `_results[frame_index]`에 저장.

        Thread safety:
        - 모든 dict 접근 `self._lock` 보호 (`with` 블록).
        - callback 예외는 swallow (메인 스레드 영향 0) + `logger.exception`.
        - pending에 없는 timestamp는 warn (정합 깨짐 신호).
        - dict 크기 10000 초과 시 warn (메모리 누수 / 호출자 pop 누락 신호).

        Args:
            result: `mp.tasks.vision.PoseLandmarkerResult` (typing 의존 회피로
                직접 타입 없음).
            output_image: `mp.Image` (본 모듈에서 사용 안 함, MediaPipe API
                시그니처 요구).
            timestamp_ms: `detect_async` 호출 시 전달한 timestamp.
        """
        try:
            with self._lock:
                frame_index = self._pending_timestamps.pop(timestamp_ms, None)
                if frame_index is None:
                    logger.warning(
                        "callback: pending에 없는 timestamp_ms=%d", timestamp_ms
                    )
                    return
                self._results[frame_index] = result
            # dict 크기 임계 — 메모리 누수 신호 (호출자가 pop 안 함 의심)
            if len(self._results) > 10000:
                logger.warning(
                    "결과 dict 크기 %d, 메모리 누수 가능 — 호출자 pop 누락 확인",
                    len(self._results),
                )
        except Exception:
            logger.exception("result_callback 예외 (swallow)")

    def process_frame(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> Optional[PoseLandmarks]:
        """동기 wrapper — detect_async + polling + 변환 (Phase 2-2c-1 + 2-2c-2 통합).

        호출자 입장 "frame 입력 → PoseLandmarks 반환" 동기 인터페이스. 내부적으로
        비동기 callback 결과를 lock 보호된 dict에서 polling 후, `_convert_result`로
        6종 LandmarkPair 조립하여 반환.

        timestamp_ms 정책: 호출자가 부여 (PoseExtractor는 시간 정책 미보유).
        monotonic increasing + 세션 내 unique 보장은 호출자 책임 (MediaPipe
        LIVE_STREAM 요구). 본 메서드는 timestamp_ms를 frame_index로도 사용
        (identity 매핑) — 별도 frame_index 인자가 필요해지면 시그니처 확장.

        Args:
            frame: BGR np.ndarray (video_preprocessor 출력 형식 가정).
            timestamp_ms: 호출자가 부여한 단조 증가 timestamp (ms).

        Returns:
            `PoseLandmarks` (6종 LandmarkPair, debug_mode=True 시 landmarks_full
            포함) 또는 None — timeout 초과, 또는 MediaPipe pose 미검출, 또는
            변환 예외 시. ExtractedFrame 조립(ProcessedFrame + 본 결과 + flags)은
            호출자(pipeline) 책임.

        Notes:
            - polling 간격 `cfg.polling_interval_sec` (기본 1ms): busy loop
              회피 + GIL 환경 callback thread에 양보. 정당화 상세는 config
              필드 description.
            - timeout 초과 시 `_pending_timestamps` 정리 안 함 — 늦게 도착한
              callback이 `_results`에 쌓이는 것은 `_on_result`의 dict 크기
              가드(10000)가 차단. 호출자는 None 받고 frame skip 결정.
            - 변환은 `_convert_result`에 위임 (Phase 2-2c-2).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        frame_index = timestamp_ms  # identity 매핑 (호출자 monotonic 보장 가정)
        with self._lock:
            self._pending_timestamps[timestamp_ms] = frame_index
        self._landmarker.detect_async(mp_image, timestamp_ms)

        timeout_sec = self._cfg.frame_timeout_sec
        poll_start = time.perf_counter()
        while True:
            with self._lock:
                result = self._results.pop(frame_index, None)
            if result is not None:
                return self._convert_result(result)
            elapsed = time.perf_counter() - poll_start
            if elapsed >= timeout_sec:
                logger.warning(
                    "process_frame timeout — timestamp_ms=%d, %.3fs 초과 "
                    "(cfg.frame_timeout_sec=%.3fs)",
                    timestamp_ms,
                    elapsed,
                    timeout_sec,
                )
                return None
            time.sleep(self._cfg.polling_interval_sec)

    def _convert_result(self, result) -> Optional[PoseLandmarks]:
        """`PoseLandmarkerResult` → `PoseLandmarks` 변환 (Phase 2-2c-2).

        MediaPipe 33점 raw landmark에서 운영 6종 (좌우 12점) 추출 + 조립.
        `debug_mode=True` 시 33점 전체 raw (x, y, z, visibility) ndarray를
        `landmarks_full`에 보존 — 디버그 자산만, 운영 6종은 z 미사용
        (docs §3-4 정합).

        MediaPipe POSE_LANDMARKS index (docs/2-3-3 §4-1):
        - shoulder L/R = 11/12, hip L/R = 23/24, knee L/R = 25/26,
          ankle L/R = 27/28, heel L/R = 29/30, foot_index L/R = 31/32.

        견고성 가드:
        - `result.pose_landmarks` None / 빈 list → None 반환 (pose 미검출, §4-2).
        - landmark 길이 33 미만 → warning + None 반환 (예외 형식).
        - 변환 예외 → `logger.exception` + None 반환 (메인 분석 흐름 보호).

        Args:
            result: `mp.tasks.vision.PoseLandmarkerResult` (typing 의존 회피로
                직접 타입 없음).

        Returns:
            6종 `PoseLandmarks` 또는 None.
        """
        try:
            pose_landmarks_list = getattr(result, "pose_landmarks", None)
            if not pose_landmarks_list:
                return None

            lms = pose_landmarks_list[0]  # num_poses=1 정책 (docs §3-3)
            if len(lms) < 33:
                logger.warning(
                    "_convert_result: landmark 길이 %d (33 미만, 예외 형식)",
                    len(lms),
                )
                return None

            def _lm(idx: int) -> Landmark:
                p = lms[idx]
                return Landmark(
                    x=float(p.x), y=float(p.y), visibility=float(p.visibility)
                )

            shoulder = LandmarkPair(left=_lm(11), right=_lm(12))
            hip = LandmarkPair(left=_lm(23), right=_lm(24))
            knee = LandmarkPair(left=_lm(25), right=_lm(26))
            ankle = LandmarkPair(left=_lm(27), right=_lm(28))
            heel = LandmarkPair(left=_lm(29), right=_lm(30))
            foot_index = LandmarkPair(left=_lm(31), right=_lm(32))

            landmarks_full: Optional[np.ndarray] = None
            if self._cfg.debug_mode:
                landmarks_full = np.array(
                    [
                        (float(p.x), float(p.y), float(p.z), float(p.visibility))
                        for p in lms
                    ],
                    dtype=np.float64,
                )

            return PoseLandmarks(
                shoulder=shoulder,
                hip=hip,
                knee=knee,
                ankle=ankle,
                heel=heel,
                foot_index=foot_index,
                landmarks_full=landmarks_full,
            )
        except Exception:
            logger.exception("_convert_result 예외 (swallow, None 반환)")
            return None


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
