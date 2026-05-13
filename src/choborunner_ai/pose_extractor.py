"""MediaPipe Tasks Pose Landmarker — frame별 랜드마크 추출.

Vertical Slice 단계 — legacy/demo_02/pose_extractor.py에서 핵심만 이식.
2-3-3 `pose_extractor` 본 구현 전 임시 모듈.

Config 의존을 풀어 `extract_poses_from_frames`의 함수 인자로 전달 (Pydantic
Settings 통합은 본 단계 면제, CLAUDE.md §11 마일스톤에서 일괄 통합 예정).

-------------------------------------------------------------------------------
좌표계
-------------------------------------------------------------------------------
- **x, y**: 이미지 대비 **정규화** 좌표 [0, 1]. 픽셀 아님.
- **z**: MediaPipe 깊이 스케일(상대값). 본 모듈 출력에 포함하되 Vertical Slice
  지표 산출에서는 사용하지 않음.
- **visibility**: [0, 1] 가시성 점수.

랜드마크 좌/우는 **피사체의 해부학적 좌/우** (시청자 화면 기준 아님).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


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
    """프레임별 포즈 추출 결과."""

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
    """BGR frame Iterable에서 프레임별 FramePose list 추출.

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
