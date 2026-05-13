"""영상 디코딩 wrapper — Vertical Slice 단계.

OpenCV `VideoCapture`를 얇게 감싸 BGR frame을 yield. 본 모듈은 2-3-2
`video_preprocessor` 본 구현(frame-level 품질 검사 포함)이 들어오기 전 임시.
지금은 rotation 적용 + frame_stride + meta 조회만 제공.

좌표·색공간: OpenCV 기본 BGR. 회전 메타데이터는 디코더에서 읽어 표시 방향에
맞춰 자동 적용 (휴대폰 세로 영상 호환).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    """디코딩 메타 — rotation 반영 후 표시 기준 크기."""

    width: int
    height: int
    fps: float
    frame_count: int
    rotation_degrees: int = 0


def _normalize_rotation(deg: int) -> int:
    d = int(deg) % 360
    return d if d in (0, 90, 180, 270) else 0


def _read_rotation_meta(cap: cv2.VideoCapture) -> int:
    """OpenCV 4.5+ `CAP_PROP_ORIENTATION_META` (도). 미지원이면 0."""
    prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if prop is None:
        return 0
    try:
        v = float(cap.get(prop))
        return _normalize_rotation(int(round(v))) if np.isfinite(v) else 0
    except (TypeError, ValueError, OverflowError):
        return 0


def _apply_rotation(image: np.ndarray, deg: int) -> np.ndarray:
    if deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _open(video_path: Path) -> tuple[cv2.VideoCapture, VideoMeta]:
    p = video_path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없음: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture 열기 실패 (코덱 미지원·손상 가능): {p}")
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_raw = float(cap.get(cv2.CAP_PROP_FPS))
    fps = fps_raw if fps_raw > 1e-6 else 30.0
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rot = _read_rotation_meta(cap)
    w, h = (raw_h, raw_w) if rot in (90, 270) else (raw_w, raw_h)
    meta = VideoMeta(width=w, height=h, fps=fps, frame_count=fc, rotation_degrees=rot)
    return cap, meta


def get_video_meta(video_path: Path) -> VideoMeta:
    """영상 메타 (width, height, fps, frame_count, rotation) 단발 조회.

    Args:
        video_path: 영상 파일 경로.

    Returns:
        디코딩 메타 — rotation 반영 후 표시 기준 크기.

    Raises:
        FileNotFoundError: 파일 부재.
        RuntimeError: VideoCapture 열기 실패.
    """
    cap, meta = _open(video_path)
    cap.release()
    return meta


def iter_frames(video_path: Path, frame_stride: int = 1) -> Iterator[np.ndarray]:
    """BGR frame을 rotation 적용 후 stride 간격으로 yield.

    Args:
        video_path: 영상 파일 경로.
        frame_stride: N마다 1 frame만 yield. 1이면 모든 frame.

    Yields:
        BGR np.ndarray (height, width, 3) — rotation 반영 후 표시 방향.

    Raises:
        FileNotFoundError: 파일 부재.
        RuntimeError: VideoCapture 열기 실패.
    """
    cap, meta = _open(video_path)
    stride = max(1, int(frame_stride))
    src_idx = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if src_idx % stride == 0:
                yield _apply_rotation(image, meta.rotation_degrees)
            src_idx += 1
    finally:
        cap.release()
