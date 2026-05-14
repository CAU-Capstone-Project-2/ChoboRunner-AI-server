"""영상 디코딩 + frame-level 품질 검사 wrapper (docs/2-3-2).

본 모듈은 두 모드를 지원한다:

**Live mode (본 구현, docs/2-3-2 §2)**:
- WebSocket binary frame stream 입력 (JPEG decode → 해상도 cap → timestamp →
  품질 검사).
- 입력 형식: bytes (JPEG) + capture_timestamp_ns.
- 출력: `ProcessedFrame`.
- Phase 2~5에서 점진 구현 예정. 본 Phase 1은 토대 (dataclass + Literal)만.

**File mode (demo path, Vertical Slice 임시)**:
- 영상 파일 경로 입력. OpenCV `VideoCapture` 기반.
- 입력: `Path`.
- 출력: BGR `np.ndarray` Iterator + `VideoMeta`.
- 함수: `get_video_meta`, `iter_frames`.
- 본 함수들은 회의 시연용 Vertical Slice 유지 목적 — docs/2-3-2 본 구현 완성 후
  점진 deprecate 또는 file→live adapter로 재구성.

좌표·색공간: OpenCV 기본 BGR. file mode는 OpenCV CAP_PROP_ORIENTATION_META 기반
회전 자동 적용 (휴대폰 세로 영상 호환). live mode는 Android 측에서 회전 처리
가정 (docs/2-3-2 §2 클라이언트 책임).

품질 플래그 (docs/2-3-2 §4, §6):
- `low_brightness`: 평균 휘도 < 50 (brightness_min)
- `motion_blur`: Laplacian variance < 100 (laplacian_var_min)
- `frame_unstable`: 인접 frame SSD 변화량 > 평균 × 2.0 (ssd_change_ratio_max)
- `timestamp_fallback`: capture timestamp 누락 → AI 서버 수신 시각 사용
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)


QualityFlag = Literal[
    "low_brightness",
    "motion_blur",
    "frame_unstable",
    "timestamp_fallback",
]
"""Frame-level 품질 플래그 (docs/2-3-2 §4, §6).

여러 플래그가 동시에 활성화될 수 있다 (예: ["motion_blur", "frame_unstable"]).
플래그 부여된 frame은 stride 누적에서 가중치 0 (docs/2-3-2 §4-4, 2-3-4 §10).
"""


@dataclass(eq=False)
class ProcessedFrame:
    """frame 단위 처리 결과 (docs/2-3-2 §6 출력 구조).

    `eq=False` 이유: `image` 필드가 `np.ndarray`라 dataclass 자동 `__eq__`가
    element-wise 비교를 생성해 bool ambiguous error 발생. 본 모듈에서 인스턴스
    비교 의미 없으므로 eq 생성 비활성.

    Attributes:
        frame_index: 도착 순서 (시간 의미 없음).
        timestamp_sec: capture timestamp 기준 (docs/2-3-2 §3-3). fallback 사용
            시 `quality_flags`에 ``"timestamp_fallback"`` 포함.
        image: BGR `np.ndarray` (H, W, 3) — 해상도 cap 적용 후 (docs/2-3-2 §3-2).
        quality_flags: 품질 검사 결과 (docs/2-3-2 §4). 빈 list = 모두 통과.
        fps_actual_recent: sliding window 기반 실측 fps (docs/2-3-2 §6).
            최근 N frame (FramePreprocessConfig.fps_tracker_window) 인접
            timestamp 간격 평균의 역수.
    """

    frame_index: int
    timestamp_sec: float
    image: np.ndarray
    quality_flags: list[QualityFlag] = field(default_factory=list)
    fps_actual_recent: float = 0.0


# ============================================================
# File mode (demo path, Vertical Slice 임시 — docs/2-3-2 본 구현 외)
# ============================================================


@dataclass
class VideoMeta:
    """File mode 디코딩 메타 — rotation 반영 후 표시 기준 크기.

    demo path / Vertical Slice 임시. live mode에는 직접 대응 안 됨
    (live mode는 ProcessedFrame.fps_actual_recent로 sliding window fps 추적).
    """

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
    """영상 메타 단발 조회 — **demo path / Vertical Slice 임시**.

    docs/2-3-2 live mode와 별개. demo_trunk.py가 영상 파일에서 fps·해상도·
    frame_count 추출 시 사용. live mode는 ProcessedFrame.fps_actual_recent로
    sliding window fps 추적.

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
    """BGR frame을 rotation 적용 후 stride 간격으로 yield — **demo path / Vertical Slice 임시**.

    docs/2-3-2 live mode (WebSocket binary stream) 와 별개. demo_trunk.py가
    영상 파일 처리 시 사용.

    Args:
        video_path: 영상 파일 경로.
        frame_stride: N마다 1 frame만 yield. 1이면 모든 frame.

    Yields:
        BGR `np.ndarray` (height, width, 3) — rotation 반영 후 표시 방향.

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
