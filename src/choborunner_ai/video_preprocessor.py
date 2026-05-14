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
from collections import deque
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


# ============================================================
# Live mode (docs/2-3-2 본 구현) — Phase 2 정규화
# ============================================================


def decode_jpeg_binary(jpeg_bytes: bytes) -> np.ndarray | None:
    """JPEG binary → BGR `np.ndarray`. 실패 시 None.

    docs/2-3-2 §3-4 디코딩 정책. `decode_failed`는 후속 단계 전달 X (품질 플래그
    정책의 예외). 호출자(통합 Preprocessor)가 None 시 해당 frame skip.

    빈 bytes / 손상 JPEG 등 invalid input 모두 None 반환 — `cv2.imdecode`가
    빈 buffer에서 C++ assertion(`!buf.empty()`)을 raise하므로 빈 bytes 체크 +
    `cv2.error` catch로 안전 처리.

    Args:
        jpeg_bytes: JPEG 압축 binary.

    Returns:
        BGR `np.ndarray` (H, W, 3) or None (decode 실패).
    """
    if not jpeg_bytes:
        return None
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    return img  # None if decode failed


def normalize_resolution(
    frame: np.ndarray, long_side_cap: int = 1280
) -> np.ndarray:
    """긴 변 cap downsample (docs/2-3-2 §3-2).

    `INTER_AREA` 보간. 미만은 업스케일 X (원본 유지) — 미만 입력은 2-3-1의
    `low_resolution` failed로 별도 처리.

    Args:
        frame: BGR `np.ndarray` (H, W, 3).
        long_side_cap: 긴 변 상한 (픽셀, default 1280 = 720p).

    Returns:
        cap 적용 후 frame. 원본보다 작거나 같음.
    """
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= long_side_cap:
        return frame  # 업스케일 X
    scale = long_side_cap / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def should_select_frame_for_fps_grid(
    capture_ts: float,
    last_selected_ts: float | None,
    fps_cap: float = 30.0,
) -> bool:
    """30fps grid 최근접 채택 결정 (docs/2-3-2 §3-1). Stateless.

    30fps 초과(60fps/50fps/45fps) → grid에 맞춰 frame skip.
    30fps 미만(24fps 등) → 보간 X, 모든 frame 채택 (state로 알아서 보장).

    Args:
        capture_ts: 현재 frame capture timestamp (초).
        last_selected_ts: 직전 채택 frame의 capture timestamp. None이면 첫 frame.
        fps_cap: 목표 fps 상한 (default 30).

    Returns:
        True면 채택, False면 skip.
    """
    if last_selected_ts is None:
        return True  # 첫 frame 항상 채택
    grid_interval = 1.0 / fps_cap  # 30fps → 33.3ms
    return (capture_ts - last_selected_ts) >= grid_interval


# ============================================================
# Live mode — Phase 3 timestamp + sliding window fps tracker
# ============================================================


class FpsTracker:
    """Sliding window 기반 fps 추정 (docs/2-3-2 §6).

    `fps_actual_recent` = 최근 N frame 인접 timestamp 간격 평균의 역수.
    매 frame 갱신. 본 클래스는 본 모듈의 **첫 stateful 컴포넌트**이며, 통합
    Preprocessor (Phase 5)가 인스턴스를 보관하고 매 frame `add()` 호출 후
    `fps_recent`를 ProcessedFrame에 첨부한다.

    빈/1개 timestamp 상태: 0.0 반환 (호출자가 fps_cap fallback 결정).
    인접 timestamp 모두 동일(`total_interval <= 0`): 0.0 반환 (zero-division
    방어).
    """

    def __init__(self, window: int = 30) -> None:
        """sliding window 크기 설정 (default 30 = docs/2-3-2 §6).

        Args:
            window: 최근 N frame timestamp 보관. deque maxlen으로 자동 truncate.
        """
        self._timestamps: deque[float] = deque(maxlen=window)

    def add(self, timestamp_sec: float) -> None:
        """timestamp 추가. window 초과 시 가장 오래된 항목 자동 제거."""
        self._timestamps.append(timestamp_sec)

    @property
    def fps_recent(self) -> float:
        """최근 fps 추정 (Hz). 2개 미만 또는 zero-interval 시 0.0."""
        n = len(self._timestamps)
        if n < 2:
            return 0.0
        # 인접 간격 평균 = (last - first) / (n - 1)
        total_interval = self._timestamps[-1] - self._timestamps[0]
        if total_interval <= 0:
            return 0.0
        mean_interval = total_interval / (n - 1)
        return 1.0 / mean_interval

    @property
    def size(self) -> int:
        """현재 window에 있는 timestamp 개수 (0 ~ window)."""
        return len(self._timestamps)


def resolve_timestamp(
    capture_ts: float | None,
    server_recv_ts: float,
) -> tuple[float, bool]:
    """capture timestamp 우선, 누락 시 server 수신 시각으로 fallback.

    docs/2-3-2 §3-3: Android `SystemClock.elapsedRealtimeNanos()` 기반
    capture_ts가 정상 경로. None이면 AI 서버 수신 시각 사용 + fallback 플래그.
    호출자(통합 Preprocessor)가 is_fallback True 시 `quality_flags`에
    `"timestamp_fallback"` 추가.

    Args:
        capture_ts: Android capture timestamp (초). None이면 fallback.
        server_recv_ts: AI 서버 수신 시각 (초).

    Returns:
        (timestamp_sec, is_fallback) — 사용할 timestamp + fallback 사용 여부.
    """
    if capture_ts is None:
        return (server_recv_ts, True)
    return (capture_ts, False)


# ============================================================
# Live mode — Phase 4 frame-level 품질 검사 3종 (stateless)
# ============================================================


def check_brightness(
    frame: np.ndarray,
    threshold: float = 50.0,
) -> bool:
    """평균 휘도 < threshold → True (`low_brightness` 플래그 부여 신호).

    docs/2-3-2 §4-1: grayscale 평균. 정상 80~150, 50 미만은 visibility 광범위
    저하 임계 영역. ⚠️ 파일럿 보정 (FramePreprocessConfig.brightness_min).

    Args:
        frame: BGR `np.ndarray` (H, W, 3).
        threshold: 평균 휘도 임계 (default 50.0, 0~255 스케일).

    Returns:
        True면 `low_brightness` 플래그 부여 대상.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


def check_motion_blur(
    frame: np.ndarray,
    laplacian_var_min: float = 100.0,
) -> bool:
    """Laplacian variance < min → True (`motion_blur` 플래그 부여 신호).

    docs/2-3-2 §4-2: `cv2.Laplacian()` + `np.var()`. 이미지 엣지가 선명할수록
    분산↑, 모션 블러로 뭉개지면 분산↓. 정상 200~1000, 100 미만 블러 임계.
    ⚠️ 파일럿 보정 (FramePreprocessConfig.laplacian_var_min).

    Args:
        frame: BGR `np.ndarray` (H, W, 3).
        laplacian_var_min: Laplacian variance 임계 (default 100.0).

    Returns:
        True면 `motion_blur` 플래그 부여 대상.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var()) < laplacian_var_min


def check_frame_stability(
    current: np.ndarray,
    previous: np.ndarray | None,
    ssd_change_ratio_max: float = 2.0,
    ssd_baseline: float | None = None,
) -> bool:
    """SSD 변화량 > baseline × ratio_max → True (`frame_unstable` 플래그 신호).

    docs/2-3-2 §4-3: 간단 버전 SSD(Sum of Squared Differences). 인접 frame 간
    전체 픽셀 차이. ⚠️ 러닝 자체 움직임을 카메라 흔들림으로 오탐 가능 — 본 함수
    리턴은 "frame-level 이상 변동 신호"로만 해석. 단정 카메라 흔들림 판정은
    stride 단위로 보는 2-3-5가 담당.

    보수 판정 가드 3개 (모두 False 반환):
    - `previous` None (첫 frame, 비교 대상 없음)
    - `ssd_baseline` None (baseline 미확정, Phase 5 통합 Preprocessor가 첫 N
      frame에 baseline 누적 후 전달)
    - `ssd_baseline <= 0` (zero-division 방어)

    Args:
        current: 현재 BGR frame.
        previous: 직전 BGR frame (None이면 첫 frame, False 반환).
        ssd_change_ratio_max: 임계 비율 (default 2.0 = 평균의 200%).
        ssd_baseline: 최근 인접 SSD 평균 (None이면 baseline 미확정, False 반환).

    Returns:
        True면 `frame_unstable` 플래그 부여 대상.
    """
    if previous is None or ssd_baseline is None:
        return False
    # ⚠️ float64 사용 이유: int32 sum 오버플로 방어. 720p (921,600 픽셀) 입력
    # 에서 픽셀당 diff² 최대 40,000 → 총 ≈3.69e10이 int32 max (2.15e9) 초과.
    # float64는 10³⁰⁸ 범위라 4K 영상까지 안전.
    diff = current.astype(np.float64) - previous.astype(np.float64)
    ssd = float(np.sum(diff * diff))
    if ssd_baseline <= 0:
        return False
    return (ssd / ssd_baseline) > ssd_change_ratio_max
