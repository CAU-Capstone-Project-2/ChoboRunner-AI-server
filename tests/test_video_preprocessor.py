"""video_preprocessor.py 통합 테스트 (Phase 6).

설계문서 docs/2-3-2 + Phase 1~5 자가 검토 학습 검증.
input_validator의 tests/test_input_validator.py 구조 그대로.

카테고리:
A. 헬퍼·기본 (5)
B. 정규화 함수 (8) — decode/resolve/grid
C. 품질 검사 함수 (6) — brightness/motion_blur/stability
D. FpsTracker + resolve_timestamp (6)
E. Preprocessor 통합 (10)

영상 파일 의존 X — 합성 JPEG bytes 생성 helper로 자기충족적.
"""
from __future__ import annotations

import typing

import cv2
import numpy as np
import pytest

from choborunner_ai.config import AppConfig, FramePreprocessConfig
from choborunner_ai.video_preprocessor import (
    FpsTracker,
    Preprocessor,
    ProcessedFrame,
    QualityFlag,
    check_brightness,
    check_frame_stability,
    check_motion_blur,
    decode_jpeg_binary,
    normalize_resolution,
    resolve_timestamp,
    should_select_frame_for_fps_grid,
)


# ============================================================
# Fixtures + helpers
# ============================================================


@pytest.fixture
def cfg() -> FramePreprocessConfig:
    """default 임계값 FramePreprocessConfig."""
    return FramePreprocessConfig()


@pytest.fixture
def app_cfg() -> AppConfig:
    """default AppConfig."""
    return AppConfig()


def make_jpeg(value: int = 100, h: int = 480, w: int = 640) -> bytes:
    """단조 색상 JPEG bytes. flat → Laplacian variance ~0 → motion_blur 트리거."""
    img = np.full((h, w, 3), value, dtype=np.uint8)
    success, buf = cv2.imencode(".jpg", img)
    assert success
    return buf.tobytes()


def make_jpeg_noisy(h: int = 480, w: int = 640, seed: int = 42) -> bytes:
    """random noise JPEG bytes. variance 큼 → motion_blur 회피용."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    success, buf = cv2.imencode(".jpg", img)
    assert success
    return buf.tobytes()


# ============================================================
# Group A — 헬퍼·기본 (5)
# ============================================================


def test_make_jpeg_roundtrip():
    jpeg = make_jpeg(100, 480, 640)
    img = decode_jpeg_binary(jpeg)
    assert img is not None
    assert img.shape == (480, 640, 3)


def test_processed_frame_dataclass_fields():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    pf = ProcessedFrame(frame_index=0, timestamp_sec=0.0, image=img)
    assert pf.frame_index == 0
    assert pf.timestamp_sec == 0.0
    assert pf.image is img
    assert pf.quality_flags == []
    assert pf.fps_actual_recent == 0.0


def test_quality_flag_literal_values():
    args = typing.get_args(QualityFlag)
    assert "low_brightness" in args
    assert "motion_blur" in args
    assert "frame_unstable" in args
    assert "timestamp_fallback" in args
    assert len(args) == 4


def test_fps_tracker_default_window(cfg):
    assert cfg.fps_tracker_window == 30


def test_frame_preprocess_config_defaults(cfg):
    assert cfg.fps_cap == 30.0
    assert cfg.resolution_long_side_cap == 1280
    assert cfg.fps_tracker_window == 30
    assert cfg.brightness_min == 50.0
    assert cfg.laplacian_var_min == 100.0
    assert cfg.ssd_change_ratio_max == 2.0


# ============================================================
# Group B — 정규화 함수 (8)
# ============================================================


def test_decode_jpeg_empty_bytes():
    assert decode_jpeg_binary(b"") is None


def test_decode_jpeg_garbage_bytes():
    assert decode_jpeg_binary(b"\x00" * 10) is None


def test_decode_jpeg_valid():
    jpeg = make_jpeg(100, 480, 640)
    img = decode_jpeg_binary(jpeg)
    assert img is not None
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_normalize_resolution_downsample():
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    out = normalize_resolution(big, 1280)
    assert max(out.shape[:2]) == 1280
    assert out.shape[1] == 1280
    assert out.shape[0] == 720


def test_normalize_resolution_unchanged():
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    out = normalize_resolution(small, 1280)
    assert out.shape == small.shape


def test_normalize_resolution_exactly_cap():
    exact = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = normalize_resolution(exact, 1280)
    assert out.shape == exact.shape


def test_should_select_first_frame_true():
    assert should_select_frame_for_fps_grid(0.0, None, 30.0) is True


def test_should_select_30fps_grid_exact():
    """1e-9 tolerance fix 검증 — i/30 누적 10 frame 모두 채택."""
    last = None
    selected = 0
    for i in range(10):
        ts = i / 30.0
        if should_select_frame_for_fps_grid(ts, last, 30.0):
            selected += 1
            last = ts
    assert selected == 10


# ============================================================
# Group C — 품질 검사 함수 (6)
# ============================================================


def test_brightness_dark_frame():
    dark = np.zeros((480, 640, 3), dtype=np.uint8)
    assert check_brightness(dark, 50.0) is True


def test_brightness_threshold_boundary():
    mid = np.full((480, 640, 3), 50, dtype=np.uint8)
    assert check_brightness(mid, 50.0) is False


def test_brightness_bright_frame():
    bright = np.full((480, 640, 3), 255, dtype=np.uint8)
    assert check_brightness(bright, 50.0) is False


def test_motion_blur_flat_frame():
    flat = np.zeros((480, 640, 3), dtype=np.uint8)
    assert check_motion_blur(flat, 100.0) is True


def test_motion_blur_noisy_frame():
    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    assert check_motion_blur(noisy, 100.0) is False


def test_frame_stability_none_guards():
    curr = np.zeros((10, 10, 3), dtype=np.uint8)
    prev = np.zeros((10, 10, 3), dtype=np.uint8)
    # previous None → False
    assert check_frame_stability(curr, None, 2.0, 100.0) is False
    # baseline None → False
    assert check_frame_stability(curr, prev, 2.0, None) is False


# ============================================================
# Group D — FpsTracker + resolve_timestamp (6)
# ============================================================


def test_fps_tracker_empty():
    t = FpsTracker(window=30)
    assert t.fps_recent == 0.0
    assert t.size == 0


def test_fps_tracker_30fps():
    t = FpsTracker(window=30)
    for i in range(10):
        t.add(i / 30.0)
    assert 29.9 < t.fps_recent < 30.1


def test_fps_tracker_window_cap():
    t = FpsTracker(window=30)
    for i in range(35):
        t.add(i / 30.0)
    assert t.size == 30


def test_fps_tracker_zero_division():
    t = FpsTracker(window=30)
    for _ in range(5):
        t.add(1.0)
    assert t.fps_recent == 0.0


def test_resolve_timestamp_normal():
    ts, fallback = resolve_timestamp(0.5, 0.6)
    assert ts == 0.5
    assert fallback is False


def test_resolve_timestamp_fallback():
    ts, fallback = resolve_timestamp(None, 0.6)
    assert ts == 0.6
    assert fallback is True


# ============================================================
# Group E — Preprocessor 통합 (10)
# ============================================================


def test_preprocessor_first_frame(cfg):
    p = Preprocessor(cfg)
    pf = p.preprocess_frame(make_jpeg_noisy(), 0.0, 0.0)
    assert pf is not None
    assert pf.frame_index == 0
    assert pf.timestamp_sec == 0.0
    assert "timestamp_fallback" not in pf.quality_flags


def test_preprocessor_decode_failure_counter(cfg):
    p = Preprocessor(cfg)
    result = p.preprocess_frame(b"", 0.0, 0.0)
    assert result is None
    assert p._total_counter == 0


def test_preprocessor_fps_grid_skip_60fps(cfg):
    p = Preprocessor(cfg)
    pf_a = p.preprocess_frame(make_jpeg_noisy(), 0.0, 0.0)
    pf_b = p.preprocess_frame(make_jpeg_noisy(), 1.0 / 60.0, 0.0)
    assert pf_a is not None
    assert pf_b is None


def test_preprocessor_fps_grid_skip_no_counter(cfg):
    p = Preprocessor(cfg)
    p.preprocess_frame(make_jpeg_noisy(), 0.0, 0.0)
    initial_counter = p._total_counter
    p.preprocess_frame(make_jpeg_noisy(), 1.0 / 60.0, 0.0)  # skip
    assert p._total_counter == initial_counter


def test_preprocessor_fallback_flag(cfg):
    p = Preprocessor(cfg)
    pf = p.preprocess_frame(make_jpeg_noisy(), None, 0.0)
    assert pf is not None
    assert "timestamp_fallback" in pf.quality_flags
    assert p.fallback_ratio == 1.0


def test_preprocessor_fallback_ratio_5_of_10(cfg):
    p = Preprocessor(cfg)
    for i in range(10):
        capture = None if i < 5 else (i / 30.0)
        p.preprocess_frame(make_jpeg_noisy(), capture, i / 30.0)
    assert p._total_counter == 10
    assert abs(p.fallback_ratio - 0.5) < 0.01


def test_preprocessor_ssd_baseline_warmup(cfg):
    """첫 frame baseline None — frame_unstable 항상 False."""
    p = Preprocessor(cfg)
    pf_first = p.preprocess_frame(make_jpeg_noisy(), 0.0, 0.0)
    assert pf_first is not None
    assert "frame_unstable" not in pf_first.quality_flags


def test_preprocessor_ssd_baseline_accumulation(cfg):
    """다중 noisy frame 후 baseline history 누적 검증."""
    p = Preprocessor(cfg)
    for i in range(5):
        p.preprocess_frame(make_jpeg_noisy(seed=i), i / 30.0, i / 30.0)
    # 5 frame 처리, 첫 frame은 prev None이라 push X.
    # 2~5 frame은 baseline mean 변동에 따라 unstable 여부에 의존하지만,
    # noisy 변동이 일반적으로 평균 근처라 ratio_max=2.0 안에서 stable 다수.
    assert len(p._ssd_history) >= 1


def test_preprocessor_unstable_skips_baseline_update():
    """unstable 판정 시 history 갱신 X (오탐 자기 강화 방지)."""
    # 매우 작은 ratio_max로 거의 모든 변동을 unstable로 트리거
    custom_cfg = FramePreprocessConfig(ssd_change_ratio_max=0.001)
    p = Preprocessor(custom_cfg)
    # frame 1: prev None → unstable False, history empty 유지
    p.preprocess_frame(make_jpeg(100), 0.0, 0.0)
    # frame 2: history empty → baseline None → unstable False, history +1
    p.preprocess_frame(make_jpeg(101), 1.0 / 30.0, 1.0 / 30.0)
    history_after_frame2 = list(p._ssd_history)
    # frame 3: history 있고 ratio_max=0.001 빡빡 → 큰 변동 unstable True
    pf3 = p.preprocess_frame(make_jpeg(200), 2.0 / 30.0, 2.0 / 30.0)
    assert pf3 is not None
    assert "frame_unstable" in pf3.quality_flags
    assert list(p._ssd_history) == history_after_frame2, "Unstable 시 history 갱신 안 됨"


def test_preprocessor_image_shape_after_resize(cfg):
    """1080x1920 입력 → 720x1280으로 resize 적용 확인."""
    p = Preprocessor(cfg)
    pf = p.preprocess_frame(make_jpeg_noisy(h=1080, w=1920), 0.0, 0.0)
    assert pf is not None
    assert max(pf.image.shape[:2]) == 1280
