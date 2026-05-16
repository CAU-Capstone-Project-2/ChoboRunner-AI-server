"""trunk_lean.py unit 테스트 (Phase 5-A-2).

5-A-1 sanity 9 case 이식 + 부동소수점 안전 경계 4건 + ic edge 3건 = 16 case.

카테고리:
A. 단일 frame trunk_lean_deg + classify (parametrize 5 + vis 1 = 6)
B. 부동소수점 안전 경계 (parametrize 4)
C. compute_at_ic (window full / partial / fail = 3)
D. compute_at_ic edge case (ic_indices [] / [0] / [N-1] = 3)

영상 파일 의존 X — 합성 PoseLandmarks (수식 검증).
"""
from __future__ import annotations

import math

import pytest

from choborunner_ai.config import TrunkLeanConfig
from choborunner_ai.metrics.trunk_lean import (
    classify,
    compute_at_ic,
    trunk_lean_deg,
)
from choborunner_ai.pose_extractor import Landmark, LandmarkPair, PoseLandmarks


@pytest.fixture
def cfg() -> TrunkLeanConfig:
    """default TrunkLeanConfig."""
    return TrunkLeanConfig()


def make_pose(
    theta_deg: float,
    vis_shoulder: float = 0.9,
    vis_hip: float = 0.9,
    h: float = 0.3,
) -> PoseLandmarks:
    """θ° 전방 기울기 PoseLandmarks 합성 (5-A-1 sanity 헬퍼 이식).

    수식: shoulder_center.x = hip_center.x + h*tan(θ), .y = hip_center.y - h.
    """
    theta_rad = math.radians(theta_deg)
    hip_x = 0.5
    hip_y = 0.7
    sc_x = hip_x + h * math.tan(theta_rad)
    sc_y = hip_y - h

    def lm(x: float, y: float, vis: float) -> Landmark:
        return Landmark(x=x, y=y, visibility=vis)

    return PoseLandmarks(
        shoulder=LandmarkPair(
            left=lm(sc_x - 0.05, sc_y, vis_shoulder),
            right=lm(sc_x + 0.05, sc_y, vis_shoulder),
        ),
        hip=LandmarkPair(
            left=lm(hip_x - 0.05, hip_y, vis_hip),
            right=lm(hip_x + 0.05, hip_y, vis_hip),
        ),
        knee=LandmarkPair(left=lm(0.5, 0.8, 0.9), right=lm(0.5, 0.8, 0.9)),
        ankle=LandmarkPair(left=lm(0.5, 0.9, 0.9), right=lm(0.5, 0.9, 0.9)),
        heel=LandmarkPair(left=lm(0.5, 0.95, 0.9), right=lm(0.5, 0.95, 0.9)),
        foot_index=LandmarkPair(left=lm(0.5, 0.98, 0.9), right=lm(0.5, 0.98, 0.9)),
    )


# ============================================================
# A. 단일 frame + classify (5 각도 + 1 vis)
# ============================================================


@pytest.mark.parametrize(
    "theta,expected_cls",
    [
        (0.0, "near_vertical"),
        (5.5, "forward_lean"),
        (9.5, "forward_lean"),
        (15.0, "above_typical"),
        (-3.0, "near_vertical"),
    ],
)
def test_trunk_lean_deg_basic(theta: float, expected_cls: str, cfg: TrunkLeanConfig):
    """기본 각도 5건 + 분류 검증 (5-A-1 sanity 이식, 안전 경계 5.5°/9.5°)."""
    pl = make_pose(theta_deg=theta)
    deg = trunk_lean_deg(pl, cfg)
    assert math.isfinite(deg)
    assert abs(deg - theta) < 1e-3
    assert classify(deg, cfg) == expected_cls


def test_low_visibility_returns_nan(cfg: TrunkLeanConfig):
    """shoulder vis 0.3 < cfg.visibility_min(0.6) → NaN + classify None."""
    pl = make_pose(theta_deg=5.5, vis_shoulder=0.3)
    deg = trunk_lean_deg(pl, cfg)
    assert not math.isfinite(deg)
    assert classify(deg, cfg) is None


# ============================================================
# B. 부동소수점 안전 경계 (4.99 / 5.01 / 9.99 / 10.01)
# ============================================================


@pytest.mark.parametrize(
    "theta,expected_cls",
    [
        (4.99, "near_vertical"),    # < 5.0
        (5.01, "forward_lean"),     # 5.0 < θ ≤ 10.0 (정확 5.0° 경계는 atan2 ulp 한계로 회피)
        (9.99, "forward_lean"),     # ≤ 10.0
        (10.01, "above_typical"),   # > 10.0
    ],
)
def test_classify_boundary(theta: float, expected_cls: str, cfg: TrunkLeanConfig):
    """부동소수점 안전 경계 4건 — docs §7-6 strict 비교 정합."""
    pl = make_pose(theta_deg=theta)
    deg = trunk_lean_deg(pl, cfg)
    assert math.isfinite(deg)
    assert abs(deg - theta) < 1e-3
    assert classify(deg, cfg) == expected_cls


# ============================================================
# C. compute_at_ic (window full / partial / fail)
# ============================================================


def test_ic_window_full_valid(cfg: TrunkLeanConfig):
    """5 frame window all valid → is_valid=True, 5/5."""
    series = [make_pose(theta_deg=5.5) for _ in range(10)]
    results = compute_at_ic(series, ic_indices=[5], cfg=cfg)
    assert len(results) == 1
    r = results[0]
    assert r.is_valid
    assert abs(r.deg - 5.5) < 1e-3
    assert r.window_valid_count == 5
    assert r.window_total_count == 5
    assert r.classification == "forward_lean"


def test_ic_window_partial_valid(cfg: TrunkLeanConfig):
    """3 valid + 2 invalid (3/5 = 0.6 ≥ 0.5) → is_valid=True."""
    series: list = []
    for i in range(10):
        if i in (4, 6):  # IC=5, window=[3,4,5,6,7], 4 and 6 invalid
            series.append(make_pose(theta_deg=5.5, vis_shoulder=0.3))
        else:
            series.append(make_pose(theta_deg=5.5))
    results = compute_at_ic(series, ic_indices=[5], cfg=cfg)
    r = results[0]
    assert r.is_valid
    assert r.window_valid_count == 3
    assert r.window_total_count == 5
    assert abs(r.deg - 5.5) < 1e-3


def test_ic_window_fail(cfg: TrunkLeanConfig):
    """1 valid + 4 invalid (1/5 = 0.2 < 0.5) → is_valid=False, NaN."""
    series: list = []
    for i in range(10):
        if i in (3, 4, 6, 7):  # IC=5, window=[3..7], 4 invalid
            series.append(make_pose(theta_deg=5.5, vis_shoulder=0.3))
        else:
            series.append(make_pose(theta_deg=5.5))
    results = compute_at_ic(series, ic_indices=[5], cfg=cfg)
    r = results[0]
    assert not r.is_valid
    assert not math.isfinite(r.deg)
    assert r.classification is None
    assert r.window_valid_count == 1
    assert r.window_total_count == 5


# ============================================================
# D. compute_at_ic edge case (ic_indices [] / [0] / [N-1])
# ============================================================


def test_ic_indices_empty(cfg: TrunkLeanConfig):
    """ic_indices=[] → 빈 list 반환 (호출자 입력 그대로 transpose)."""
    series = [make_pose(theta_deg=5.5) for _ in range(10)]
    results = compute_at_ic(series, ic_indices=[], cfg=cfg)
    assert results == []


def test_ic_at_video_start(cfg: TrunkLeanConfig):
    """IC=0 (영상 시작) → window 축소 [0..2] (3 frame), 축소 모집단 100% valid."""
    series = [make_pose(theta_deg=5.5) for _ in range(10)]
    results = compute_at_ic(series, ic_indices=[0], cfg=cfg)
    r = results[0]
    assert r.window_total_count == 3  # [0, 1, 2]
    assert r.window_valid_count == 3
    assert r.is_valid
    assert abs(r.deg - 5.5) < 1e-3


def test_ic_at_video_end(cfg: TrunkLeanConfig):
    """IC=N-1 (영상 끝) → window 축소 [N-3..N-1] (3 frame)."""
    series = [make_pose(theta_deg=5.5) for _ in range(10)]
    results = compute_at_ic(series, ic_indices=[9], cfg=cfg)  # N=10, N-1=9
    r = results[0]
    assert r.window_total_count == 3  # [7, 8, 9]
    assert r.window_valid_count == 3
    assert r.is_valid
    assert abs(r.deg - 5.5) < 1e-3
