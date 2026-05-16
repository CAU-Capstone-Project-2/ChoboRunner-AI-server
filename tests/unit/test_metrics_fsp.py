"""foot_strike.py unit 테스트 (Phase 5-D-2).

5-D-1 sanity 14 case 이식 + edge case 보강 (5-C-2 패턴 일관).

카테고리 (~20 case):
A. 분류 경계 7 (parametrize): 5.01/4.99/0/-4.99/-5.01/10/-10
B. visibility/vector 가드 2: low_vis / vector_zero
C. IC window 3: full / partial / fail (ic_window_offset=1, 3 frame)
D. direction 대칭 + analysis_side 대칭 2
E. edge 7: empty / None 섞임 / ic_indices=[] / ic at start / ic at end /
   invalid analysis_side / invalid direction

영상 의존 X — 합성 PoseLandmarks (수식 검증).
"""
from __future__ import annotations

import math

import pytest

from choborunner_ai.config import FootStrikeConfig
from choborunner_ai.metrics.foot_strike import (
    classify,
    compute_at_ic,
    compute_series,
    foot_strike_deg,
)
from choborunner_ai.pose_extractor import Landmark, LandmarkPair, PoseLandmarks


@pytest.fixture
def cfg() -> FootStrikeConfig:
    return FootStrikeConfig()


def make_pose(
    theta_foot_deg: float,
    direction: str = "left_to_right",
    analysis_side: str = "left",
    vis_heel: float = 0.9,
    vis_foot: float = 0.9,
    foot_at_heel: bool = False,
) -> PoseLandmarks:
    """5-D-1 sanity 헬퍼 이식 (수학적 정확)."""
    theta_rad = math.radians(theta_foot_deg)
    L = 0.05
    heel_x, heel_y = 0.5, 0.95

    if foot_at_heel:
        foot_x, foot_y = heel_x, heel_y
    else:
        x_sign = 1.0 if direction == "left_to_right" else -1.0
        foot_x = heel_x + x_sign * L * math.cos(theta_rad)
        foot_y = heel_y - L * math.sin(theta_rad)

    def lm(x: float, y: float, v: float) -> Landmark:
        return Landmark(x=x, y=y, visibility=v)

    static = lm(0.5, 0.95, 0.9)

    if analysis_side == "left":
        heel_pair = LandmarkPair(left=lm(heel_x, heel_y, vis_heel), right=static)
        foot_pair = LandmarkPair(left=lm(foot_x, foot_y, vis_foot), right=static)
    else:
        heel_pair = LandmarkPair(left=static, right=lm(heel_x, heel_y, vis_heel))
        foot_pair = LandmarkPair(left=static, right=lm(foot_x, foot_y, vis_foot))

    return PoseLandmarks(
        shoulder=LandmarkPair(left=lm(0.45, 0.25, 0.9), right=lm(0.55, 0.25, 0.9)),
        hip=LandmarkPair(left=lm(0.45, 0.55, 0.9), right=lm(0.55, 0.55, 0.9)),
        knee=LandmarkPair(left=lm(0.45, 0.75, 0.9), right=lm(0.55, 0.75, 0.9)),
        ankle=LandmarkPair(left=lm(0.45, 0.9, 0.9), right=lm(0.55, 0.9, 0.9)),
        heel=heel_pair,
        foot_index=foot_pair,
    )


# ============================================================
# A. 분류 경계 (7 parametrize)
# ============================================================


@pytest.mark.parametrize(
    "theta,expected_cls",
    [
        (5.01, "rfs"),
        (4.99, "mfs"),
        (0.0, "mfs"),
        (-4.99, "mfs"),
        (-5.01, "ffs"),
        (10.0, "rfs"),
        (-10.0, "ffs"),
    ],
)
def test_foot_strike_deg_classification(
    theta: float, expected_cls: str, cfg: FootStrikeConfig
):
    """docs §5-5 strict 경계 (rfs >= +5 / -5 < mfs < +5 / ffs <= -5)."""
    pl = make_pose(theta_foot_deg=theta)
    deg = foot_strike_deg(pl, "left", "left_to_right", cfg)
    assert math.isfinite(deg)
    assert abs(deg - theta) < 1e-3
    assert classify(deg, cfg) == expected_cls


# ============================================================
# B. visibility / vector 가드
# ============================================================


def test_low_visibility_returns_nan(cfg: FootStrikeConfig):
    """heel vis 0.3 → NaN (Uncertain)."""
    pl = make_pose(theta_foot_deg=10.0, vis_heel=0.3)
    deg = foot_strike_deg(pl, "left", "left_to_right", cfg)
    assert not math.isfinite(deg)
    assert classify(deg, cfg) is None


def test_vector_zero_returns_nan(cfg: FootStrikeConfig):
    """foot == heel → NaN."""
    pl = make_pose(theta_foot_deg=10.0, foot_at_heel=True)
    deg = foot_strike_deg(pl, "left", "left_to_right", cfg)
    assert not math.isfinite(deg)


# ============================================================
# C. IC window (ic_window_offset=1, 3 frame)
# ============================================================


def test_ic_window_full_valid(cfg: FootStrikeConfig):
    series = [make_pose(theta_foot_deg=10.0) for _ in range(10)]
    results = compute_at_ic(
        series, ic_indices=[5], analysis_side="left",
        direction="left_to_right", cfg=cfg,
    )
    r = results[0]
    assert r.is_valid
    assert abs(r.deg - 10.0) < 1e-3
    assert r.window_total_count == 3
    assert r.window_valid_count == 3
    assert r.classification == "rfs"


def test_ic_window_partial_valid(cfg: FootStrikeConfig):
    """2 valid + 1 invalid (2/3 >= 0.5)."""
    series = []
    for i in range(10):
        if i == 6:
            series.append(make_pose(theta_foot_deg=10.0, vis_heel=0.3))
        else:
            series.append(make_pose(theta_foot_deg=10.0))
    results = compute_at_ic(
        series, ic_indices=[5], analysis_side="left",
        direction="left_to_right", cfg=cfg,
    )
    r = results[0]
    assert r.is_valid
    assert r.window_valid_count == 2
    assert r.window_total_count == 3


def test_ic_window_fail(cfg: FootStrikeConfig):
    """1 valid + 2 invalid (1/3 < 0.5)."""
    series = []
    for i in range(10):
        if i in (4, 6):
            series.append(make_pose(theta_foot_deg=10.0, vis_heel=0.3))
        else:
            series.append(make_pose(theta_foot_deg=10.0))
    results = compute_at_ic(
        series, ic_indices=[5], analysis_side="left",
        direction="left_to_right", cfg=cfg,
    )
    r = results[0]
    assert not r.is_valid
    assert not math.isfinite(r.deg)
    assert r.classification is None


# ============================================================
# D. direction + analysis_side 대칭
# ============================================================


def test_direction_symmetry(cfg: FootStrikeConfig):
    """같은 θ_foot 자세 + 두 direction → 동일 결과."""
    pl_l2r = make_pose(theta_foot_deg=10.0, direction="left_to_right")
    pl_r2l = make_pose(theta_foot_deg=10.0, direction="right_to_left")
    deg_l = foot_strike_deg(pl_l2r, "left", "left_to_right", cfg)
    deg_r = foot_strike_deg(pl_r2l, "left", "right_to_left", cfg)
    assert abs(deg_l - deg_r) < 1e-9
    assert classify(deg_l, cfg) == classify(deg_r, cfg)


def test_analysis_side_symmetry(cfg: FootStrikeConfig):
    pl_left = make_pose(theta_foot_deg=10.0, analysis_side="left")
    pl_right = make_pose(theta_foot_deg=10.0, analysis_side="right")
    deg_l = foot_strike_deg(pl_left, "left", "left_to_right", cfg)
    deg_r = foot_strike_deg(pl_right, "right", "left_to_right", cfg)
    assert abs(deg_l - deg_r) < 1e-9


# ============================================================
# E. edge case
# ============================================================


def test_compute_series_empty(cfg: FootStrikeConfig):
    assert compute_series([], "left", "left_to_right", cfg) == []


def test_compute_series_with_none(cfg: FootStrikeConfig):
    series: list = [
        make_pose(theta_foot_deg=10.0),
        None,
        make_pose(theta_foot_deg=10.0),
    ]
    out = compute_series(series, "left", "left_to_right", cfg)
    assert len(out) == 3
    assert math.isfinite(out[0])
    assert not math.isfinite(out[1])
    assert math.isfinite(out[2])


def test_compute_at_ic_empty_indices(cfg: FootStrikeConfig):
    series = [make_pose(theta_foot_deg=10.0) for _ in range(10)]
    assert compute_at_ic(
        series, ic_indices=[], analysis_side="left",
        direction="left_to_right", cfg=cfg,
    ) == []


def test_compute_at_ic_video_start(cfg: FootStrikeConfig):
    """IC=0 → window [0..1] (2 frame, offset=1)."""
    series = [make_pose(theta_foot_deg=10.0) for _ in range(10)]
    results = compute_at_ic(
        series, ic_indices=[0], analysis_side="left",
        direction="left_to_right", cfg=cfg,
    )
    r = results[0]
    assert r.window_total_count == 2  # [0, 1]
    assert r.window_valid_count == 2
    assert r.is_valid


def test_compute_at_ic_video_end(cfg: FootStrikeConfig):
    """IC=N-1 → window [N-2..N-1] (2 frame)."""
    series = [make_pose(theta_foot_deg=10.0) for _ in range(10)]
    results = compute_at_ic(
        series, ic_indices=[9], analysis_side="left",
        direction="left_to_right", cfg=cfg,
    )
    r = results[0]
    assert r.window_total_count == 2
    assert r.window_valid_count == 2
    assert r.is_valid


def test_analysis_side_invalid_raises(cfg: FootStrikeConfig):
    pl = make_pose(theta_foot_deg=10.0)
    with pytest.raises(ValueError, match="analysis_side"):
        foot_strike_deg(pl, "top", "left_to_right", cfg)  # type: ignore[arg-type]


def test_direction_invalid_raises(cfg: FootStrikeConfig):
    pl = make_pose(theta_foot_deg=10.0)
    with pytest.raises(ValueError, match="direction"):
        foot_strike_deg(pl, "left", "diagonal", cfg)  # type: ignore[arg-type]
