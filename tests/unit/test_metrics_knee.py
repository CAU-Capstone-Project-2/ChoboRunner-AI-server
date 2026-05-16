"""knee_flexion.py unit 테스트 (Phase 5-C-2).

5-C-1 sanity 12 case 이식 + edge case 보강:
A. 분류 경계 6 (parametrize): 0° / 14.99° / 15.01° / 20° / 24.99° / 25.01°
B. visibility/vector 가드 2: low_vis / vector_zero
C. IC window 3: full / partial / fail
D. analysis_side 2: 좌우 대칭 / invalid (ValueError)
E. edge 5 보강: empty / None 섞임 / ic_indices=[] / ic at start / ic at end

영상 의존 X — 합성 PoseLandmarks (수식 검증).

⚠️ catch (trunk_lean과 차이, 압축 모드라도 박힌 자산):
- 분류 경계 strict (< / < / >=) — docs §6-6 정합
- arccos vector 2개 둘 다 가드 (v1=hip-knee, v2=ankle-knee)
"""
from __future__ import annotations

import math

import pytest

from choborunner_ai.config import KneeFlexionConfig
from choborunner_ai.metrics.knee_flexion import (
    classify,
    compute_at_ic,
    compute_series,
    knee_flexion_deg,
)
from choborunner_ai.pose_extractor import Landmark, LandmarkPair, PoseLandmarks


@pytest.fixture
def cfg() -> KneeFlexionConfig:
    return KneeFlexionConfig()


def make_pose(
    theta_kf_deg: float,
    vis_hip: float = 0.9,
    vis_knee: float = 0.9,
    vis_ankle: float = 0.9,
    analysis_side: str = "left",
    hip_at_knee: bool = False,
) -> PoseLandmarks:
    """5-C-1 sanity 헬퍼 이식 (수학적 정확).

    knee = (0.5, 0.7), hip = (0.5, 0.55), ankle = (0.5 + 0.2*sin θ, 0.7 + 0.2*cos θ)
    → knee_flexion 정확히 θ_kf와 일치.
    """
    theta_rad = math.radians(theta_kf_deg)
    knee_x, knee_y = 0.5, 0.7
    if hip_at_knee:
        hip_x, hip_y = knee_x, knee_y
    else:
        hip_x, hip_y = 0.5, 0.55
    ankle_x = 0.5 + 0.2 * math.sin(theta_rad)
    ankle_y = 0.7 + 0.2 * math.cos(theta_rad)

    def lm(x: float, y: float, v: float) -> Landmark:
        return Landmark(x=x, y=y, visibility=v)

    if analysis_side == "left":
        hip_pair = LandmarkPair(left=lm(hip_x, hip_y, vis_hip), right=lm(0.55, 0.55, 0.9))
        knee_pair = LandmarkPair(left=lm(knee_x, knee_y, vis_knee), right=lm(0.55, 0.7, 0.9))
        ankle_pair = LandmarkPair(left=lm(ankle_x, ankle_y, vis_ankle), right=lm(0.55, 0.85, 0.9))
    else:
        hip_pair = LandmarkPair(left=lm(0.45, 0.55, 0.9), right=lm(hip_x, hip_y, vis_hip))
        knee_pair = LandmarkPair(left=lm(0.45, 0.7, 0.9), right=lm(knee_x, knee_y, vis_knee))
        ankle_pair = LandmarkPair(left=lm(0.45, 0.85, 0.9), right=lm(ankle_x, ankle_y, vis_ankle))

    return PoseLandmarks(
        shoulder=LandmarkPair(left=lm(0.45, 0.25, 0.9), right=lm(0.55, 0.25, 0.9)),
        hip=hip_pair,
        knee=knee_pair,
        ankle=ankle_pair,
        heel=LandmarkPair(left=lm(0.5, 0.95, 0.9), right=lm(0.5, 0.95, 0.9)),
        foot_index=LandmarkPair(left=lm(0.52, 0.97, 0.9), right=lm(0.52, 0.97, 0.9)),
    )


# ============================================================
# A. 분류 경계 6 (parametrize)
# ============================================================


@pytest.mark.parametrize(
    "theta,expected_cls",
    [
        (0.0, "below_typical"),
        (14.99, "below_typical"),
        (15.01, "typical"),
        (20.0, "typical"),
        (24.99, "typical"),
        (25.01, "above_typical"),
    ],
)
def test_knee_flexion_deg_classification(
    theta: float, expected_cls: str, cfg: KneeFlexionConfig
):
    """docs §6-6 strict 경계 (< 15 / 15 ≤ kf < 25 / ≥ 25)."""
    pl = make_pose(theta_kf_deg=theta)
    deg = knee_flexion_deg(pl, "left", cfg)
    assert math.isfinite(deg)
    assert abs(deg - theta) < 1e-3
    assert classify(deg, cfg) == expected_cls


# ============================================================
# B. visibility / vector 가드
# ============================================================


def test_low_visibility_returns_nan(cfg: KneeFlexionConfig):
    """knee vis 0.3 < cfg.visibility_min 0.6 → NaN."""
    pl = make_pose(theta_kf_deg=20.0, vis_knee=0.3)
    deg = knee_flexion_deg(pl, "left", cfg)
    assert not math.isfinite(deg)
    assert classify(deg, cfg) is None


def test_vector_zero_returns_nan(cfg: KneeFlexionConfig):
    """hip == knee 좌표 (v1 길이 ~0) → NaN.

    catch: trunk는 1 vector, knee는 2 vector 가드.
    """
    pl = make_pose(theta_kf_deg=20.0, hip_at_knee=True)
    deg = knee_flexion_deg(pl, "left", cfg)
    assert not math.isfinite(deg)


# ============================================================
# C. IC window
# ============================================================


def test_ic_window_full_valid(cfg: KneeFlexionConfig):
    series = [make_pose(theta_kf_deg=20.0) for _ in range(10)]
    results = compute_at_ic(series, ic_indices=[5], analysis_side="left", cfg=cfg)
    r = results[0]
    assert r.is_valid
    assert abs(r.deg - 20.0) < 1e-3
    assert r.window_valid_count == 5
    assert r.window_total_count == 5
    assert r.classification == "typical"


def test_ic_window_partial_valid(cfg: KneeFlexionConfig):
    """3 valid + 2 invalid (3/5=0.6 ≥ 0.5) → is_valid=True."""
    series = []
    for i in range(10):
        if i in (4, 6):
            series.append(make_pose(theta_kf_deg=20.0, vis_knee=0.3))
        else:
            series.append(make_pose(theta_kf_deg=20.0))
    results = compute_at_ic(series, ic_indices=[5], analysis_side="left", cfg=cfg)
    r = results[0]
    assert r.is_valid
    assert r.window_valid_count == 3
    assert r.window_total_count == 5
    assert abs(r.deg - 20.0) < 1e-3


def test_ic_window_fail(cfg: KneeFlexionConfig):
    """1 valid + 4 invalid (1/5=0.2 < 0.5) → is_valid=False, NaN."""
    series = []
    for i in range(10):
        if i in (3, 4, 6, 7):
            series.append(make_pose(theta_kf_deg=20.0, vis_knee=0.3))
        else:
            series.append(make_pose(theta_kf_deg=20.0))
    results = compute_at_ic(series, ic_indices=[5], analysis_side="left", cfg=cfg)
    r = results[0]
    assert not r.is_valid
    assert not math.isfinite(r.deg)
    assert r.classification is None


# ============================================================
# D. analysis_side
# ============================================================


def test_analysis_side_symmetry(cfg: KneeFlexionConfig):
    """make_pose는 analysis_side에 따라 좌표 분배 → 동일 θ 동일 결과."""
    pl_left = make_pose(theta_kf_deg=20.0, analysis_side="left")
    pl_right = make_pose(theta_kf_deg=20.0, analysis_side="right")
    deg_l = knee_flexion_deg(pl_left, "left", cfg)
    deg_r = knee_flexion_deg(pl_right, "right", cfg)
    assert abs(deg_l - deg_r) < 1e-9


def test_analysis_side_invalid_raises(cfg: KneeFlexionConfig):
    """analysis_side='top' → ValueError."""
    pl = make_pose(theta_kf_deg=20.0)
    with pytest.raises(ValueError, match="analysis_side"):
        knee_flexion_deg(pl, "top", cfg)  # type: ignore[arg-type]


# ============================================================
# E. edge case 보강
# ============================================================


def test_compute_series_empty(cfg: KneeFlexionConfig):
    """빈 입력 → 빈 list."""
    assert compute_series([], "left", cfg) == []


def test_compute_series_with_none(cfg: KneeFlexionConfig):
    """None 섞임 → NaN, valid는 정상 값."""
    series: list = [
        make_pose(theta_kf_deg=20.0),
        None,
        make_pose(theta_kf_deg=20.0),
    ]
    out = compute_series(series, "left", cfg)
    assert len(out) == 3
    assert math.isfinite(out[0])
    assert not math.isfinite(out[1])
    assert math.isfinite(out[2])


def test_compute_at_ic_empty_indices(cfg: KneeFlexionConfig):
    """ic_indices=[] → 빈 list."""
    series = [make_pose(theta_kf_deg=20.0) for _ in range(10)]
    assert compute_at_ic(series, ic_indices=[], analysis_side="left", cfg=cfg) == []


def test_compute_at_ic_video_start(cfg: KneeFlexionConfig):
    """IC=0 (영상 시작) → window 축소 [0..2] (3 frame), 축소 모집단 100% valid."""
    series = [make_pose(theta_kf_deg=20.0) for _ in range(10)]
    results = compute_at_ic(series, ic_indices=[0], analysis_side="left", cfg=cfg)
    r = results[0]
    assert r.window_total_count == 3
    assert r.window_valid_count == 3
    assert r.is_valid
    assert abs(r.deg - 20.0) < 1e-3


def test_compute_at_ic_video_end(cfg: KneeFlexionConfig):
    """IC=N-1 (영상 끝) → window 축소 [N-3..N-1] (3 frame)."""
    series = [make_pose(theta_kf_deg=20.0) for _ in range(10)]
    results = compute_at_ic(series, ic_indices=[9], analysis_side="left", cfg=cfg)
    r = results[0]
    assert r.window_total_count == 3
    assert r.window_valid_count == 3
    assert r.is_valid
    assert abs(r.deg - 20.0) < 1e-3
