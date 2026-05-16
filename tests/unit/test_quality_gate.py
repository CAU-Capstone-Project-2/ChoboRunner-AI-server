"""quality_gate.py unit 테스트 — docs/2-3-5 §5-2 + §5-3 (Phase 8-A).

본 파일은 Phase 8-A 신규 — `tmp/phase_8_a_sanity.py` 8 case 이식 + pytest 형식.

⚠️ §5-1 (evaluate_frame_visibility / evaluate_visibility_accumulation, Phase 4)는
   본 파일에서 다루지 않음 (Phase 4는 sanity script만 작성 — scripts/sanity/
   phase_4_c_integration.py). 향후 §5-1 pytest 이식은 별도 cleanup 후보.

8 case:
- A. evaluate_frame_body_inclusion (3)
- B. evaluate_frame_foot_cutoff (2)
- C. evaluate_body_inclusion_accumulation (2)
- D. evaluate_foot_cutoff_accumulation (1)
"""
from __future__ import annotations

import pytest

from choborunner_ai.config import VisibilityCheckConfig
from choborunner_ai.pose_extractor import Landmark, LandmarkPair, PoseLandmarks
from choborunner_ai.quality_gate import (
    FrameGeometryResult,
    evaluate_body_inclusion_accumulation,
    evaluate_foot_cutoff_accumulation,
    evaluate_frame_body_inclusion,
    evaluate_frame_foot_cutoff,
)


def _lm(x: float, y: float, vis: float = 0.9) -> Landmark:
    return Landmark(x=x, y=y, visibility=vis)


def _pair(x: float, y: float, vis: float = 0.9) -> LandmarkPair:
    return LandmarkPair(left=_lm(x - 0.05, y, vis), right=_lm(x + 0.05, y, vis))


def _normal_pl(
    nose_vis: float = 0.9, shoulder_x_offset: float = 0.0
) -> PoseLandmarks:
    """13점 정상 PoseLandmarks. shoulder_x_offset으로 좌표 out-of-range 케이스 생성."""
    return PoseLandmarks(
        shoulder=LandmarkPair(
            left=_lm(0.45 + shoulder_x_offset, 0.20),
            right=_lm(0.55 + shoulder_x_offset, 0.20),
        ),
        hip=_pair(0.50, 0.45),
        knee=_pair(0.50, 0.65),
        ankle=_pair(0.50, 0.85),
        heel=_pair(0.48, 0.88),
        foot_index=_pair(0.52, 0.88),
        nose=_lm(0.50, 0.10, nose_vis),
    )


@pytest.fixture
def cfg() -> VisibilityCheckConfig:
    return VisibilityCheckConfig()


# ============================================================
# A. evaluate_frame_body_inclusion (§5-2)
# ============================================================


def test_body_inclusion_normal(cfg: VisibilityCheckConfig):
    """정상 13점, nose+ankle visibility 0.9 → is_valid=True."""
    pl = _normal_pl()
    r = evaluate_frame_body_inclusion(pl, cfg)
    assert r.is_valid
    assert r.passed_checks["body_visibility"]
    assert r.passed_checks["body_coords"]
    assert r.failed_reasons == []
    assert abs(r.check_values["nose_visibility"] - 0.9) < 1e-9
    assert r.check_values["coord_out_of_range_count"] == 0.0


def test_body_inclusion_nose_visibility_fail(cfg: VisibilityCheckConfig):
    """nose visibility 0.5 < 0.6 임계 → body_visibility 실패."""
    pl = _normal_pl(nose_vis=0.5)
    r = evaluate_frame_body_inclusion(pl, cfg)
    assert not r.is_valid
    assert not r.passed_checks["body_visibility"]
    assert r.passed_checks["body_coords"]  # 좌표는 정상
    assert "body_not_fully_visible" in r.failed_reasons


def test_body_inclusion_coord_out_of_range(cfg: VisibilityCheckConfig):
    """shoulder x > 1.0 (out of [0,1] range) → body_coords 실패."""
    pl = _normal_pl(shoulder_x_offset=0.6)  # shoulder.x → 1.05, 1.15
    r = evaluate_frame_body_inclusion(pl, cfg)
    assert not r.is_valid
    assert r.passed_checks["body_visibility"]  # visibility는 정상
    assert not r.passed_checks["body_coords"]
    assert "body_not_fully_visible" in r.failed_reasons
    assert r.check_values["coord_out_of_range_count"] >= 1.0


# ============================================================
# B. evaluate_frame_foot_cutoff (§5-3)
# ============================================================


def test_foot_cutoff_normal(cfg: VisibilityCheckConfig):
    """정상 ankle/heel/foot y < 0.95 (0.85, 0.88) → is_valid=True."""
    pl = _normal_pl()
    r = evaluate_frame_foot_cutoff(pl, "left", cfg)
    assert r.is_valid
    assert r.passed_checks["foot_cutoff"]
    assert r.failed_reasons == []


def test_foot_cutoff_one_point_violation(cfg: VisibilityCheckConfig):
    """foot_index y=0.97 (5-7 α AND 해석: 1점 위반도 fail) → foot_out_of_frame."""
    pl = PoseLandmarks(
        shoulder=_pair(0.50, 0.20),
        hip=_pair(0.50, 0.45),
        knee=_pair(0.50, 0.65),
        ankle=_pair(0.50, 0.85),
        heel=_pair(0.48, 0.88),
        # 분석측(left) foot_index y=0.97 — 1점만 위반
        foot_index=LandmarkPair(left=_lm(0.47, 0.97), right=_lm(0.57, 0.88)),
        nose=_lm(0.50, 0.10, 0.9),
    )
    r = evaluate_frame_foot_cutoff(pl, "left", cfg)
    assert not r.is_valid
    assert not r.passed_checks["foot_cutoff"]
    assert "foot_out_of_frame" in r.failed_reasons
    assert abs(r.check_values["foot_index_y"] - 0.97) < 1e-9


# ============================================================
# C. evaluate_body_inclusion_accumulation (§5-2 누적)
# ============================================================


def test_body_inclusion_accumulation_pass(cfg: VisibilityCheckConfig):
    """70% 통과 (7 valid / 3 invalid) → None (임계 60% 이상)."""
    results = (
        [FrameGeometryResult(is_valid=True) for _ in range(7)]
        + [FrameGeometryResult(is_valid=False) for _ in range(3)]
    )
    out = evaluate_body_inclusion_accumulation(results, cfg)
    assert out is None


def test_body_inclusion_accumulation_fail(cfg: VisibilityCheckConfig):
    """50% 통과 (5 valid / 5 invalid) → 'body_not_fully_visible' (임계 60% 미달)."""
    results = (
        [FrameGeometryResult(is_valid=True) for _ in range(5)]
        + [FrameGeometryResult(is_valid=False) for _ in range(5)]
    )
    out = evaluate_body_inclusion_accumulation(results, cfg)
    assert out == "body_not_fully_visible"


# ============================================================
# D. evaluate_foot_cutoff_accumulation (§5-3 누적)
# ============================================================


def test_foot_cutoff_accumulation_fail(cfg: VisibilityCheckConfig):
    """50% 통과 → 'foot_out_of_frame' (임계 60% 미달)."""
    results = (
        [FrameGeometryResult(is_valid=True) for _ in range(5)]
        + [FrameGeometryResult(is_valid=False) for _ in range(5)]
    )
    out = evaluate_foot_cutoff_accumulation(results, cfg)
    assert out == "foot_out_of_frame"
