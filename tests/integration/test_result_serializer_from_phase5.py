"""result_serializer integration 테스트 — Phase 5 dataclass 입력 통합 (Phase 7-B).

Phase 5 dataclass 합성 → helper → AngleStats/Metrics 생성 통합 검증.

흐름:
- TrunkLeanResult / KneeFlexionResult / FootStrikeResult list 합성
- compute_angle_stats × 3 (trunk / knee / foot_strike_deg)
- compute_fsp_dominant (FootStrikeClassification list)
- AnalysisResultMessage 조립 (status='success' 분기)

3 case:
1. integration_normal: 5 stride 정상 → success 응답 정상 조립
2. integration_with_nan: NaN 섞인 stride → helper NaN 제외 + n_strides 정확성
3. integration_fsp_all_none: FootStrikeClassification 모두 None → fsp dominant None
"""
from __future__ import annotations

import math

import pytest

from choborunner_ai.metrics.foot_strike import (
    FootStrikeClassification,
    FootStrikeResult,
)
from choborunner_ai.metrics.knee_flexion import KneeFlexionResult
from choborunner_ai.metrics.trunk_lean import TrunkLeanResult
from choborunner_ai.result_serializer import (
    AnalysisResultMessage,
    AngleStats,
    MetricDetails,
    Metrics,
    QualitySummary,
    Resolution,
    VideoMeta,
    compute_angle_stats,
    compute_fsp_dominant,
)


def _make_trunk(deg: float, ic: int, is_valid: bool = True) -> TrunkLeanResult:
    return TrunkLeanResult(
        deg=deg,
        classification="near_vertical" if is_valid and math.isfinite(deg) else None,
        ic_frame_index=ic,
        window_valid_count=5 if is_valid else 0,
        window_total_count=5,
        is_valid=is_valid,
    )


def _make_knee(deg: float, ic: int, is_valid: bool = True) -> KneeFlexionResult:
    return KneeFlexionResult(
        deg=deg,
        classification="typical" if is_valid and math.isfinite(deg) else None,
        ic_frame_index=ic,
        window_valid_count=5 if is_valid else 0,
        window_total_count=5,
        is_valid=is_valid,
    )


def _make_fsp(
    deg: float, cls: FootStrikeClassification | None, ic: int, is_valid: bool = True
) -> FootStrikeResult:
    return FootStrikeResult(
        deg=deg,
        classification=cls,
        ic_frame_index=ic,
        window_valid_count=3 if is_valid else 0,
        window_total_count=3,
        is_valid=is_valid,
    )


def _video_meta() -> VideoMeta:
    return VideoMeta(
        duration_sec=10.0,
        fps_actual=30.0,
        resolution=Resolution(width=1280, height=720),
        total_frames=300,
    )


def test_integration_normal():
    """5 stride 정상 → AnalysisResultMessage success 정상 조립."""
    trunks = [_make_trunk(d, i * 30) for i, d in enumerate([6.5, 7.2, 7.8, 8.4, 9.1])]
    knees = [_make_knee(d, i * 30) for i, d in enumerate([17.8, 19.0, 19.4, 21.0, 21.2])]
    fsps = [
        _make_fsp(d, "mfs", i * 30)
        for i, d in enumerate([1.0, 2.0, 2.3, 3.0, 3.4])
    ]
    fsp_cls = [r.classification for r in fsps]

    trunk_stats = compute_angle_stats([r.deg for r in trunks])
    knee_stats = compute_angle_stats([r.deg for r in knees])
    fsp_stats = compute_angle_stats([r.deg for r in fsps])
    dom, ratio, dist = compute_fsp_dominant(fsp_cls)

    assert trunk_stats.n_strides == 5
    assert abs(trunk_stats.median - 7.8) < 1e-9
    assert knee_stats.n_strides == 5
    assert abs(knee_stats.median - 19.4) < 1e-9
    assert fsp_stats.n_strides == 5
    assert abs(fsp_stats.median - 2.3) < 1e-9
    assert dom == "MFS"
    assert abs(ratio - 1.0) < 1e-9

    # AnalysisResultMessage success 조립
    msg = AnalysisResultMessage(
        status="success",
        video_meta=_video_meta(),
        analysis_side="right",
        metrics=Metrics(
            foot_strike_pattern=dom,
            foot_strike_angle_deg=fsp_stats.median,
            initial_knee_flexion_deg=knee_stats.median,
            trunk_lean_deg=trunk_stats.median,
        ),
        metric_details=MetricDetails(
            foot_strike_angle_deg=fsp_stats,
            initial_knee_flexion_deg=knee_stats,
            trunk_lean_deg=trunk_stats,
        ),
        quality_summary=QualitySummary(
            valid_frame_ratio=0.85,
            ic_candidate_count=5,
            valid_stride_count=5,
            landmark_visibility_avg=0.8,
            target_tracking_stability="stable",
        ),
        primary_reason_code=None,
        reason_codes=[],
    )
    # JSON round-trip 통과 (Pydantic 직렬화 무결성)
    j = msg.model_dump_json()
    back = AnalysisResultMessage.model_validate_json(j)
    assert back.status == "success"
    assert back.metrics is not None
    assert back.metrics.foot_strike_pattern == "MFS"
    assert back.metric_details is not None
    assert back.metric_details.trunk_lean_deg.n_strides == 5


def test_integration_with_nan():
    """5 stride 중 일부 NaN → helper NaN 제외 + n_strides 정확."""
    trunks = [
        _make_trunk(7.0, 10),
        _make_trunk(float("nan"), 30, is_valid=False),
        _make_trunk(7.5, 50),
        _make_trunk(float("nan"), 70, is_valid=False),
        _make_trunk(8.0, 90),
    ]
    stats = compute_angle_stats([r.deg for r in trunks])
    assert stats.n_strides == 3  # NaN 2건 제외
    assert abs(stats.median - 7.5) < 1e-9


def test_integration_fsp_all_none():
    """FootStrikeClassification 모두 None → fsp dominant None."""
    fsps = [_make_fsp(float("nan"), None, i * 30, is_valid=False) for i in range(5)]
    fsp_cls = [r.classification for r in fsps]
    dom, ratio, dist = compute_fsp_dominant(fsp_cls)
    assert dom is None
    assert ratio == 0.0
    assert dist == {}
