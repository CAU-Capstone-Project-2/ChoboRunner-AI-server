# -*- coding: utf-8 -*-
"""Phase 8-J integration test — Pipeline end-to-end + build_analysis_result 회귀 보호.

Phase 8 묶음 종료 마지막 sub-phase — 회귀 보호 자산.

⚠️ scope (lock 8-J-1 α + 8-J-2 β):
- A. 영상 시나리오 (1 case): jaemin.mp4 e2e (Phase 8-I sanity 영구화)
- B. 합성 status 매트릭스 (3 case): success / low_confidence / failed
  · docs/2-3-7 §5-5 status별 필드 매트릭스 정합
- C. helper 단위 (2 case): compute_classification_dominant + compute_uncertain_stride_count

⚠️ slow marker 없음 (lock 8-J-3 β) — 기본 회귀 포함.
⚠️ 합성 builder = test 내부 함수 (lock 8-J-5).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from choborunner_ai.config import AppConfig
from choborunner_ai.metrics.foot_strike import FootStrikeResult
from choborunner_ai.metrics.ic_detector import ICResult
from choborunner_ai.metrics.knee_flexion import KneeFlexionResult
from choborunner_ai.metrics.trunk_lean import TrunkLeanResult
from choborunner_ai.pipeline import Pipeline, PipelineResult
from choborunner_ai.pose_extractor import Landmark, LandmarkPair, PoseLandmarks
from choborunner_ai.quality_gate import ReasonCodeEntry
from choborunner_ai.result_serializer import (
    AnalysisResultMessage,
    Metrics,
    MetricDetails,
    QualitySummary,
    build_analysis_result,
    compute_classification_dominant,
    compute_uncertain_stride_count,
)
from choborunner_ai.video_preprocessor import VideoMeta as Phase6VideoMeta


VIDEO_PATH = Path("legacy/demo_02/jaemin.mp4")
MAX_FRAMES = 100  # 빠른 회귀 (전체 982 중 100)


# ============================================================
# A. 영상 시나리오 (jaemin.mp4 e2e, Phase 8-I sanity 영구화)
# ============================================================


@pytest.mark.skipif(not VIDEO_PATH.exists(), reason="jaemin.mp4 not present")
def test_pipeline_e2e_jaemin_video():
    """jaemin.mp4 100 frame e2e — Phase 8-I sanity 영구화 회귀 보호.

    실측 결과 (Phase 8-I commit f65cea0):
    - status='low_confidence' / primary='unstable_knee_angle'
    - foot=MFS, knee_median≈26.28°, trunk_median≈2.91°
    - feedback 3개 (system_info TTS + posture_warning + posture_info)
    - quality_summary: ic=4, stride=4
    """
    cfg = AppConfig()
    with Pipeline(cfg) as p:
        result = p.run_on_video_file(
            video_path=VIDEO_PATH,
            analysis_side="left",
            max_frames=MAX_FRAMES,
            direction="left_to_right",
        )

    # PipelineResult 보존 (디버깅 자산)
    assert result.pose_landmarks_count > 0
    assert len(result.ic_results) > 0

    # AnalysisResultMessage 채움
    ar = result.analysis_result
    assert ar is not None
    assert isinstance(ar, AnalysisResultMessage)

    # jaemin 실측 status (knee >20° → unstable_knee_angle 또는 above_typical warning)
    assert ar.status in ("success", "low_confidence")
    # foot_dominant 산출 정상 (Uncertain 아님)
    assert ar.metrics is not None
    assert ar.metrics.foot_strike_pattern in ("RFS", "MFS", "FFS")
    # 회귀 보호: knee 26° 근처 (jaemin 실측)
    assert 20.0 < ar.metrics.initial_knee_flexion_deg < 35.0

    # JSON 직렬화 + round-trip
    json_str = ar.model_dump_json()
    back = AnalysisResultMessage.model_validate_json(json_str)
    assert back.status == ar.status


# ============================================================
# B. 합성 status 매트릭스 (3 case, docs/2-3-7 §5-5 정합)
# ============================================================


def _make_landmarks_series(n: int = 30) -> list:
    """정상 visibility 합성 landmarks (30 frame)."""
    pl = PoseLandmarks(
        shoulder=LandmarkPair(
            left=Landmark(0.45, 0.20, 0.9), right=Landmark(0.55, 0.20, 0.9)
        ),
        hip=LandmarkPair(
            left=Landmark(0.48, 0.45, 0.9), right=Landmark(0.52, 0.45, 0.9)
        ),
        knee=LandmarkPair(
            left=Landmark(0.5, 0.65, 0.9), right=Landmark(0.5, 0.65, 0.9)
        ),
        ankle=LandmarkPair(
            left=Landmark(0.5, 0.85, 0.9), right=Landmark(0.5, 0.85, 0.9)
        ),
        heel=LandmarkPair(
            left=Landmark(0.48, 0.88, 0.9), right=Landmark(0.52, 0.88, 0.9)
        ),
        foot_index=LandmarkPair(
            left=Landmark(0.52, 0.88, 0.9), right=Landmark(0.52, 0.88, 0.9)
        ),
    )
    return [pl] * n


def _build_synthetic_pipeline_result(scenario: str) -> PipelineResult:
    """합성 PipelineResult builder (lock 8-J-5).

    Scenarios:
    - 'success': reason_code_entries=[] + 모두 typical (metrics 정상)
    - 'low_confidence': unstable_knee_angle entry + 모두 정상 metrics
    - 'failed': foot_out_of_frame entry (자세 metrics 무관)
    """
    video_meta = Phase6VideoMeta(width=1280, height=720, fps=30.0, frame_count=300)
    landmarks_series = _make_landmarks_series(30)

    # 5 stride (모두 정상 classification)
    ic_results = [
        ICResult(frame_index=i * 5, confidence="high", stage1_offset=0)
        for i in range(5)
    ]
    trunk_lean_results = [
        TrunkLeanResult(
            deg=7.0,
            classification="forward_lean",
            ic_frame_index=i * 5,
            window_valid_count=5,
            window_total_count=5,
            is_valid=True,
        )
        for i in range(5)
    ]
    knee_flexion_results = [
        KneeFlexionResult(
            deg=20.0,
            classification="typical",
            ic_frame_index=i * 5,
            window_valid_count=5,
            window_total_count=5,
            is_valid=True,
        )
        for i in range(5)
    ]
    foot_strike_results = [
        FootStrikeResult(
            deg=2.0,
            classification="mfs",
            ic_frame_index=i * 5,
            window_valid_count=3,
            window_total_count=3,
            is_valid=True,
        )
        for i in range(5)
    ]

    # scenario별 reason_code_entries
    if scenario == "success":
        entries: list[ReasonCodeEntry] = []
    elif scenario == "low_confidence":
        entries = [
            ReasonCodeEntry(
                reason_code="unstable_knee_angle", severity="low_confidence"
            )
        ]
    elif scenario == "failed":
        entries = [
            ReasonCodeEntry(reason_code="foot_out_of_frame", severity="failed")
        ]
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    return PipelineResult(
        video_meta=video_meta,
        frame_results=[],
        landmarks_series=landmarks_series,
        ic_results=ic_results,
        trunk_lean_results=trunk_lean_results,
        knee_flexion_results=knee_flexion_results,
        foot_strike_results=foot_strike_results,
        reason_code_entries=entries,
        analysis_result=None,
        pose_landmarks_count=30,
        pose_not_detected_count=0,
    )


def test_synthetic_success():
    """status='success' — reason_code_entries=[], 전체 필드 채움."""
    pr = _build_synthetic_pipeline_result("success")
    ar = build_analysis_result(pr, "left")

    assert ar.status == "success"
    assert ar.primary_reason_code is None
    assert ar.reason_codes == []
    assert ar.analysis_side == "left"
    assert ar.message is None  # success → message None (lock 8-I-8)
    # 전체 필드 채움
    assert isinstance(ar.metrics, Metrics)
    assert isinstance(ar.metric_details, MetricDetails)
    assert isinstance(ar.quality_summary, QualitySummary)
    assert ar.metrics.foot_strike_pattern == "MFS"
    assert abs(ar.metrics.trunk_lean_deg - 7.0) < 1e-6
    assert abs(ar.metrics.initial_knee_flexion_deg - 20.0) < 1e-6


def test_synthetic_low_confidence():
    """status='low_confidence' — unstable_knee_angle entry, 전체 필드 + confidence_prefix=True."""
    pr = _build_synthetic_pipeline_result("low_confidence")
    ar = build_analysis_result(pr, "left")

    assert ar.status == "low_confidence"
    assert ar.primary_reason_code == "unstable_knee_angle"
    assert ar.reason_codes == ["unstable_knee_angle"]
    # message = REASON_CODE_USER_MESSAGES["unstable_knee_angle"]
    assert ar.message is not None
    assert "무릎" in ar.message  # 한글 메시지 정합
    # 전체 필드 (low_conf도 채움)
    assert ar.metrics is not None
    assert ar.metric_details is not None
    assert ar.quality_summary is not None
    # 자세 피드백 confidence_prefix=True (system_info 제외)
    if ar.feedback_messages:
        posture_msgs = [m for m in ar.feedback_messages if m.category != "system_info"]
        assert all(m.confidence_prefix for m in posture_msgs)


def test_synthetic_failed():
    """status='failed' — foot_out_of_frame, metrics/metric_details/quality_summary/feedback_messages None.

    docs/2-3-7 §5-5 failed 시 필드 매트릭스 정합 (lock 8-I-9).
    """
    pr = _build_synthetic_pipeline_result("failed")
    ar = build_analysis_result(pr, "left")

    assert ar.status == "failed"
    assert ar.primary_reason_code == "foot_out_of_frame"
    assert ar.reason_codes == ["foot_out_of_frame"]
    assert ar.message is not None
    assert "발이 화면" in ar.message
    # docs §5-5 failed 필드 매트릭스 — 아래 모두 None
    assert ar.analysis_side is None
    assert ar.metrics is None
    assert ar.metric_details is None
    assert ar.quality_summary is None
    assert ar.feedback_messages is None


# ============================================================
# C. helper 단위 (compute_classification_dominant + compute_uncertain_stride_count)
# ============================================================


def test_compute_classification_dominant_mode():
    """Generic mode 산출 — Phase 7-A compute_fsp_dominant 패턴 일관 (lock 8-I-4 α)."""
    # trunk_lean classification
    out = compute_classification_dominant(
        ["forward_lean", "forward_lean", "above_typical", "forward_lean", "near_vertical"]
    )
    assert out == "forward_lean"

    # knee_flexion classification
    out = compute_classification_dominant(
        ["typical", "below_typical", "typical", "typical"]
    )
    assert out == "typical"

    # 모두 None → None
    out = compute_classification_dominant([None, None, None])
    assert out is None

    # 빈 list → None
    out = compute_classification_dominant([])
    assert out is None

    # None 섞임 — None 제외 mode
    out = compute_classification_dominant(["typical", None, "typical", "below_typical"])
    assert out == "typical"


def test_compute_uncertain_stride_count():
    """foot_strike classification=None count (Phase 8-G UNCERTAIN_STRIDE_THRESHOLD 입력)."""
    results = [
        FootStrikeResult(
            deg=2.0,
            classification="mfs",
            ic_frame_index=0,
            window_valid_count=3,
            window_total_count=3,
            is_valid=True,
        ),
        FootStrikeResult(
            deg=float("nan"),
            classification=None,
            ic_frame_index=10,
            window_valid_count=0,
            window_total_count=3,
            is_valid=False,
        ),
        FootStrikeResult(
            deg=float("nan"),
            classification=None,
            ic_frame_index=20,
            window_valid_count=0,
            window_total_count=3,
            is_valid=False,
        ),
    ]
    assert compute_uncertain_stride_count(results) == 2
    assert compute_uncertain_stride_count([]) == 0
