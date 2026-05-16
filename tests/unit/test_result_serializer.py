"""result_serializer.py unit 테스트 (Phase 7-B).

Phase 7-A sanity 16 case 이식 + tie-breaking 1 case 보강 = 19 case.

카테고리:
A. 통계 helper (7 case + tie 1)
B. convert_phase6_video_meta (1)
C. 4 메시지 JSON round-trip (7)
D. Pydantic Literal 검증 (1)

영상 의존 X — 합성 데이터 + Pydantic 모델 직렬화/역직렬화.

⚠️ tie-breaking catch (Phase 7-A 정정 가치):
- compute_fsp_dominant은 Counter.most_common(1)[0]을 사용
- Python Counter는 dict insertion order 보존 → 동률 시 첫 등장 라벨 우세
- 본 동작은 *확정 사양*으로 회귀 case에 박음 (발표 Q&A 대비)
"""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from choborunner_ai.result_serializer import (
    AnalysisProgressMessage,
    AnalysisResultMessage,
    AngleStats,
    ErrorMessage,
    FeedbackMessage,
    FrameInferenceMessage,
    FrameInferenceResult,
    MetricDetails,
    Metrics,
    QualitySummary,
    Resolution,
    VideoMeta,
    compute_angle_stats,
    compute_fsp_dominant,
    convert_fsp_label,
    convert_phase6_video_meta,
)
from choborunner_ai.video_preprocessor import VideoMeta as Phase6VideoMeta


# ============================================================
# A. 통계 helper
# ============================================================


def test_compute_angle_stats_normal():
    stats = compute_angle_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.n_strides == 5
    assert abs(stats.median - 3.0) < 1e-9
    assert abs(stats.iqr[0] - 2.0) < 1e-9
    assert abs(stats.iqr[1] - 4.0) < 1e-9


def test_compute_angle_stats_with_nan():
    stats = compute_angle_stats(
        [1.0, float("nan"), 3.0, float("nan"), 5.0]
    )
    assert stats.n_strides == 3
    assert abs(stats.median - 3.0) < 1e-9


def test_compute_angle_stats_empty():
    stats = compute_angle_stats([])
    assert stats.n_strides == 0
    assert not math.isfinite(stats.median)


def test_compute_fsp_dominant_normal():
    dom, ratio, dist = compute_fsp_dominant(["rfs", "rfs", "mfs", "rfs"])
    assert dom == "RFS"
    assert abs(ratio - 0.75) < 1e-9
    assert dist == {"RFS": 3, "MFS": 1}


def test_compute_fsp_dominant_all_none():
    dom, ratio, dist = compute_fsp_dominant([None, None, None])
    assert dom is None
    assert ratio == 0.0
    assert dist == {}


def test_compute_fsp_dominant_empty():
    dom, ratio, dist = compute_fsp_dominant([])
    assert dom is None
    assert ratio == 0.0
    assert dist == {}


def test_compute_fsp_dominant_tie_breaking():
    """동률 시 첫 등장 라벨 우세 (Counter insertion order 보존).

    Python Counter / dict는 insertion order 보존이라 동률 시 첫 입력 라벨이
    most_common(1)[0]로 반환됨. 본 동작은 확정 사양.
    """
    dom, ratio, dist = compute_fsp_dominant(["rfs", "mfs", "rfs", "mfs"])
    assert dom == "RFS"  # 'rfs' 먼저 등장
    assert abs(ratio - 0.5) < 1e-9
    assert dist == {"RFS": 2, "MFS": 2}


@pytest.mark.parametrize(
    "lower,upper",
    [("rfs", "RFS"), ("mfs", "MFS"), ("ffs", "FFS")],
)
def test_convert_fsp_label(lower: str, upper: str):
    assert convert_fsp_label(lower) == upper  # type: ignore[arg-type]


# ============================================================
# B. convert_phase6_video_meta
# ============================================================


def test_convert_phase6_video_meta():
    phase6 = Phase6VideoMeta(
        width=1920, height=1080, fps=30.0, frame_count=300, rotation_degrees=90
    )
    docs_meta = convert_phase6_video_meta(phase6)
    assert docs_meta.total_frames == 300
    assert abs(docs_meta.fps_actual - 30.0) < 1e-9
    assert docs_meta.resolution.width == 1920
    assert docs_meta.resolution.height == 1080
    assert abs(docs_meta.duration_sec - 10.0) < 1e-9


# ============================================================
# C. 4 메시지 JSON round-trip
# ============================================================


def _make_video_meta() -> VideoMeta:
    return VideoMeta(
        duration_sec=12.4,
        fps_actual=29.8,
        resolution=Resolution(width=1280, height=720),
        total_frames=369,
    )


def test_frame_inference_roundtrip():
    msg = FrameInferenceMessage(
        frame_index=42,
        timestamp_sec=1.4,
        result=FrameInferenceResult(pose_detected=True, frame_quality_flags=[]),
    )
    j = msg.model_dump_json()
    back = FrameInferenceMessage.model_validate_json(j)
    assert back.type == "frame_inference"
    assert back.frame_index == 42
    assert back.result.pose_detected is True


def test_analysis_progress_basic():
    msg = AnalysisProgressMessage(
        stage="collecting_strides",
        valid_stride_count=2,
        elapsed_sec=3.5,
        message="분석 중입니다.",
    )
    j = msg.model_dump_json()
    back = AnalysisProgressMessage.model_validate_json(j)
    assert back.stage == "collecting_strides"
    assert back.feedback_messages is None


def test_analysis_progress_with_feedback():
    msg = AnalysisProgressMessage(
        stage="analyzing",
        valid_stride_count=4,
        elapsed_sec=4.2,
        feedback_messages=[
            FeedbackMessage(
                category="posture_warning",
                metric="trunk_lean",
                tts_text="상체 기울임 주의",
                display_text="상체가 일반 범위보다 더 앞으로 기울어졌습니다.",
                priority=2,
                tts_enabled=True,
                confidence_prefix=False,
            )
        ],
    )
    j = msg.model_dump_json()
    back = AnalysisProgressMessage.model_validate_json(j)
    assert back.feedback_messages is not None
    assert len(back.feedback_messages) == 1
    assert back.feedback_messages[0].category == "posture_warning"
    assert back.feedback_messages[0].priority == 2


def test_analysis_result_success():
    msg = AnalysisResultMessage(
        status="success",
        video_meta=_make_video_meta(),
        analysis_side="right",
        metrics=Metrics(
            foot_strike_pattern="MFS",
            foot_strike_angle_deg=2.3,
            initial_knee_flexion_deg=19.4,
            trunk_lean_deg=7.8,
        ),
        metric_details=MetricDetails(
            foot_strike_angle_deg=AngleStats(
                median=2.3, iqr=[1.0, 3.4], n_strides=5
            ),
            initial_knee_flexion_deg=AngleStats(
                median=19.4, iqr=[17.8, 21.2], n_strides=5
            ),
            trunk_lean_deg=AngleStats(
                median=7.8, iqr=[6.5, 9.1], n_strides=5
            ),
        ),
        quality_summary=QualitySummary(
            valid_frame_ratio=0.84,
            ic_candidate_count=6,
            valid_stride_count=5,
            landmark_visibility_avg=0.78,
            target_tracking_stability="stable",
        ),
        primary_reason_code=None,
        reason_codes=[],
    )
    j = msg.model_dump_json()
    back = AnalysisResultMessage.model_validate_json(j)
    assert back.status == "success"
    assert back.metrics is not None
    assert back.metrics.foot_strike_pattern == "MFS"
    assert back.reason_codes == []


def test_analysis_result_low_confidence():
    msg = AnalysisResultMessage(
        status="low_confidence",
        video_meta=_make_video_meta(),
        analysis_side="right",
        metrics=Metrics(
            foot_strike_pattern="MFS",
            foot_strike_angle_deg=1.8,
            initial_knee_flexion_deg=17.2,
            trunk_lean_deg=8.4,
        ),
        metric_details=MetricDetails(
            foot_strike_angle_deg=AngleStats(
                median=1.8, iqr=[0.5, 3.0], n_strides=5
            ),
            initial_knee_flexion_deg=AngleStats(
                median=17.2, iqr=[15.0, 19.0], n_strides=5
            ),
            trunk_lean_deg=AngleStats(
                median=8.4, iqr=[7.0, 9.5], n_strides=5
            ),
        ),
        quality_summary=QualitySummary(
            valid_frame_ratio=0.7,
            ic_candidate_count=5,
            valid_stride_count=5,
            landmark_visibility_avg=0.65,
            target_tracking_stability="borderline",
        ),
        primary_reason_code="low_ic_confidence",
        reason_codes=["low_ic_confidence", "unstable_foot_angle"],
        message="착지 시점을 안정적으로 찾기 어렵습니다.",
    )
    j = msg.model_dump_json()
    back = AnalysisResultMessage.model_validate_json(j)
    assert back.status == "low_confidence"
    assert back.primary_reason_code == "low_ic_confidence"
    assert len(back.reason_codes) == 2


def test_analysis_result_failed():
    msg = AnalysisResultMessage(
        status="failed",
        video_meta=VideoMeta(
            duration_sec=2.3,
            fps_actual=30.0,
            resolution=Resolution(width=1280, height=720),
            total_frames=69,
        ),
        primary_reason_code="too_short",
        reason_codes=["too_short"],
        message="러닝 구간이 너무 짧아 분석이 어렵습니다.",
    )
    j = msg.model_dump_json()
    back = AnalysisResultMessage.model_validate_json(j)
    assert back.status == "failed"
    # docs §5-4: failed는 metrics/metric_details/analysis_side/quality_summary
    # /feedback_messages 모두 None
    assert back.metrics is None
    assert back.metric_details is None
    assert back.analysis_side is None
    assert back.quality_summary is None
    assert back.feedback_messages is None
    assert back.primary_reason_code == "too_short"


def test_error_message():
    msg = ErrorMessage(
        frame_index=42,
        error_code="internal_processing_failed",
        error_detail="MediaPipe 추론 중 예외 발생",
    )
    j = msg.model_dump_json()
    back = ErrorMessage.model_validate_json(j)
    assert back.type == "error"
    assert back.frame_index == 42
    assert back.error_code == "internal_processing_failed"


# ============================================================
# D. Pydantic Literal 검증
# ============================================================


def test_literal_validation_invalid_status():
    """status='invalid_status' → ValidationError."""
    with pytest.raises(ValidationError):
        AnalysisResultMessage(
            status="invalid_status",  # type: ignore[arg-type]
            video_meta=_make_video_meta(),
        )
