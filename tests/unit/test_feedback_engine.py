# -*- coding: utf-8 -*-
"""feedback_engine.py unit 테스트 — docs/2-3-6 (Phase 8-G).

Phase 8-A~8-F (quality_gate) 패턴 일관 — 신규 모듈 = 신규 test 파일.

⚠️ docs §4 16 룰 매트릭스 + §3-4 status 분기 + §3-5 출력 한도 정합 검증.

12 case (sanity 16 case 중 핵심 이식):
- A. status 분기 (3): success+typical / low_conf+primary / failed+primary
- B. 룰 매트릭스 (5): foot RFS+MFS+FFS / Uncertain / knee Below + Above / trunk Above
- C. 분산 + 출력 한도 (2): foot IQR 위반 / 화면 한도 3 + TTS 1
- D. edge (2): knee/trunk None skip / 모두 None success → []
"""
from __future__ import annotations

import pytest

from choborunner_ai.feedback_engine import (
    FeedbackContext,
    compute_feedback_messages,
)


def _ctx_default(**overrides) -> FeedbackContext:
    """기본 ctx 빌더 — 모두 None/typical default."""
    defaults = dict(
        status="success",
        primary_reason_code=None,
        trunk_classification=None,
        knee_classification=None,
        foot_dominant=None,
        foot_iqr=None,
        knee_iqr=None,
        trunk_iqr=None,
        uncertain_stride_count=0,
    )
    defaults.update(overrides)
    return FeedbackContext(**defaults)


# ============================================================
# A. status 분기 (docs §3-4)
# ============================================================


def test_feedback_success_all_typical_screen_limit():
    """success + 모두 typical + 분산 작음 → 화면 3 한도 (good_pace 잘림)."""
    ctx = _ctx_default(
        trunk_classification="forward_lean",
        knee_classification="typical",
        foot_dominant="MFS",
        foot_iqr=(1.0, 3.0), knee_iqr=(17.0, 19.0), trunk_iqr=(6.0, 8.0),
    )
    out = compute_feedback_messages(ctx)
    assert len(out) == 3
    assert all(m.category == "posture_info" for m in out)


def test_feedback_low_confidence_with_primary():
    """low_confidence + primary + 자세 → system_info(primary) + 자세 confidence_prefix=True."""
    ctx = _ctx_default(
        status="low_confidence", primary_reason_code="low_ic_confidence",
        trunk_classification="above_typical", knee_classification="below_typical",
        foot_dominant="RFS",
    )
    out = compute_feedback_messages(ctx)
    sys_msgs = [m for m in out if m.category == "system_info"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0].confidence_prefix is False  # system_info prefix X

    posture_msgs = [m for m in out if m.category != "system_info"]
    assert all(m.confidence_prefix for m in posture_msgs)  # 자세 prefix=True

    tts_count = sum(1 for m in out if m.tts_enabled)
    assert tts_count == 1  # TTS 1개 한도


def test_feedback_failed_only_system_info():
    """failed + primary → system_info(primary) 1개만, 자세 룰 skip."""
    ctx = _ctx_default(
        status="failed", primary_reason_code="foot_out_of_frame",
        trunk_classification="above_typical", knee_classification="below_typical",
        foot_dominant="RFS",
    )
    out = compute_feedback_messages(ctx)
    assert len(out) == 1
    assert out[0].category == "system_info"
    assert "발이 화면 아래" in out[0].display_text


# ============================================================
# B. 룰 매트릭스 (docs §4-1/§4-2/§4-3)
# ============================================================


@pytest.mark.parametrize(
    "foot_dominant,expected_substr",
    [("RFS", "뒤꿈치 중심"), ("MFS", "발 중간 부위"), ("FFS", "앞발 중심")],
)
def test_feedback_foot_strike_patterns(foot_dominant, expected_substr):
    """§4-1 foot strike 3 패턴 모두 posture_info + TTS False."""
    ctx = _ctx_default(foot_dominant=foot_dominant)
    out = compute_feedback_messages(ctx)
    assert len(out) == 1
    assert out[0].category == "posture_info"
    assert out[0].tts_enabled is False
    assert expected_substr in out[0].display_text


def test_feedback_foot_uncertain_5_strides():
    """§4-1 Uncertain 5 stride 이상 지속 → system_info + TTS True."""
    ctx = _ctx_default(foot_dominant=None, uncertain_stride_count=5)
    out = compute_feedback_messages(ctx)
    assert len(out) == 1
    assert out[0].category == "system_info"
    assert out[0].tts_enabled is True
    assert "착지 패턴이 안정적이지 않" in out[0].tts_text


def test_feedback_knee_below_typical_warning():
    """§4-2 knee Below Typical → posture_warning + TTS True."""
    ctx = _ctx_default(knee_classification="below_typical")
    out = compute_feedback_messages(ctx)
    assert out[0].category == "posture_warning"
    assert out[0].tts_enabled is True
    assert "무릎 굴곡이 작은" in out[0].tts_text


def test_feedback_knee_above_typical_warning():
    """§4-2 knee Above Typical → posture_warning + TTS True."""
    ctx = _ctx_default(knee_classification="above_typical")
    out = compute_feedback_messages(ctx)
    assert out[0].category == "posture_warning"
    assert "많이 굽혀지고" in out[0].tts_text


def test_feedback_trunk_above_typical_warning():
    """§4-3 trunk Above Typical → posture_warning + TTS True."""
    ctx = _ctx_default(trunk_classification="above_typical")
    out = compute_feedback_messages(ctx)
    assert out[0].category == "posture_warning"
    assert "많이 기울어져" in out[0].tts_text


# ============================================================
# C. 분산 + 출력 한도 (docs §4-4 + §3-5)
# ============================================================


def test_feedback_variance_foot_iqr_violation():
    """§4-4 foot IQR 위반 → VARIANCE_MESSAGE 1개 (system_info + TTS True)."""
    ctx = _ctx_default(
        foot_iqr=(0.0, 10.0),  # 폭 10 > 5 위반
        knee_iqr=(17.0, 19.0), trunk_iqr=(6.0, 8.0),
    )
    out = compute_feedback_messages(ctx)
    assert len(out) == 1
    assert out[0].category == "system_info"
    assert "측정 분산" in out[0].display_text


def test_feedback_screen_limit_3_tts_limit_1():
    """§3-5 출력 한도 — 화면 3개 + TTS 1개.

    warning 2 + variance 1 + posture_info 1 → priority 정렬 후 3개 한도.
    """
    ctx = _ctx_default(
        trunk_classification="above_typical",  # warning
        knee_classification="below_typical",  # warning
        foot_dominant="MFS",  # info
        foot_iqr=(0.0, 10.0),  # variance system_info
        knee_iqr=(17.0, 19.0), trunk_iqr=(6.0, 8.0),
    )
    out = compute_feedback_messages(ctx)
    assert len(out) == 3
    # priority 정렬: system_info(1) > warning(2) > warning(2)
    assert out[0].category == "system_info"
    assert out[1].category == "posture_warning"
    assert out[2].category == "posture_warning"
    tts_count = sum(1 for m in out if m.tts_enabled)
    assert tts_count == 1


# ============================================================
# D. edge case
# ============================================================


def test_feedback_classification_none_skip():
    """lock 8-G-13 — knee/trunk None → 해당 지표 skip, foot만 평가."""
    ctx = _ctx_default(
        foot_dominant="RFS",
        # knee_classification=None, trunk_classification=None (default)
    )
    out = compute_feedback_messages(ctx)
    assert len(out) == 1
    assert out[0].metric == "foot_strike"


def test_feedback_all_none_success_empty():
    """모두 None + status='success' → 빈 list."""
    ctx = _ctx_default()
    out = compute_feedback_messages(ctx)
    assert out == []
