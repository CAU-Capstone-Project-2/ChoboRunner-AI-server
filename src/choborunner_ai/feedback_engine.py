# -*- coding: utf-8 -*-
"""docs/2-3-6 Rule-based 실시간 피드백 — feedback_engine 정식판 (Phase 8-G).

본 모듈은 docs/2-3-6 (Rule-based 피드백) 단일 정답 구현. Phase 5 dataclass
classification + Phase 8-F ResponseStatusResult 입력 → list[FeedbackMessage]
산출 (Phase 7-A schema).

Phase 8-G lock 15 결정 (Day 7 잠금):
- Critical 4: scope α (full 16 룰 + 출력 한도) / 함수 α 통합 / FeedbackContext α
  신설 (frozen) / 한글 메시지 α 코드 내부 dict
- Default 11: 지표별 dict / 출력 한도 내부 / TTS 1 priority 정렬 / IQR 5°/5°/2.5° /
  "좋은 페이스" 조건 / reason_code 매핑 16 entry / 신규 모듈 / classification None 처리 /
  status 분기 (failed/low_conf/success)

⚠️ scope α 명세 (lock 8-G-1 α):
- docs §4 16 룰 매트릭스 본체:
  · §4-1 Foot Strike Pattern (4 룰: RFS/MFS/FFS/Uncertain ≥5 stride)
  · §4-2 Initial Knee Flexion (3 룰: Below/Typical/Above Typical)
  · §4-3 Trunk Lean (3 룰: Near Vertical/Forward Lean/Above Typical)
  · §4-4 측정 분산 (3 지표 통합 1 룰: IQR > 분류 임계 폭 50%)
  · §4-5 "좋은 페이스" (success + 모두 typical + 분산 작음)
- docs §3-4 status 분기:
  · failed: system_info(primary) 1개만, 자세 룰 skip
  · low_confidence: 자세 룰 + confidence_prefix=True + system_info(primary)
  · success: 자세 룰 모두 + "좋은 페이스" 가능
- docs §3-5 출력 한도: 화면 3개 + TTS 1개
- docs §5 reason_code → system_info 매핑 (REASON_CODE_USER_MESSAGES 16 entry)

⚠️ Phase 8-I (실시간 Pipeline) scope 보류:
- docs §3-1 첫 출력 약 2~3초 후
- docs §3-2 빈도 제한 (동일 5초 / 다른 2초 / 긍정 30초)
- docs §3-5 TTS cycle 정의
- docs §4-5 "분석을 시작합니다" / "분석이 완료되었습니다" 메시지
본 8-G는 stateless (1 호출 = 1 응답).

⚠️ docs §4-4 분류 임계 폭 50% 직역 (catch — docs 명시 부족):
- Foot Strike: rfs_above (5°) − ffs_below (−5°) = 10° → 50% = **5°**
- Knee Flexion: above_typical (25°) − below_typical (15°) = 10° → 50% = **5°**
- Trunk Lean: forward_above (10°) − near_vertical_below (5°) = 5° → 50% = **2.5°**
파일럿 보정 후보 + 향후 config 이동 anchor.

⚠️ classification None 처리 (lock 8-G-13):
- foot_strike None → uncertain_stride_count 반영, 누적 ≥ 5 stride 시 Uncertain 룰
- knee_flexion / trunk_lean None → 해당 지표 자세 피드백 skip

⚠️ Phase 9 + 메타데이터 미구현 reason_code 매핑 보류 (lock 8-G-10):
- target_switch_detected / unstable_landmark_sequence (Phase 9)
- too_short / low_resolution / low_fps (메타데이터, docs/2-3-1)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from choborunner_ai.metrics.knee_flexion import KneeFlexionClassification
from choborunner_ai.metrics.trunk_lean import TrunkLeanClassification
from choborunner_ai.quality_gate import ReasonCode
from choborunner_ai.result_serializer import (
    AnalysisStatus,
    FeedbackCategory,
    FeedbackMessage,
    FootStrikePatternLabel,
)


# ============================================================
# IQR 임계 폭 50% 상수 (lock 8-G-8 — docs §4-4 직역)
# ============================================================


FOOT_VARIANCE_THRESHOLD = 5.0
"""Foot Strike IQR 임계 폭 50%. (rfs_above 5° − ffs_below −5°) × 50% = 5°."""

KNEE_VARIANCE_THRESHOLD = 5.0
"""Knee Flexion IQR 임계 폭 50%. (above_typical 25° − below_typical 15°) × 50% = 5°."""

TRUNK_VARIANCE_THRESHOLD = 2.5
"""Trunk Lean IQR 임계 폭 50%. (forward_above 10° − near_vertical_below 5°) × 50% = 2.5°."""

UNCERTAIN_STRIDE_THRESHOLD = 5
"""docs §4-1 — foot_strike Uncertain (classification=None) 5 stride 이상 지속 시 트리거."""


# ============================================================
# FeedbackContext dataclass (lock 8-G-3 α — frozen)
# ============================================================


@dataclass(frozen=True)
class FeedbackContext:
    """compute_feedback_messages 입력 dataclass — 8+ 인자 통합 (Phase 8-B-1/8-F 패턴).

    Phase 8-I integration 시점 산출 책임:
    - status / primary_reason_code: Phase 8-F ResponseStatusResult에서 추출
    - trunk_classification / knee_classification: dominant or median classification (호출자 산출)
    - foot_dominant: Phase 7-A compute_fsp_dominant UPPER 결과
    - foot_iqr / knee_iqr / trunk_iqr: Phase 7-A compute_angle_stats IQR (Q1, Q3)
    - uncertain_stride_count: foot_strike_results 중 classification=None count

    Attributes:
        status: 'success' / 'low_confidence' / 'failed' (Phase 8-F ResponseStatusResult).
        primary_reason_code: 사용자 노출 대표 reason code (1개). success 시 None.
        trunk_classification: 'near_vertical' / 'forward_lean' / 'above_typical' 또는 None.
        knee_classification: 'below_typical' / 'typical' / 'above_typical' 또는 None.
        foot_dominant: 'RFS' / 'MFS' / 'FFS' (UPPER, Phase 7-A 패턴) 또는 None (Uncertain).
        foot_iqr / knee_iqr / trunk_iqr: (Q1, Q3) 튜플 또는 None.
        uncertain_stride_count: foot_strike classification=None 누적 count.
    """

    status: AnalysisStatus
    primary_reason_code: Optional[ReasonCode]
    trunk_classification: Optional[TrunkLeanClassification]
    knee_classification: Optional[KneeFlexionClassification]
    foot_dominant: Optional[FootStrikePatternLabel]
    foot_iqr: Optional[tuple[float, float]]
    knee_iqr: Optional[tuple[float, float]]
    trunk_iqr: Optional[tuple[float, float]]
    uncertain_stride_count: int = 0


# ============================================================
# 룰 매트릭스 dict (lock 8-G-5 α — 지표별 + 한글 메시지)
# ============================================================


# §4-1 Foot Strike Pattern (RFS/MFS/FFS, 모두 posture_info + TTS False)
FOOT_STRIKE_MESSAGES: dict[FootStrikePatternLabel, tuple[Optional[str], str]] = {
    "RFS": (
        None,  # TTS X
        "현재 뒤꿈치 중심 착지(rearfoot strike) 패턴이 관찰됩니다.",
    ),
    "MFS": (
        None,
        "현재 발 중간 부위 착지(midfoot strike) 패턴이 관찰됩니다.",
    ),
    "FFS": (
        None,
        "현재 앞발 중심 착지(forefoot strike) 패턴이 관찰됩니다.",
    ),
}

# §4-1 Uncertain 5 stride 이상 지속 (system_info + TTS True)
FOOT_UNCERTAIN_MESSAGE: tuple[str, str] = (
    "착지 패턴이 안정적이지 않아요.",
    "착지 패턴 측정이 안정적이지 않습니다. 자세를 일정하게 유지해주세요.",
)


# §4-2 Initial Knee Flexion (Below/Typical/Above)
KNEE_FLEXION_MESSAGES: dict[KneeFlexionClassification, tuple[Optional[str], str]] = {
    "below_typical": (
        "무릎 굴곡이 작은 편이에요.",
        "무릎 굴곡이 일반 범위(15~25°)보다 작게 관찰됩니다. "
        "착지 충격을 줄이기 위해 무릎을 조금 더 부드럽게 사용해보세요.",
    ),
    "typical": (
        None,
        "착지 시 무릎 굴곡이 일반 범위에 있습니다.",
    ),
    "above_typical": (
        "무릎이 많이 굽혀지고 있어요.",
        "착지 시 무릎 굴곡이 일반 범위보다 큽니다. "
        "충격 흡수에는 유리할 수 있지만 에너지 효율이 낮아질 수 있습니다.",
    ),
}


# §4-3 Trunk Lean (Near Vertical/Forward Lean/Above Typical)
TRUNK_LEAN_MESSAGES: dict[TrunkLeanClassification, tuple[Optional[str], str]] = {
    "near_vertical": (
        None,
        "상체가 거의 수직에 가깝습니다. 약간의 전방 기울기는 러닝 효율에 도움이 될 수 있습니다.",
    ),
    "forward_lean": (
        None,
        "상체 전경사가 일반 범위에 있습니다.",
    ),
    "above_typical": (
        "상체가 많이 기울어져 있어요.",
        "상체가 일반 범위보다 더 앞으로 기울어져 있습니다. 자세가 무너지지 않도록 주의해보세요.",
    ),
}


# §4-4 측정 분산 (3 지표 통합 1 메시지, system_info + TTS True)
VARIANCE_MESSAGE: tuple[str, str] = (
    "자세를 일정하게 유지해주세요.",
    "측정 분산이 큰 편입니다. 자세를 일정하게 유지해주세요.",
)


# §4-5 "좋은 페이스" — success + 모두 typical + 분산 작음 (posture_info + TTS True 예외)
GOOD_PACE_MESSAGE: tuple[str, str] = (
    "좋은 페이스 유지해주세요.",
    "현재 자세가 안정적입니다. 좋은 페이스 유지해주세요.",
)


# §5 reason_code → system_info 메시지 매핑 (docs/2-3-5 §8 표 인용, lock 8-G-10)
# ⚠️ Phase 9 + 메타데이터 미구현 reason_code 매핑 보류:
#    target_switch_detected / unstable_landmark_sequence / too_short / low_resolution / low_fps
REASON_CODE_USER_MESSAGES: dict[ReasonCode, str] = {
    # §5-1 visibility (Phase 4, docs §8-2)
    "lower_body_not_visible": "다리가 잘 보이지 않습니다. 화면에 다리 전체가 보이도록 촬영해주세요.",
    "foot_not_visible": "발이 잘 보이지 않습니다. 화면에 발이 보이도록 다시 촬영해주세요.",
    "upper_body_not_visible": "어깨와 상체가 잘 보이지 않습니다.",
    "low_landmark_visibility": "자세 추정의 신뢰도가 낮습니다. 밝은 곳에서 다시 촬영해주세요.",
    # §5-2 / §5-3 (Phase 8-A, docs §8-2)
    "body_not_fully_visible": "머리부터 발끝까지 전신이 화면에 들어오도록 다시 촬영해주세요.",
    "foot_out_of_frame": "발이 화면 아래에서 잘립니다. 카메라를 조금 멀리 두세요.",
    # §5-4 측면 구도 (Phase 8-B-2, docs §8-3)
    "invalid_view": "측면이 아닌 각도로 촬영된 것 같습니다. 몸의 옆면이 보이도록 다시 촬영해주세요.",
    # §5-5 / §5-6 (Phase 8-C, docs §8-6)
    "camera_unstable": "카메라가 흔들렸습니다. 다음에는 카메라를 고정해서 촬영해주세요.",
    "unstable_foot_angle": "발 착지 자세가 매번 달라 안정적인 측정이 어려웠습니다.",
    "unstable_knee_angle": "무릎 각도가 매번 달라 안정적인 측정이 어려웠습니다.",
    "unstable_trunk_angle": "상체 자세가 매번 달라 안정적인 측정이 어려웠습니다.",
    # §5-7 IC 검증 (Phase 8-D, docs §8-5)
    "insufficient_stride": "분석 가능한 발 착지 시점이 충분하지 않습니다. 더 길게 달려주세요.",
    "low_ic_confidence": "착지 시점을 안정적으로 찾기 어렵습니다. 다시 촬영해주세요.",
    "insufficient_window": "일부 구간 자세가 흐트러져 측정이 어렵습니다. 다시 촬영해주세요.",
    # §4 추적 안정성 (Phase 8-E scope γ, docs §8-4)
    "target_lost": "분석 대상자가 화면에서 사라졌습니다. 화면에 계속 보이도록 다시 촬영해주세요.",
    "background_person_interference": "주변에 다른 사람이 있어 추정이 불안정합니다. 가능하면 다시 촬영해주세요.",
}


# ============================================================
# 내부 유틸 함수
# ============================================================


_CATEGORY_TO_PRIORITY: dict[FeedbackCategory, int] = {
    "system_info": 1,
    "posture_warning": 2,
    "posture_info": 3,
}


def _iqr_violates(
    iqr: Optional[tuple[float, float]], threshold: float
) -> bool:
    """IQR (Q1, Q3) 폭이 임계 초과 시 True. None이면 False."""
    if iqr is None:
        return False
    q1, q3 = iqr
    return (q3 - q1) > threshold


def _build_message(
    category: FeedbackCategory,
    metric: Optional[str],
    tts_text: Optional[str],
    display_text: str,
    tts_enabled: bool,
    confidence_prefix: bool,
) -> FeedbackMessage:
    """FeedbackMessage 생성 — category에서 priority 자동 매핑."""
    return FeedbackMessage(
        category=category,
        metric=metric,
        tts_text=tts_text,
        display_text=display_text,
        priority=_CATEGORY_TO_PRIORITY[category],  # type: ignore[arg-type]
        tts_enabled=tts_enabled,
        confidence_prefix=confidence_prefix,
    )


# ============================================================
# 메인 함수 — compute_feedback_messages (lock 8-G-2 α 통합)
# ============================================================


def compute_feedback_messages(ctx: FeedbackContext) -> list[FeedbackMessage]:
    """docs/2-3-6 룰 매트릭스 + status 분기 + 출력 한도 적용 (Phase 8-G).

    ⚠️ status 분기 (docs §3-4):
    - 'failed': system_info(primary) 1개만, 자세 룰 skip (lock 8-G-14)
    - 'low_confidence': 자세 룰 + confidence_prefix=True + system_info(primary)
    - 'success': 자세 룰 모두 + "좋은 페이스" 가능 (lock 8-G-15)

    ⚠️ 룰 매트릭스 (docs §4, 16 룰):
    - §4-1 Foot Strike: foot_dominant + uncertain_stride_count
    - §4-2 Knee Flexion (3 분류)
    - §4-3 Trunk Lean (3 분류)
    - §4-4 측정 분산 (3 IQR 중 1 위반 시 1 메시지)
    - §4-5 "좋은 페이스" (success + 모두 typical + 분산 작음)

    ⚠️ 출력 한도 (docs §3-5, lock 8-G-6/7):
    - 화면: priority 정렬 상위 3개
    - TTS: priority 정렬 후 tts_enabled=True 첫 1개만 유지 (나머지 override False)

    ⚠️ classification None 처리 (lock 8-G-13):
    - foot_strike None → uncertain_stride_count 활용, foot 메시지 skip
    - knee_flexion / trunk_lean None → 해당 지표 자세 피드백 skip

    Args:
        ctx: FeedbackContext (Phase 8-I integration 시점 산출).

    Returns:
        list[FeedbackMessage] — 화면 최대 3개, TTS 최대 1개 enabled.
    """
    confidence_prefix = ctx.status == "low_confidence"
    messages: list[FeedbackMessage] = []

    # ── status='failed': system_info(primary) 1개만, 자세 룰 skip ──
    if ctx.status == "failed":
        if (
            ctx.primary_reason_code is not None
            and ctx.primary_reason_code in REASON_CODE_USER_MESSAGES
        ):
            msg_text = REASON_CODE_USER_MESSAGES[ctx.primary_reason_code]
            messages.append(
                _build_message(
                    category="system_info",
                    metric=None,
                    tts_text=msg_text,
                    display_text=msg_text,
                    tts_enabled=True,
                    confidence_prefix=False,
                )
            )
        return messages

    # ── status='low_confidence': system_info(primary) 추가 ──
    if (
        ctx.status == "low_confidence"
        and ctx.primary_reason_code is not None
        and ctx.primary_reason_code in REASON_CODE_USER_MESSAGES
    ):
        msg_text = REASON_CODE_USER_MESSAGES[ctx.primary_reason_code]
        messages.append(
            _build_message(
                category="system_info",
                metric=None,
                tts_text=msg_text,
                display_text=msg_text,
                tts_enabled=True,
                confidence_prefix=False,
            )
        )

    # ── 자세 룰 평가 (low_confidence + success 공통) ──

    # §4-1 Foot Strike (Uncertain 누적 5+ 우선, 그 외 dominant 메시지)
    if ctx.uncertain_stride_count >= UNCERTAIN_STRIDE_THRESHOLD:
        tts, display = FOOT_UNCERTAIN_MESSAGE
        messages.append(
            _build_message(
                category="system_info",
                metric="foot_strike",
                tts_text=tts,
                display_text=display,
                tts_enabled=True,
                confidence_prefix=confidence_prefix,
            )
        )
    elif ctx.foot_dominant is not None and ctx.foot_dominant in FOOT_STRIKE_MESSAGES:
        tts, display = FOOT_STRIKE_MESSAGES[ctx.foot_dominant]
        messages.append(
            _build_message(
                category="posture_info",
                metric="foot_strike",
                tts_text=tts,
                display_text=display,
                tts_enabled=False,  # §4-1 RFS/MFS/FFS TTS False
                confidence_prefix=confidence_prefix,
            )
        )

    # §4-2 Knee Flexion
    if (
        ctx.knee_classification is not None
        and ctx.knee_classification in KNEE_FLEXION_MESSAGES
    ):
        tts, display = KNEE_FLEXION_MESSAGES[ctx.knee_classification]
        is_warning = ctx.knee_classification != "typical"
        category: FeedbackCategory = "posture_warning" if is_warning else "posture_info"
        messages.append(
            _build_message(
                category=category,
                metric="knee_flexion",
                tts_text=tts,
                display_text=display,
                tts_enabled=is_warning,
                confidence_prefix=confidence_prefix,
            )
        )

    # §4-3 Trunk Lean
    if (
        ctx.trunk_classification is not None
        and ctx.trunk_classification in TRUNK_LEAN_MESSAGES
    ):
        tts, display = TRUNK_LEAN_MESSAGES[ctx.trunk_classification]
        is_warning = ctx.trunk_classification == "above_typical"
        category = "posture_warning" if is_warning else "posture_info"
        messages.append(
            _build_message(
                category=category,
                metric="trunk_lean",
                tts_text=tts,
                display_text=display,
                tts_enabled=is_warning,
                confidence_prefix=confidence_prefix,
            )
        )

    # §4-4 측정 분산 (3 IQR 중 1개라도 위반 시 1 메시지)
    if (
        _iqr_violates(ctx.foot_iqr, FOOT_VARIANCE_THRESHOLD)
        or _iqr_violates(ctx.knee_iqr, KNEE_VARIANCE_THRESHOLD)
        or _iqr_violates(ctx.trunk_iqr, TRUNK_VARIANCE_THRESHOLD)
    ):
        tts, display = VARIANCE_MESSAGE
        messages.append(
            _build_message(
                category="system_info",
                metric=None,
                tts_text=tts,
                display_text=display,
                tts_enabled=True,
                confidence_prefix=False,
            )
        )

    # §4-5 "좋은 페이스" (success + 모두 typical + 분산 작음)
    if (
        ctx.status == "success"
        and ctx.trunk_classification == "forward_lean"
        and ctx.knee_classification == "typical"
        and ctx.foot_dominant in ("RFS", "MFS", "FFS")
        and not _iqr_violates(ctx.foot_iqr, FOOT_VARIANCE_THRESHOLD)
        and not _iqr_violates(ctx.knee_iqr, KNEE_VARIANCE_THRESHOLD)
        and not _iqr_violates(ctx.trunk_iqr, TRUNK_VARIANCE_THRESHOLD)
    ):
        tts, display = GOOD_PACE_MESSAGE
        messages.append(
            _build_message(
                category="posture_info",
                metric=None,
                tts_text=tts,
                display_text=display,
                tts_enabled=True,  # §4-5 예외 (긴 빈도, 동기 부여)
                confidence_prefix=False,
            )
        )

    # ── 출력 한도 적용 (docs §3-5, lock 8-G-6/7) ──
    messages.sort(key=lambda m: m.priority)
    messages = messages[:3]  # 화면 한도 3개

    tts_kept = False
    final_messages: list[FeedbackMessage] = []
    for msg in messages:
        if msg.tts_enabled:
            if tts_kept:
                final_messages.append(msg.model_copy(update={"tts_enabled": False}))
            else:
                tts_kept = True
                final_messages.append(msg)
        else:
            final_messages.append(msg)

    return final_messages
