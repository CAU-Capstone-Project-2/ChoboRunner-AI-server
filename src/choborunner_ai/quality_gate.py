"""docs/2-3-5 §5-1 visibility 검증 — Pose 추출 후 frame-level 품질 검사.

본 모듈은 docs/2-3-5 §5-1 (Landmark visibility 검사) 영역을 담당한다.
docs/2-3-5 §1 단일 정답 3가지 중 본 Phase 4 scope:
- [O] Pose 후 품질 검사 기준 §5-1만 (§5-2~5-7은 별도 Phase)
- [X] 분석 상태값 분기 §6 — 별도 Phase
- [X] Reason code 우선순위·사용자 메시지 §8-7 — 별도 Phase

Phase 4 작업 단위:
- Phase 4-A (본 단계): Literal[ReasonCode] + FrameVisibilityResult + evaluate_frame_visibility
- Phase 4-B: evaluate_visibility_accumulation (docs §5-1 5번 유효 frame 비율)
- Phase 4-C: 통합 sanity end-to-end (scripts/sanity/)

PoseQualityFlag (pose_extractor.py) vs ReasonCode (본 모듈) 분리 (Day 5 decision 6):
- PoseQualityFlag: 자료구조 신호 — frame-level 추상 표시 (예 'low_pose_visibility').
  추출 단계 부여 또는 future 활용 미정.
- ReasonCode: docs §8 SoT — 사용자 메시지 매핑 (예 'low_landmark_visibility').
  본 quality_gate 부여 + 응답 메시지(2-3-7)로 전달.
- 두 Literal 이름 충돌 X, 의미 겹침은 있지만 책임 분리 유지.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from choborunner_ai.config import VisibilityCheckConfig
from choborunner_ai.pose_extractor import PoseLandmarks

logger = logging.getLogger(__name__)


# ============================================================
# Reason Code Literal (docs §8-2 visibility 그룹)
# ============================================================


ReasonCode = Literal[
    "lower_body_not_visible",
    "foot_not_visible",
    "upper_body_not_visible",
    "low_landmark_visibility",
]
"""docs/2-3-5 §8-2 visibility 카테고리 reason code 4종.

각 코드 강도(failed vs low_confidence)는 REASON_CODE_SEVERITY 참조.
사용자 메시지는 docs §8-2 표 (SoT). 응답 메시지 매핑 + 우선순위 1개 선택은
§6 status 분기 + §8-7 우선순위 별도 Phase에서.
"""


Severity = Literal["failed", "low_confidence"]


REASON_CODE_SEVERITY: dict[ReasonCode, Severity] = {
    "lower_body_not_visible": "low_confidence",
    "foot_not_visible": "failed",
    "upper_body_not_visible": "low_confidence",
    "low_landmark_visibility": "low_confidence",
}
"""docs §8-2 visibility reason code 강도 매핑.

`foot_not_visible`만 failed — 발 미가시 시 IC 검출 불가, 핵심 지표 산출 자체
불가 (docs §5-1 표 정합). 나머지 3종은 low_confidence — 지표 산출은 가능하나
신뢰도 낮음.

사용 위치는 §6 status 분기 별도 Phase. 본 dict 정의는 reason code 사전과
함께 본 모듈에 위치 (SoT 정합).
"""


# ============================================================
# Frame-level visibility 평가 결과 dataclass
# ============================================================


@dataclass
class FrameVisibilityResult:
    """단일 frame visibility 평가 결과 (docs §5-1 1~4 검사).

    Attributes:
        passed_categories: 4 카테고리 통과 여부 dict — keys
            'lower_body' / 'foot' / 'upper_body' / 'overall_avg'.
        category_averages: 4 카테고리 실제 visibility 평균값 dict (디버깅 자산,
            파일럿 보정 시 분포 확인용). keys 동일.
        failed_reasons: 실패 카테고리에 매핑된 reason code list. 순서 결정적
            (lower_body / foot / upper_body / overall_avg 순).
        is_valid: 4 카테고리 모두 통과 시 True (= failed_reasons 빈 list).
    """

    passed_categories: dict[str, bool]
    category_averages: dict[str, float]
    failed_reasons: list[ReasonCode]
    is_valid: bool


# ============================================================
# 단일 frame visibility 평가 함수 (docs §5-1 1~4)
# ============================================================


def evaluate_frame_visibility(
    pl: PoseLandmarks,
    analysis_side: Literal["left", "right"],
    cfg: VisibilityCheckConfig,
) -> FrameVisibilityResult:
    """단일 frame visibility 평가 (docs/2-3-5 §5-1 1~4).

    4 카테고리 평균 + 임계 비교:
    1. lower_body = mean(hip[side].vis, knee[side].vis, ankle[side].vis)
       임계 cfg.visibility_threshold_lower_body (default 0.6) →
       실패 시 'lower_body_not_visible' (low_confidence)
    2. foot = mean(heel[side].vis, foot_index[side].vis)
       임계 cfg.visibility_threshold_foot (default 0.6) →
       실패 시 'foot_not_visible' (failed)
    3. upper_body = mean(shoulder.left.vis, shoulder.right.vis)
       임계 cfg.visibility_threshold_upper_body (default 0.6) →
       실패 시 'upper_body_not_visible' (low_confidence)
    4. overall_avg = mean(12 landmark vis 전체)
       임계 cfg.visibility_threshold_overall_avg (default 0.5) →
       실패 시 'low_landmark_visibility' (low_confidence)

    docs §5-1 5번 (유효 frame 비율 ≥ 60%)는 본 함수 범위 외 — Phase 4-B
    `evaluate_visibility_accumulation`이 본 함수 결과 list를 입력으로 받음.

    Args:
        pl: 6 LandmarkPair PoseLandmarks (Phase 3 출력).
        analysis_side: 분석측 'left' 또는 'right'. 분석측 결정은 본 Phase
            범위 외 (docs §3-1, 별도 Phase에서 결정 후 입력).
        cfg: VisibilityCheckConfig (config.py).

    Returns:
        FrameVisibilityResult — 4 카테고리 평가 결과.

    Raises:
        ValueError: analysis_side가 'left'/'right' 아닌 경우 (Literal 타입
            보장 + 런타임 가드).

    견고성 가드:
    - analysis_side Literal 런타임 검증.
    - 평가 자체 예외 -> logger.exception + 4 카테고리 모두 실패 default 반환
      (메인 분석 흐름 보호, docs §6 failed 우선순위 정합).
    """
    if analysis_side not in ("left", "right"):
        raise ValueError(
            f"analysis_side는 'left' 또는 'right'만 허용, got {analysis_side!r}"
        )
    try:
        side_hip = pl.hip.left if analysis_side == "left" else pl.hip.right
        side_knee = pl.knee.left if analysis_side == "left" else pl.knee.right
        side_ankle = pl.ankle.left if analysis_side == "left" else pl.ankle.right
        side_heel = pl.heel.left if analysis_side == "left" else pl.heel.right
        side_foot = (
            pl.foot_index.left if analysis_side == "left" else pl.foot_index.right
        )

        avg_lower_body = (
            side_hip.visibility + side_knee.visibility + side_ankle.visibility
        ) / 3.0
        avg_foot = (side_heel.visibility + side_foot.visibility) / 2.0
        avg_upper_body = (
            pl.shoulder.left.visibility + pl.shoulder.right.visibility
        ) / 2.0

        all_vis = [
            pl.shoulder.left.visibility, pl.shoulder.right.visibility,
            pl.hip.left.visibility, pl.hip.right.visibility,
            pl.knee.left.visibility, pl.knee.right.visibility,
            pl.ankle.left.visibility, pl.ankle.right.visibility,
            pl.heel.left.visibility, pl.heel.right.visibility,
            pl.foot_index.left.visibility, pl.foot_index.right.visibility,
        ]
        avg_overall = sum(all_vis) / len(all_vis)

        passed = {
            "lower_body": avg_lower_body >= cfg.visibility_threshold_lower_body,
            "foot": avg_foot >= cfg.visibility_threshold_foot,
            "upper_body": avg_upper_body >= cfg.visibility_threshold_upper_body,
            "overall_avg": avg_overall >= cfg.visibility_threshold_overall_avg,
        }
        averages = {
            "lower_body": avg_lower_body,
            "foot": avg_foot,
            "upper_body": avg_upper_body,
            "overall_avg": avg_overall,
        }
        failed_reasons: list[ReasonCode] = []
        if not passed["lower_body"]:
            failed_reasons.append("lower_body_not_visible")
        if not passed["foot"]:
            failed_reasons.append("foot_not_visible")
        if not passed["upper_body"]:
            failed_reasons.append("upper_body_not_visible")
        if not passed["overall_avg"]:
            failed_reasons.append("low_landmark_visibility")

        return FrameVisibilityResult(
            passed_categories=passed,
            category_averages=averages,
            failed_reasons=failed_reasons,
            is_valid=len(failed_reasons) == 0,
        )
    except Exception:
        logger.exception(
            "evaluate_frame_visibility 예외 (swallow, 4 카테고리 모두 실패 default)"
        )
        return FrameVisibilityResult(
            passed_categories={
                "lower_body": False,
                "foot": False,
                "upper_body": False,
                "overall_avg": False,
            },
            category_averages={
                "lower_body": 0.0,
                "foot": 0.0,
                "upper_body": 0.0,
                "overall_avg": 0.0,
            },
            failed_reasons=[
                "lower_body_not_visible",
                "foot_not_visible",
                "upper_body_not_visible",
                "low_landmark_visibility",
            ],
            is_valid=False,
        )
