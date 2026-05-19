"""docs/2-3-5 §5-1 + §5-2 + §5-3 — Pose 추출 후 frame-level 품질 검사.

본 모듈은 docs/2-3-5 단일 정답 3가지 중 §5 영역을 점진 확장하며 담당한다.
docs/2-3-5 §1 단일 정답 3가지 중 본 모듈 scope:
- [O] §5-1 Landmark visibility (Phase 4 완료)
- [O] §5-2 body 포함 검사 + §5-3 발 잘림 검사 (Phase 8-A 본 확장)
- [X] §5-4 ~ §5-7 / §4 추적 안정성 — 별도 Phase 8-B ~ 8-E
- [X] §6 status 분기 — Phase 8-F
- [X] §8-7 reason code 우선순위 — Phase 8-F

Phase 4 작업 단위 (완료):
- Phase 4-A: Literal[ReasonCode] + FrameVisibilityResult + evaluate_frame_visibility
- Phase 4-B: evaluate_visibility_accumulation (§5-1 5번 유효 frame 비율)
- Phase 4-C: 통합 sanity end-to-end (scripts/sanity/)

Phase 8-A 작업 단위 (본 확장):
- §5-2 body 포함: nose+양측 ankle visibility + 13점 좌표 [0, 1] 범위
- §5-3 발 잘림: 분석측 ankle/heel/foot_index y < 0.95
- 신규 reason code 2종: `body_not_fully_visible`, `foot_out_of_frame` (둘 다 failed)
- Phase 8-A lock 5-1 α: Phase 3 PoseLandmarks에 nose 필드 추가 (default None)
- Phase 8-A lock 5-5 β: FrameGeometryResult 단일 dataclass로 §5-2/§5-3 묶음 — §5-1
  visibility 평균 카테고리와 의미적 분리 유지.
- Phase 8-A lock 5-6 α: 누적 평가 함수 분리 (body / foot 각각). 반환 타입
  `Optional[ReasonCode]` (§5-1의 `list[ReasonCode]`와 다름 — 향후 §6 status
  분기 단계에서 통일 검토).

PoseQualityFlag (pose_extractor.py) vs ReasonCode (본 모듈) 분리 (Day 5 decision 6):
- PoseQualityFlag: 자료구조 신호 — frame-level 추상 표시 (예 'low_pose_visibility').
  추출 단계 부여 또는 future 활용 미정.
- ReasonCode: docs §8 SoT — 사용자 메시지 매핑 (예 'low_landmark_visibility').
  본 quality_gate 부여 + 응답 메시지(2-3-7)로 전달.
- 두 Literal 이름 충돌 X, 의미 겹침은 있지만 책임 분리 유지.

"유효 frame" 정의 분리 (Day 5 decision 7):
- 본 모듈 §5-1 5번 한정: FrameVisibilityResult.is_valid=True (4 카테고리 모두 통과).
- docs §3-1 정의(분석측 5 landmark 평균 임계 통과)와 다름 — §3-1 정의는
  분석측 결정 정책 범위, §5-1까지 확장은 docs 명시 X.
- 향후 §3-1 진입 Phase에서 별도 함수로 분리 예정.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from choborunner_ai.config import (
    ICValidationConfig,
    MetricVariabilityConfig,
    SideViewConfig,
    StrideExclusionConfig,
    TrackingStabilityConfig,
    VisibilityCheckConfig,
)
from choborunner_ai.metrics.ic_detector import ICConfidence
from choborunner_ai.pose_extractor import PoseLandmarks
from choborunner_ai.result_serializer import AnalysisStatus

logger = logging.getLogger(__name__)


# ============================================================
# Reason Code Literal (docs §8-2 visibility 그룹)
# ============================================================


ReasonCode = Literal[
    # §5-1 visibility (Phase 4)
    "lower_body_not_visible",
    "foot_not_visible",
    "upper_body_not_visible",
    "low_landmark_visibility",
    # §5-2 body 포함 / §5-3 발 잘림 (Phase 8-A)
    "body_not_fully_visible",
    "foot_out_of_frame",
    # §5-4 측면 구도 (Phase 8-B-2) — context-dependent 2 severity
    "invalid_view",
    # §5-5 카메라 흔들림 / §5-6 지표 변동성 (Phase 8-C, stride-level)
    "camera_unstable",
    "unstable_foot_angle",
    "unstable_knee_angle",
    "unstable_trunk_angle",
    # §5-7 IC 검증 (Phase 8-D, stride-level — severity 혼합 첫 등장)
    "insufficient_stride",
    "low_ic_confidence",
    "insufficient_window",
    # §4 추적 안정성 (Phase 8-E scope γ — visibility 1 신호 2 reason_code)
    # ⚠️ Phase 9 분할 anchor: target_switch_detected / unstable_landmark_sequence
    # 미구현 (lock 8-E-2 β YAGNI). target_switch는 pelvis+scale+visibility 3 신호
    # AND 동시 트리거, unstable_landmark는 heel/foot mid-stance 잔차 (mid-stance 시점
    # 정의 catch 7-3). scope γ 채택으로 catch 7-2 (scale 산출 부재) + 7-3 해소.
    "target_lost",
    "background_person_interference",
    # §4 target_switch_detected (Phase 9-A — pelvis spike + scale spike + visibility 일시 붕괴 3 AND)
    "target_switch_detected",
]
"""docs/2-3-5 §8-2/§8-3 visibility/geometry/side-view 카테고리 reason code 7종.

§5-1 (Phase 4) — `lower_body_not_visible` / `foot_not_visible` /
`upper_body_not_visible` / `low_landmark_visibility`.

§5-2 / §5-3 (Phase 8-A) — `body_not_fully_visible` (전신 미포함, 머리/발끝
잘림 의심) / `foot_out_of_frame` (발 화면 하단 잘림 의심).

§5-4 (Phase 8-B-2) — `invalid_view` (측면 구도 미달, context-dependent 2 severity):
- failed: 1차 조건 (hip x 거리) 통과 frame ratio < 0.6
- low_confidence: 1차 통과 frame 중 보조 조건 (shoulder x / torso yaw) 모두
  위반 frame ratio ≥ 0.3 (heuristic, β 1차 통과 분모)

각 코드 강도(failed vs low_confidence)는 REASON_CODE_SEVERITY 참조 (fixed
severity 코드) 또는 산출 함수가 직접 결정 (context-dependent 코드 — invalid_view).
사용자 메시지는 docs §8-2/§8-3 표 (SoT). 응답 메시지 매핑 + 우선순위 1개 선택은
§6 status 분기 + §8-7 우선순위 별도 Phase (Phase 8-F)에서.
"""


Severity = Literal["failed", "low_confidence"]


REASON_CODE_SEVERITY: dict[ReasonCode, Severity] = {
    # §5-1 (Phase 4)
    "lower_body_not_visible": "low_confidence",
    "foot_not_visible": "failed",
    "upper_body_not_visible": "low_confidence",
    "low_landmark_visibility": "low_confidence",
    # §5-2 / §5-3 (Phase 8-A) — docs §8-2 표 둘 다 failed
    "body_not_fully_visible": "failed",
    "foot_out_of_frame": "failed",
    # §5-4 (Phase 8-B-2) — context-dependent. default 'failed' (β 채택, 더 심각한
    # 케이스 default). 산출 함수가 context (frame ratio)로 override 가능.
    "invalid_view": "failed",
    # §5-5 / §5-6 (Phase 8-C, stride-level) — docs §8-6 표 모두 low_confidence
    "camera_unstable": "low_confidence",
    "unstable_foot_angle": "low_confidence",
    "unstable_knee_angle": "low_confidence",
    "unstable_trunk_angle": "low_confidence",
    # §5-7 (Phase 8-D, stride-level) — docs §8-5 표 severity 혼합
    "insufficient_stride": "failed",
    "low_ic_confidence": "low_confidence",
    "insufficient_window": "low_confidence",
    # §4 (Phase 8-E scope γ) — docs §8-4 표 severity 혼합
    "target_lost": "failed",
    "background_person_interference": "low_confidence",
    # §4 (Phase 9-A) — docs §8-4 표 failed
    "target_switch_detected": "failed",
}
"""docs §8-2 visibility/geometry reason code **기본/default** severity 사전.

⚠️ 의미 변화 (Phase 8-B-1 δ 도입):
- 기존 (Phase 4~8-A): 고정 severity 사전 (단일 severity per code)
- 신규 (Phase 8-B-1~): **기본 severity 사전** — 산출 함수가 context-dependent
  코드(예: docs §8-3 `invalid_view` failed/low_confidence 2강도)에 대해 override
  가능. fixed severity 코드(§5-1/§5-2/§5-3)는 본 dict lookup 그대로 사용.
- §5-1: `foot_not_visible`만 failed — 발 미가시 시 IC 검출 불가, 핵심 지표 산출
  자체 불가. 나머지 3종은 low_confidence.
- §5-2 / §5-3 (Phase 8-A): `body_not_fully_visible` / `foot_out_of_frame` 둘 다
  failed (docs §8-2 표) — 전신/발 미포함은 분석 자체가 의미 없음.

사용 위치는 §6 status 분기 별도 Phase (Phase 8-F). 본 dict 정의는 reason code
사전과 함께 본 모듈에 위치 (SoT 정합).

Phase 8-B 진입 시 `invalid_view` 추가 — 본 dict에는 등록 안 함 또는 default
하나만 등록 (context에 따라 산출 함수가 override). Phase 8-B-2 신규 시점 결정.
"""


# ============================================================
# ReasonCodeEntry — 누적 평가 반환 typed dataclass (Phase 8-B-1 δ)
# ============================================================


@dataclass(frozen=True)
class ReasonCodeEntry:
    """누적 평가 함수의 반환 단위 — (reason_code, severity) typed pair.

    Phase 8-B-1 δ 시그니처 통일:
    - 기존 (Phase 4): `evaluate_visibility_accumulation -> list[ReasonCode]`
    - 기존 (Phase 8-A): `evaluate_*_accumulation -> Optional[ReasonCode]`
    - 신규 (Phase 8-B-1~): 모든 누적 평가 함수 반환 `list[ReasonCodeEntry]` 통일.

    severity 결정 정책:
    - fixed severity 코드: REASON_CODE_SEVERITY[code] default 그대로 사용
      (§5-1 `lower_body_not_visible` 등 / §5-2 `body_not_fully_visible` /
      §5-3 `foot_out_of_frame`)
    - context-dependent 코드: 산출 함수가 context(예: frame 비율)로 결정
      (docs §8-3 `invalid_view` failed/low_confidence — Phase 8-B-2 진입 시)

    frozen=True 이유: 누적 평가 결과 불변 보장. status 분기 (Phase 8-F)가 본
    list를 반복 평가하므로 mutation 회피.

    Attributes:
        reason_code: docs/2-3-5 §8 reason code 사전 키.
        severity: 'failed' 또는 'low_confidence' (REASON_CODE_SEVERITY default 또는
            산출 시점 override).
    """

    reason_code: ReasonCode
    severity: Severity


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


# ============================================================
# 누적 visibility 평가 (docs §5-1 5번 유효 frame 비율)
# ============================================================


def evaluate_visibility_accumulation(
    results: list[FrameVisibilityResult],
    cfg: VisibilityCheckConfig,
) -> list[ReasonCodeEntry]:
    """누적 평가 — 유효 frame 비율 (docs/2-3-5 §5-1 5번).

    유효 frame 정의: `FrameVisibilityResult.is_valid=True` (4 카테고리 모두
    통과). docs §3-1의 "유효 frame" 정의(분석측 5 landmark 평균 임계 통과)와
    다름 — 본 모듈 §5-1 5번 한정 정의 (Phase 4 decision 7).

    모집단: 입력 list 전체. 빈 list → 빈 list 반환 (모듈 경계, decision 5:
    frame 부족은 docs/2-3-1 `too_short` trigger 책임, 본 모듈 책임 외).

    ⚠️ 시그니처 변경 (Phase 8-B-1 δ 도입):
    - 기존 (Phase 4): `-> list[ReasonCode]`
    - 신규: `-> list[ReasonCodeEntry]` — severity 정보 동반 (REASON_CODE_SEVERITY
      default 'low_landmark_visibility' → 'low_confidence' wrap).

    Args:
        results: list[FrameVisibilityResult] (Phase 4-A 출력 누적).
        cfg: VisibilityCheckConfig. `valid_frame_ratio_min` 사용 (default 0.6).

    Returns:
        list[ReasonCodeEntry]:
        - 유효 비율 ≥ `valid_frame_ratio_min` 또는 빈 입력 → `[]`
        - 유효 비율 < `valid_frame_ratio_min` →
          `[ReasonCodeEntry('low_landmark_visibility', 'low_confidence')]`

    견고성 가드:
    - 빈 list → `[]` (zero division 가드, 모듈 경계).
    - 평가 예외 → `logger.exception` + failed-safe (low_landmark_visibility,
      low_confidence) 반환.
    """
    try:
        if not results:
            return []
        valid_count = sum(1 for r in results if r.is_valid)
        ratio = valid_count / len(results)
        if ratio < cfg.valid_frame_ratio_min:
            return [
                ReasonCodeEntry(
                    reason_code="low_landmark_visibility",
                    severity=REASON_CODE_SEVERITY["low_landmark_visibility"],
                )
            ]
        return []
    except Exception:
        logger.exception(
            "evaluate_visibility_accumulation 예외 "
            "(swallow, low_landmark_visibility 반환)"
        )
        return [
            ReasonCodeEntry(
                reason_code="low_landmark_visibility",
                severity=REASON_CODE_SEVERITY["low_landmark_visibility"],
            )
        ]


# ============================================================
# §5-2 body 포함 + §5-3 발 잘림 결과 dataclass (Phase 8-A lock 5-5 β)
# ============================================================


@dataclass
class FrameGeometryResult:
    """단일 frame §5-2 + §5-3 평가 결과 (Phase 8-A 묶음 dataclass).

    §5-1 FrameVisibilityResult와 의미적으로 분리 — §5-1은 4 카테고리 visibility
    평균 검사, 본 dataclass는 좌표 기반 (nose 단일점 + 좌표 [0,1] 범위 + y < 0.95)
    검사. 호출 함수에 따라 일부 entry만 채워질 수 있다 (`evaluate_frame_body_inclusion`
    는 body_*, `evaluate_frame_foot_cutoff`는 foot_* 키만 채움).

    FrameVisibilityResult 패턴 차용 — passed_checks dict + check_values dict +
    failed_reasons list + is_valid bool. 디버깅·파일럿 보정 자산 보존.

    Attributes:
        passed_checks: 본 frame 검사 통과 여부 dict.
            §5-2 키: 'body_visibility' (nose+양측 ankle vis 임계 통과) /
                     'body_coords' (13점 좌표 [0,1] 범위 통과).
            §5-3 키: 'foot_cutoff' (분석측 ankle/heel/foot_index y < 0.95).
        check_values: 디버깅용 실제 값 dict (파일럿 임계 보정 시 분포 확인).
            §5-2 키: 'nose_visibility' / 'ankle_left_visibility' /
                     'ankle_right_visibility' / 'coord_out_of_range_count'.
            §5-3 키: 'ankle_y' / 'heel_y' / 'foot_index_y'.
        failed_reasons: 실패 시 reason code list (단일 코드라도 list 형식).
            §5-2 실패 → ['body_not_fully_visible'], §5-3 실패 → ['foot_out_of_frame'].
        is_valid: 본 함수 검사 모두 통과 시 True (= failed_reasons 빈 list).
    """

    passed_checks: dict[str, bool] = field(default_factory=dict)
    check_values: dict[str, float] = field(default_factory=dict)
    failed_reasons: list[ReasonCode] = field(default_factory=list)
    is_valid: bool = False


# ============================================================
# §5-2 body 포함 frame-level 검사 (Phase 8-A)
# ============================================================


def evaluate_frame_body_inclusion(
    pl: PoseLandmarks,
    cfg: VisibilityCheckConfig,
) -> FrameGeometryResult:
    """단일 frame body 포함 평가 (docs/2-3-5 §5-2).

    docs §5-2 2개 sub-check (frame-level 통과 = 둘 다 통과):
    1. nose, 양측 ankle 모두 visibility ≥ `cfg.body_inclusion_visibility_min`
       (default 0.6). docs 표 "nose, ankle 모두" 정합 — analysis_side 무관
       (전신 포함 검사는 측 무관, 양측 ankle 둘 다 보여야 "전신").
    2. 13점 (6 LandmarkPair 12점 + nose) 모두 x, y 좌표가
       [`cfg.coordinate_min`, `cfg.coordinate_max`] (default [0.0, 1.0]) 범위 내.

    누적 평가 (§5-2 frame 비율 60% 이상)는 `evaluate_body_inclusion_accumulation`.

    nose=None 처리 (Phase 8-A lock 5-1 α 정합):
    - 합성 fixture / legacy test의 PoseLandmarks(nose=None 기본)를 위해
      conservative fail-safe — nose=None이면 'body_visibility' check 자동 실패
      ('nose_visibility'=0.0 기록). 'body_coords' check는 nose 제외하고 12점만
      검사 (None은 좌표 비교 자체 불가).
    - 운영 path (`_convert_result`)는 항상 nose=_lm(0) 채우므로 영향 없음.

    Args:
        pl: PoseLandmarks (운영 13점 또는 합성 nose=None 12점).
        cfg: VisibilityCheckConfig.
            - `body_inclusion_visibility_min` (§5-2 1번 임계)
            - `coordinate_min` / `coordinate_max` (§5-2 2번 범위)

    Returns:
        FrameGeometryResult — passed_checks 'body_visibility' / 'body_coords'
        + check_values + failed_reasons (실패 시 ['body_not_fully_visible']) +
        is_valid (둘 다 통과 시 True).

    견고성 가드: 평가 예외 → logger.exception + failed-safe (is_valid=False
    + failed_reasons=['body_not_fully_visible']).
    """
    try:
        # ── §5-2 1번: nose + 양측 ankle visibility 임계 ─────
        if pl.nose is None:
            # 합성 fixture / legacy test path — nose 없음을 visibility=0 취급
            nose_vis = 0.0
        else:
            nose_vis = pl.nose.visibility
        ankle_left_vis = pl.ankle.left.visibility
        ankle_right_vis = pl.ankle.right.visibility

        vis_threshold = cfg.body_inclusion_visibility_min
        body_visibility_ok = (
            nose_vis >= vis_threshold
            and ankle_left_vis >= vis_threshold
            and ankle_right_vis >= vis_threshold
        )

        # ── §5-2 2번: 13점 (또는 nose=None 시 12점) 좌표 [0, 1] 범위 ─────
        coord_lo = cfg.coordinate_min
        coord_hi = cfg.coordinate_max

        def _in_range(v: float) -> bool:
            return coord_lo <= v <= coord_hi

        out_of_range_count = 0
        # 6 LandmarkPair × 2 = 12점 좌표 (x, y 각각 확인)
        for pair in (pl.shoulder, pl.hip, pl.knee, pl.ankle, pl.heel, pl.foot_index):
            for lm in (pair.left, pair.right):
                if not (_in_range(lm.x) and _in_range(lm.y)):
                    out_of_range_count += 1
        # nose (있는 경우)
        if pl.nose is not None:
            if not (_in_range(pl.nose.x) and _in_range(pl.nose.y)):
                out_of_range_count += 1
        body_coords_ok = out_of_range_count == 0

        passed_checks = {
            "body_visibility": body_visibility_ok,
            "body_coords": body_coords_ok,
        }
        check_values = {
            "nose_visibility": nose_vis,
            "ankle_left_visibility": ankle_left_vis,
            "ankle_right_visibility": ankle_right_vis,
            "coord_out_of_range_count": float(out_of_range_count),
        }
        is_valid = body_visibility_ok and body_coords_ok
        failed_reasons: list[ReasonCode] = (
            [] if is_valid else ["body_not_fully_visible"]
        )

        return FrameGeometryResult(
            passed_checks=passed_checks,
            check_values=check_values,
            failed_reasons=failed_reasons,
            is_valid=is_valid,
        )
    except Exception:
        logger.exception(
            "evaluate_frame_body_inclusion 예외 (swallow, body_not_fully_visible default)"
        )
        return FrameGeometryResult(
            passed_checks={"body_visibility": False, "body_coords": False},
            check_values={},
            failed_reasons=["body_not_fully_visible"],
            is_valid=False,
        )


# ============================================================
# §5-3 발 잘림 frame-level 검사 (Phase 8-A)
# ============================================================


def evaluate_frame_foot_cutoff(
    pl: PoseLandmarks,
    analysis_side: Literal["left", "right"],
    cfg: VisibilityCheckConfig,
) -> FrameGeometryResult:
    """단일 frame 발 잘림 평가 (docs/2-3-5 §5-3).

    docs §5-3 단일 check: 분석측 ankle, heel, foot_index의 y 좌표가
    `cfg.foot_cutoff_y_max` (default 0.95) 미만이면 통과.

    ⚠️ docs §5-3 해석 catch (Phase 8-A lock 5-7 α 채택 사유):
    docs §5-3 표 "분석측 ankle, heel, foot_index의 y 좌표 0.95 미만" — 표 직역
    상 두 해석 가능:
    - 해석 (α, 본 구현 채택): **3점 모두 y < 0.95** (AND, "발 잘리지 않음" 조건).
      "발이 잘렸을 가능성" 판정은 3점 중 어느 하나라도 y ≥ 0.95면 잘림 의심.
    - 해석 (β): 3점 중 하나라도 y ≥ 0.95면 즉시 fail (위 α와 사실상 동치).
    α 채택 사유: docs 표 "y 좌표 0.95 미만"을 frame-level 통과 조건으로 직역.
    별도 docs 보강 (해석 명시) 후보 — 본 구현 주석 박음 보존.

    누적 평가 (§5-3 frame 비율 60% 이상)는 `evaluate_foot_cutoff_accumulation`.

    visibility 가드 동반 X (Phase 8-A lock 5-8 α): visibility 검사는 §5-1 책임.
    본 §5-3는 docs 단일 정답 그대로 y 좌표만 검사.

    Args:
        pl: PoseLandmarks (6 LandmarkPair 운영 모드).
        analysis_side: 'left' 또는 'right' (분석측 결정은 별도 Phase, 입력으로 받음).
        cfg: VisibilityCheckConfig.
            - `foot_cutoff_y_max` (§5-3 y 임계, default 0.95)

    Returns:
        FrameGeometryResult — passed_checks 'foot_cutoff' + check_values 'ankle_y'
        / 'heel_y' / 'foot_index_y' + failed_reasons (실패 시 ['foot_out_of_frame'])
        + is_valid.

    Raises:
        ValueError: analysis_side가 'left'/'right' 아닌 경우.

    견고성 가드: 평가 예외 → logger.exception + failed-safe.
    """
    if analysis_side not in ("left", "right"):
        raise ValueError(
            f"analysis_side는 'left' 또는 'right'만 허용, got {analysis_side!r}"
        )
    try:
        side_ankle = pl.ankle.left if analysis_side == "left" else pl.ankle.right
        side_heel = pl.heel.left if analysis_side == "left" else pl.heel.right
        side_foot = (
            pl.foot_index.left if analysis_side == "left" else pl.foot_index.right
        )

        y_max = cfg.foot_cutoff_y_max
        # 5-7 α 해석: 3점 모두 y < 0.95 충족 시 통과
        foot_cutoff_ok = (
            side_ankle.y < y_max and side_heel.y < y_max and side_foot.y < y_max
        )

        passed_checks = {"foot_cutoff": foot_cutoff_ok}
        check_values = {
            "ankle_y": side_ankle.y,
            "heel_y": side_heel.y,
            "foot_index_y": side_foot.y,
        }
        failed_reasons: list[ReasonCode] = (
            [] if foot_cutoff_ok else ["foot_out_of_frame"]
        )

        return FrameGeometryResult(
            passed_checks=passed_checks,
            check_values=check_values,
            failed_reasons=failed_reasons,
            is_valid=foot_cutoff_ok,
        )
    except Exception:
        logger.exception(
            "evaluate_frame_foot_cutoff 예외 (swallow, foot_out_of_frame default)"
        )
        return FrameGeometryResult(
            passed_checks={"foot_cutoff": False},
            check_values={},
            failed_reasons=["foot_out_of_frame"],
            is_valid=False,
        )


# ============================================================
# §5-2 / §5-3 누적 평가 (Phase 8-A lock 5-6 α — 분리 함수)
# ============================================================


def evaluate_body_inclusion_accumulation(
    results: list[FrameGeometryResult],
    cfg: VisibilityCheckConfig,
) -> list[ReasonCodeEntry]:
    """§5-2 body 포함 누적 평가 — 통과 frame 비율 (docs/2-3-5 §5-2 frame 비율 60%).

    유효 frame 정의: `FrameGeometryResult.is_valid=True` (body_visibility AND
    body_coords 통과). 본 함수는 `evaluate_frame_body_inclusion` 출력 누적만
    입력으로 받음 — `evaluate_frame_foot_cutoff` 출력 섞으면 의미 깨짐.

    모집단: 입력 list 전체. 빈 list → 빈 list (frame 부족은 docs/2-3-1 `too_short`
    책임, 본 모듈 책임 외 — §5-1 패턴 일관).

    ⚠️ 시그니처 변경 (Phase 8-B-1 δ 도입):
    - 기존 (Phase 8-A): `-> Optional[ReasonCode]`
    - 신규: `-> list[ReasonCodeEntry]` — `body_not_fully_visible`은 fixed severity
      'failed' (REASON_CODE_SEVERITY default 그대로). Phase 8-A lock 5-6 catch
      해소 — 모든 누적 함수 시그니처 통일.

    Args:
        results: list[FrameGeometryResult] (`evaluate_frame_body_inclusion` 누적).
        cfg: VisibilityCheckConfig. `body_inclusion_frame_ratio_min` (default 0.6).

    Returns:
        list[ReasonCodeEntry]:
        - 통과 비율 ≥ `body_inclusion_frame_ratio_min` 또는 빈 입력 → `[]`
        - 통과 비율 < `body_inclusion_frame_ratio_min` →
          `[ReasonCodeEntry('body_not_fully_visible', 'failed')]`

    견고성 가드: 빈 list → `[]` (zero division), 평가 예외 → failed-safe
    (body_not_fully_visible, failed) 반환.
    """
    try:
        if not results:
            return []
        valid_count = sum(1 for r in results if r.is_valid)
        ratio = valid_count / len(results)
        if ratio < cfg.body_inclusion_frame_ratio_min:
            return [
                ReasonCodeEntry(
                    reason_code="body_not_fully_visible",
                    severity=REASON_CODE_SEVERITY["body_not_fully_visible"],
                )
            ]
        return []
    except Exception:
        logger.exception(
            "evaluate_body_inclusion_accumulation 예외 "
            "(swallow, body_not_fully_visible 반환)"
        )
        return [
            ReasonCodeEntry(
                reason_code="body_not_fully_visible",
                severity=REASON_CODE_SEVERITY["body_not_fully_visible"],
            )
        ]


def evaluate_foot_cutoff_accumulation(
    results: list[FrameGeometryResult],
    cfg: VisibilityCheckConfig,
) -> list[ReasonCodeEntry]:
    """§5-3 발 잘림 누적 평가 — 통과 frame 비율 (docs/2-3-5 §5-3 frame 비율 60%).

    유효 frame 정의: `FrameGeometryResult.is_valid=True` (foot_cutoff 통과).
    본 함수는 `evaluate_frame_foot_cutoff` 출력 누적만 입력으로 받음.

    모집단: 입력 list 전체. 빈 list → 빈 list (§5-1 패턴 일관, frame 부족 책임 외).

    ⚠️ 시그니처 변경 (Phase 8-B-1 δ 도입):
    - 기존 (Phase 8-A): `-> Optional[ReasonCode]`
    - 신규: `-> list[ReasonCodeEntry]` — `foot_out_of_frame`은 fixed severity
      'failed' (REASON_CODE_SEVERITY default 그대로).

    Args:
        results: list[FrameGeometryResult] (`evaluate_frame_foot_cutoff` 누적).
        cfg: VisibilityCheckConfig. `foot_cutoff_frame_ratio_min` (default 0.6).

    Returns:
        list[ReasonCodeEntry]:
        - 통과 비율 ≥ `foot_cutoff_frame_ratio_min` 또는 빈 입력 → `[]`
        - 통과 비율 < `foot_cutoff_frame_ratio_min` →
          `[ReasonCodeEntry('foot_out_of_frame', 'failed')]`

    견고성 가드: 빈 list → `[]`, 평가 예외 → failed-safe.
    """
    try:
        if not results:
            return []
        valid_count = sum(1 for r in results if r.is_valid)
        ratio = valid_count / len(results)
        if ratio < cfg.foot_cutoff_frame_ratio_min:
            return [
                ReasonCodeEntry(
                    reason_code="foot_out_of_frame",
                    severity=REASON_CODE_SEVERITY["foot_out_of_frame"],
                )
            ]
        return []
    except Exception:
        logger.exception(
            "evaluate_foot_cutoff_accumulation 예외 "
            "(swallow, foot_out_of_frame 반환)"
        )
        return [
            ReasonCodeEntry(
                reason_code="foot_out_of_frame",
                severity=REASON_CODE_SEVERITY["foot_out_of_frame"],
            )
        ]


# ============================================================
# §5-4 측면 구도 결과 dataclass (Phase 8-B-2 lock 8-B-4 α)
# ============================================================


@dataclass
class FrameSideViewResult:
    """단일 frame §5-4 측면 구도 평가 결과 (1차 + 보조 a + 보조 b sub-check).

    Phase 8-A FrameGeometryResult 패턴 차용하되 ⚠️ `failed_reasons` 필드 생략 —
    frame-level이 reason code와 1:1 매핑 X (`invalid_view` 부여 여부는 누적
    `evaluate_side_view_accumulation`에서 1차/보조 위반 frame ratio 기반으로
    결정). frame-level 산출은 sub-check 결과만 기록.

    Attributes:
        passed_checks: sub-check 통과 여부 dict.
            - 'primary_hip_x': 좌우 hip x 거리 < hip_x_distance_max (default 0.05)
            - 'secondary_a_shoulder_x': 좌우 shoulder x 거리 <
              shoulder_x_distance_max (default 0.07)
            - 'secondary_b_torso_yaw': hip x 거리 / hip-shoulder y 거리 <
              torso_yaw_proxy_max (default 0.15). 분모 < 1e-6면 자동 위반.
        check_values: 디버깅용 실제 값 dict (파일럿 임계 보정 시 분포 확인).
            - 'hip_x_distance', 'shoulder_x_distance',
              'torso_yaw_ratio', 'hip_shoulder_y_distance'.
        is_valid: frame-level 정상 = primary AND (secondary_a OR secondary_b).
            docs §5-4 표 "정상 측면 구도" 직역.
    """

    passed_checks: dict[str, bool] = field(default_factory=dict)
    check_values: dict[str, float] = field(default_factory=dict)
    is_valid: bool = False


# ============================================================
# §5-4 측면 구도 frame-level 검사 (Phase 8-B-2)
# ============================================================


def evaluate_frame_side_view(
    pl: PoseLandmarks,
    cfg: SideViewConfig,
) -> FrameSideViewResult:
    """단일 frame 측면 구도 평가 (docs/2-3-5 §5-4).

    docs §5-4 3 sub-check:
    1. 1차 조건 (필수): |hip.L.x - hip.R.x| < `cfg.hip_x_distance_max` (0.05).
    2. 보조 조건 a: |shoulder.L.x - shoulder.R.x| < `cfg.shoulder_x_distance_max`
       (0.07).
    3. 보조 조건 b: torso yaw proxy = hip_x_distance / hip_shoulder_y_distance <
       `cfg.torso_yaw_proxy_max` (0.15). 분모 < 1e-6이면 자동 위반.

    torso yaw proxy 정의 (lock 8-B-6 α — docs §5-4 명시 부족):
    - 분자: |hip.L.x - hip.R.x| (= 1차 조건과 동일)
    - 분모: |mean(hip.L.y, hip.R.y) - mean(shoulder.L.y, shoulder.R.y)|
      → 분석측 무관 (양측 평균, "전신 비틀림" 의미)
    ⚠️ docs §5-4 본문 "좌우 hip 거리와 hip-shoulder 수직 거리의 비율" 직역 시
    분석측 단일 vs 양측 평균 모호. 본 구현 양측 평균 채택 (전신 비틀림 의미).
    별도 docs 보강 후보.

    frame-level 판정:
    - is_valid (정상) = primary AND (secondary_a OR secondary_b)
    - ⚠️ frame-level에서 `invalid_view` reason code 부여 X — 누적 `evaluate_side_view_accumulation`이
      1차/보조 위반 frame ratio 기반으로 결정 (lock 8-B-4 α + catch 8-B-2-α 패턴).

    Args:
        pl: PoseLandmarks (6 LandmarkPair, shoulder/hip만 사용 — nose/knee/ankle/
            heel/foot_index 미사용).
        cfg: SideViewConfig.

    Returns:
        FrameSideViewResult — 3 sub-check 결과 + check_values + is_valid.

    견고성 가드: 평가 예외 → logger.exception + failed-safe (is_valid=False +
    빈 passed_checks/check_values).
    """
    try:
        # ── 1차 조건: 좌우 hip x 거리 ─────────────────────
        hip_x_distance = abs(pl.hip.left.x - pl.hip.right.x)
        primary_ok = hip_x_distance < cfg.hip_x_distance_max

        # ── 보조 조건 a: 좌우 shoulder x 거리 ─────────────
        shoulder_x_distance = abs(pl.shoulder.left.x - pl.shoulder.right.x)
        secondary_a_ok = shoulder_x_distance < cfg.shoulder_x_distance_max

        # ── 보조 조건 b: torso yaw proxy (분모 epsilon 가드) ─
        hip_y_mean = (pl.hip.left.y + pl.hip.right.y) / 2.0
        shoulder_y_mean = (pl.shoulder.left.y + pl.shoulder.right.y) / 2.0
        hip_shoulder_y_distance = abs(hip_y_mean - shoulder_y_mean)

        # catch 8-B-2-β: 분모 < 1e-6 → torso_yaw_ratio = inf, 자동 위반 처리
        if hip_shoulder_y_distance < 1e-6:
            torso_yaw_ratio = float("inf")
            secondary_b_ok = False
        else:
            torso_yaw_ratio = hip_x_distance / hip_shoulder_y_distance
            secondary_b_ok = torso_yaw_ratio < cfg.torso_yaw_proxy_max

        # ── frame-level 정상 판정 ──────────────────────────
        is_valid = primary_ok and (secondary_a_ok or secondary_b_ok)

        passed_checks = {
            "primary_hip_x": primary_ok,
            "secondary_a_shoulder_x": secondary_a_ok,
            "secondary_b_torso_yaw": secondary_b_ok,
        }
        check_values = {
            "hip_x_distance": hip_x_distance,
            "shoulder_x_distance": shoulder_x_distance,
            "torso_yaw_ratio": torso_yaw_ratio,
            "hip_shoulder_y_distance": hip_shoulder_y_distance,
        }
        return FrameSideViewResult(
            passed_checks=passed_checks,
            check_values=check_values,
            is_valid=is_valid,
        )
    except Exception:
        logger.exception(
            "evaluate_frame_side_view 예외 (swallow, is_valid=False default)"
        )
        return FrameSideViewResult(
            passed_checks={
                "primary_hip_x": False,
                "secondary_a_shoulder_x": False,
                "secondary_b_torso_yaw": False,
            },
            check_values={},
            is_valid=False,
        )


# ============================================================
# §5-4 측면 구도 누적 평가 (Phase 8-B-2 — 2 severity 분기)
# ============================================================


def evaluate_side_view_accumulation(
    results: list[FrameSideViewResult],
    cfg: SideViewConfig,
) -> list[ReasonCodeEntry]:
    """§5-4 측면 구도 누적 평가 — 2 severity 분기 (docs/2-3-5 §5-4 + lock).

    docs §5-4 판정 매트릭스:
    - 1차 조건 위반 frame 비율 > 40% (= 1차 통과 ratio < 60%) →
      `invalid_view` failed (docs §5-4 명시).
    - 1차 통과 frame 중 보조 조건 (a + b) 모두 위반 frame ratio ≥ 0.3 →
      `invalid_view` low_confidence (⚠️ docs §5-4 명시 X, heuristic
      `cfg.secondary_violation_frame_ratio_max`).

    β 1차 통과 분모 (lock 8-B-7 β 채택):
    - 보조 위반 ratio 분모 = 1차 통과 frame 수 (전체 X)
    - "측면처럼 hip은 좁은데 어깨/torso 비틀림 의심" 케이스 명확히 잡음.

    반환 둘 다 (lock catch γ): failed + low_confidence 동시 트리거 가능 시
    list에 둘 다 entry 추가 — Phase 8-F (status 분기) + 응답 메시지에 위임.
    Phase 8-F가 §6-3 우선순위 (failed > low_confidence)로 status 결정 +
    응답 reason_codes 배열에 둘 다 또는 dedup 결정.

    Args:
        results: list[FrameSideViewResult] (`evaluate_frame_side_view` 누적).
        cfg: SideViewConfig.
            - `primary_condition_frame_ratio_min` (failed 임계, default 0.6)
            - `secondary_violation_frame_ratio_max` (low_conf 임계, default 0.3
              heuristic)

    Returns:
        list[ReasonCodeEntry]:
        - 전부 정상 또는 빈 입력 → `[]`
        - 1차 통과 ratio < 0.6 → `[ReasonCodeEntry('invalid_view', 'failed')]` 추가
        - 1차 통과 + 보조 모두 위반 ratio ≥ 0.3 →
          `[ReasonCodeEntry('invalid_view', 'low_confidence')]` 추가
        - 두 조건 동시 충족 → 둘 다 entry (list length 2, 둘 다 `invalid_view`)

    견고성 가드: 빈 list → `[]`, 평가 예외 → failed-safe (default 'failed' entry).
    """
    try:
        if not results:
            return []

        total = len(results)
        primary_pass_count = sum(1 for r in results if r.passed_checks.get("primary_hip_x", False))
        primary_pass_ratio = primary_pass_count / total

        entries: list[ReasonCodeEntry] = []

        # 1차 통과 ratio 미달 → failed 강도
        if primary_pass_ratio < cfg.primary_condition_frame_ratio_min:
            entries.append(
                ReasonCodeEntry(reason_code="invalid_view", severity="failed")
            )

        # 1차 통과 frame 중 보조 (a or b) 모두 위반 ratio (β 1차 통과 분모)
        if primary_pass_count > 0:
            secondary_violation_count = sum(
                1
                for r in results
                if r.passed_checks.get("primary_hip_x", False)
                and not (
                    r.passed_checks.get("secondary_a_shoulder_x", False)
                    or r.passed_checks.get("secondary_b_torso_yaw", False)
                )
            )
            secondary_violation_ratio = secondary_violation_count / primary_pass_count
            if secondary_violation_ratio >= cfg.secondary_violation_frame_ratio_max:
                entries.append(
                    ReasonCodeEntry(
                        reason_code="invalid_view", severity="low_confidence"
                    )
                )

        return entries
    except Exception:
        logger.exception(
            "evaluate_side_view_accumulation 예외 "
            "(swallow, invalid_view failed default 반환)"
        )
        return [
            ReasonCodeEntry(
                reason_code="invalid_view",
                severity=REASON_CODE_SEVERITY["invalid_view"],
            )
        ]


# ============================================================
# §5-5 카메라 흔들림 stride-level 평가 (Phase 8-C)
# ============================================================


def evaluate_camera_stability(
    landmarks_series: list[Optional[PoseLandmarks]],
    ic_indices: list[int],
    cfg: StrideExclusionConfig,
) -> list[ReasonCodeEntry]:
    """stride-level 카메라 흔들림 평가 (docs/2-3-5 §5-5 + docs/2-3-4 §10).

    각 stride 구간 [IC[n], IC[n+1]-1] frame들의 pelvis_x 변동을 산출, 임계 초과
    stride가 1개라도 있으면 `camera_unstable` (low_confidence) 트리거.

    docs 정합:
    - docs/2-3-5 §5-5: "pelvis_x의 stride 평균 대비 변동 ±30% 이내" → 위반 시
      `camera_unstable` 트리거 (low_confidence)
    - docs/2-3-4 §10 표: "pelvis_x 변동이 stride 평균 ±30% 이상" → "해당 stride 제외"
      + "camera_unstable 트리거 신호" (stride 제외 처리는 Phase 8-I integration 책임)

    ⚠️ docs §5-5 인용 catch (5-1): 본문 "2-3-4 문서 9장 기준 인용"은 잘못된 인용
    (실제 §10 표). docs 보강 후보.

    ⚠️ pelvis_x 변동 산출 방식 (lock 8-C-2 α — config docstring "stride 평균 대비
    변동" 직역):
        variation_ratio = (max(pelvis_x_in_stride) - min(pelvis_x_in_stride))
                          / mean(pelvis_x_in_stride)
    peak-to-peak ratio. docs 명시 부족, β(절대 폭) / γ(stddev/mean)도 가능 —
    파일럿 데이터 보정 후보.

    ⚠️ 산출 정책 (lock 8-C-1 α): 5 stride 중 1개라도 위반 → camera_unstable 트리거.
    docs §10 "해당 stride 제외 + 신호" 직역. 파일럿 5~10영상 후 β (비율 임계)
    전환 검토 후보.

    Args:
        landmarks_series: list[PoseLandmarks | None] (Phase 6 흐름 정합, None
            frame skip — lock catch 5-11 β).
        ic_indices: list[int] (compute_ic_indices 결과의 frame_index 추출 — 본
            함수는 ICResult가 아닌 단순 frame_index list만 받음).
        cfg: StrideExclusionConfig.
            - `camera_pelvis_x_stride_variation_max` (default 0.30)

    Returns:
        list[ReasonCodeEntry]:
        - 모든 stride 정상 또는 ic_indices 길이 < 2 → `[]`
        - 1개 이상 stride 위반 → `[ReasonCodeEntry('camera_unstable', 'low_confidence')]`

    ⚠️ Phase 8-I integration anchor: 본 함수는 landmarks_series 입력 받음. Phase 6
    PipelineResult는 현재 raw landmarks 미보존 (frame_results만). Phase 8-I 진입
    시 PipelineResult.landmarks_series 필드 추가 필요.

    견고성 가드:
    - ic_indices 길이 < 2 → 빈 list 반환 (stride 1개 미만)
    - stride 구간 내 None frame skip
    - stride frame 0개 또는 mean=0 → 해당 stride skip
    - 평가 예외 → failed-safe (camera_unstable entry 반환)
    """
    try:
        if len(ic_indices) < 2:
            return []

        for stride_idx in range(len(ic_indices) - 1):
            start = ic_indices[stride_idx]
            end = ic_indices[stride_idx + 1]  # IC[n+1] frame 미포함 (range)

            # stride 구간 frame의 pelvis_x (None skip)
            pelvis_x_list: list[float] = []
            for frame_idx in range(start, end):
                if frame_idx < 0 or frame_idx >= len(landmarks_series):
                    continue
                pl = landmarks_series[frame_idx]
                if pl is None:
                    continue
                # pelvis_x = 좌우 hip 양측 평균 (lock 8-C-7, §5-2/§5-4 일관)
                pelvis_x = (pl.hip.left.x + pl.hip.right.x) / 2.0
                pelvis_x_list.append(pelvis_x)

            if len(pelvis_x_list) < 2:
                # stride 내 유효 frame < 2 → 변동 계산 불가, skip
                continue

            x_min = min(pelvis_x_list)
            x_max = max(pelvis_x_list)
            x_mean = sum(pelvis_x_list) / len(pelvis_x_list)
            if abs(x_mean) < 1e-9:
                # mean ≈ 0 → ratio 무의미, skip (zero-division 가드)
                continue
            variation_ratio = (x_max - x_min) / x_mean

            if variation_ratio > cfg.camera_pelvis_x_stride_variation_max:
                # lock 8-C-1 α: 단일 stride 위반도 트리거
                return [
                    ReasonCodeEntry(
                        reason_code="camera_unstable",
                        severity=REASON_CODE_SEVERITY["camera_unstable"],
                    )
                ]

        return []
    except Exception:
        logger.exception(
            "evaluate_camera_stability 예외 (swallow, camera_unstable 반환)"
        )
        return [
            ReasonCodeEntry(
                reason_code="camera_unstable",
                severity=REASON_CODE_SEVERITY["camera_unstable"],
            )
        ]


# ============================================================
# §5-6 지표 변동성 stride-level 평가 (Phase 8-C)
# ============================================================


def evaluate_metric_variability(
    foot_degs: list[float],
    knee_degs: list[float],
    trunk_degs: list[float],
    cfg: MetricVariabilityConfig,
) -> list[ReasonCodeEntry]:
    """stride-level 지표 변동성 평가 (docs/2-3-5 §5-6).

    3 metric stride 간 표본 표준편차 (`ddof=1`) 임계 비교:
    - Foot angle stddev > `cfg.foot_stddev_max_deg` (5°) → `unstable_foot_angle`
    - Knee flexion stddev > `cfg.knee_stddev_max_deg` (7°) → `unstable_knee_angle`
    - Trunk lean stddev > `cfg.trunk_stddev_max_deg` (4°) → `unstable_trunk_angle`

    모두 low_confidence 강도 (docs §8-6 정합).

    ⚠️ ddof=1 채택 사유 (lock 5-9): 표본 표준편차 — Python convention default
    ddof=0 (모집단) 과 다름. 본 측정 stride 5개는 무한 모집단 표본이므로 표본
    stddev (n-1 분모) 채택. docs §5-6 명시 X, docs 보강 후보.

    입력 형태 (lock 8-C-8 α — Phase 7-A `compute_angle_stats` 패턴 일관):
    list[float] (각 stride의 metric deg 값). NaN 자동 제외 (numpy.nanstd 사용).

    Args:
        foot_degs: 각 stride foot_strike_deg list (분석측 단일값).
        knee_degs: 각 stride knee_flexion_deg list.
        trunk_degs: 각 stride trunk_lean_deg list.
        cfg: MetricVariabilityConfig.

    Returns:
        list[ReasonCodeEntry]:
        - 3 metric stddev 모두 임계 이내 → `[]`
        - 각 위반 metric별로 entry 추가 (최대 3개)
        - 빈 list / 길이 < 2 list → 해당 metric skip (n=1 stddev 미정의)

    견고성 가드:
    - 입력 list 길이 < 2 → 해당 metric skip
    - NaN 자동 제외 (np.nanstd)
    - 유효 값 < 2 → skip
    - 평가 예외 → failed-safe (3 metric entry 모두 반환)
    """
    try:
        entries: list[ReasonCodeEntry] = []

        def _stddev_safe(values: list[float]) -> Optional[float]:
            """ddof=1 stddev, NaN 제외, 유효 < 2 → None."""
            finite = [v for v in values if not (v != v)]  # NaN 검출 (v != v)
            # 또는 math.isfinite 사용 — 명시적
            finite = [v for v in values if np.isfinite(v)]
            if len(finite) < 2:
                return None
            return float(np.std(finite, ddof=1))

        foot_std = _stddev_safe(foot_degs)
        if foot_std is not None and foot_std > cfg.foot_stddev_max_deg:
            entries.append(
                ReasonCodeEntry(
                    reason_code="unstable_foot_angle",
                    severity=REASON_CODE_SEVERITY["unstable_foot_angle"],
                )
            )

        knee_std = _stddev_safe(knee_degs)
        if knee_std is not None and knee_std > cfg.knee_stddev_max_deg:
            entries.append(
                ReasonCodeEntry(
                    reason_code="unstable_knee_angle",
                    severity=REASON_CODE_SEVERITY["unstable_knee_angle"],
                )
            )

        trunk_std = _stddev_safe(trunk_degs)
        if trunk_std is not None and trunk_std > cfg.trunk_stddev_max_deg:
            entries.append(
                ReasonCodeEntry(
                    reason_code="unstable_trunk_angle",
                    severity=REASON_CODE_SEVERITY["unstable_trunk_angle"],
                )
            )

        return entries
    except Exception:
        logger.exception(
            "evaluate_metric_variability 예외 (swallow, 3 metric entry 반환)"
        )
        return [
            ReasonCodeEntry(
                reason_code="unstable_foot_angle",
                severity=REASON_CODE_SEVERITY["unstable_foot_angle"],
            ),
            ReasonCodeEntry(
                reason_code="unstable_knee_angle",
                severity=REASON_CODE_SEVERITY["unstable_knee_angle"],
            ),
            ReasonCodeEntry(
                reason_code="unstable_trunk_angle",
                severity=REASON_CODE_SEVERITY["unstable_trunk_angle"],
            ),
        ]


# ============================================================
# §5-7 IC 검증 stride-level 평가 (Phase 8-D — severity 혼합 첫 등장)
# ============================================================


def evaluate_ic_validation(
    ic_confidences: list[ICConfidence],
    trunk_window_valid_ratios: list[float],
    cfg: ICValidationConfig,
) -> list[ReasonCodeEntry]:
    """stride-level IC 검증 (docs/2-3-5 §5-7).

    3 reason_code 통합 평가 (Phase 8-C `evaluate_metric_variability` 패턴):
    1. `insufficient_stride` (failed, docs §8-5): `len(ic_confidences) <
       cfg.min_total_ic` (default 3). lock 8-D-15 α — 전체 ICResult 카운트 (confidence 무관).
    2. `low_ic_confidence` (low_confidence, docs §8-5): high/medium 신뢰도 IC 수 <
       `cfg.min_high_medium_confidence_ic` (default 2).
    3. `insufficient_window` (low_confidence, docs §8-5 + docs/2-3-4 §10): 단일
       stride라도 trunk_lean window valid ratio < `cfg.trunk_lean_window_min_valid_ratio`
       (default 0.5) → 트리거 (lock 8-D-3 α 단일 위반 트리거, Phase 8-C 8-C-1 α 패턴).

    ⚠️ severity 혼합 첫 등장 (catch 7-2):
    한 함수에서 failed (insufficient_stride) + low_confidence (low_ic_confidence,
    insufficient_window) 둘 다 산출. Phase 8-B-2 `invalid_view`는 동일 reason_code의
    2강도였으나, 본 8-D는 **다른 reason_code의 서로 다른 severity**. Phase 8-F status
    분기 시 failed 1개라도 → status=failed.

    ⚠️ stride time 영역 분리 (lock 8-D-1 α, catch 7-1):
    docs/2-3-4 §10 표는 stride time > 1.5s → "본 stride 분석 제외 + insufficient_stride
    신호" 명시. 본 함수는 IC count만 평가 — stride time 검사는 Phase 8-I integration
    scope (stride exclusion 처리 후 남은 stride 수가 insufficient_stride에 반영).
    docs §5-7 표는 stride time 누락 — **docs 보강 후보**.

    ⚠️ insufficient_window 민감도 (lock 8-D-3 α):
    단일 stride window valid ratio < 0.5도 트리거. 파일럿 5~10영상 후 β (비율 임계)
    전환 검토 후보. Phase 8-C 8-C-1 α 패턴 일관.

    ⚠️ 빈 입력 처리:
    - `ic_confidences = []` (lock 8-D-8 B): insufficient_stride + low_ic_confidence
      둘 다 트리거 (정보 보존, Phase 8-F가 failed 우선 결정).
    - `trunk_window_valid_ratios = []` (lock 8-D-9 A): insufficient_window skip
      (trunk_lean 산출 자체 안 됐다는 신호 — 다른 reason code가 책임. false positive 회피).

    ⚠️ trunk_window_valid_ratios 입력 (catch 7-4):
    각 stride의 `TrunkLeanResult.window_valid_count / window_total_count` ratio list.
    `window_total_count = 0` (영상 경계) 가능성 — 호출자(Phase 8-I)가 안전 산출
    책임 (ratio 0 또는 1로 fallback). 본 함수는 ratio list 받기만.

    ⚠️ Phase 8-I integration anchor (catch 7-5, 7-11):
    본 함수 입력은 Pipeline에서 추출:
    - `ic_confidences`: ICResult list의 `.confidence` 추출
    - `trunk_window_valid_ratios`: TrunkLeanResult list의 `window_valid_count /
      window_total_count` 산출
    Phase 8-I 진입 시 PipelineResult에 ICResult list + TrunkLeanResult list 보존
    필드 추가 필요 (Phase 8-C `landmarks_series` anchor와 함께 누적 2건).

    Args:
        ic_confidences: 전체 IC 신뢰도 list (각 stride의 ICResult.confidence).
            len → insufficient_stride 판정.
            sum(c in ('high','medium')) → low_ic_confidence 판정.
        trunk_window_valid_ratios: 각 stride의 trunk_lean window valid ratio list
            (TrunkLeanResult.window_valid_count / window_total_count).
            any(< 0.5) → insufficient_window 트리거.
        cfg: ICValidationConfig.
            - `min_total_ic` (default 3)
            - `min_high_medium_confidence_ic` (default 2)
            - `trunk_lean_window_min_valid_ratio` (default 0.5)

    Returns:
        list[ReasonCodeEntry]: 트리거된 reason code별 entry (최대 3개).
        - 모두 통과 → `[]`
        - 빈 ic_confidences → `[insufficient_stride(failed), low_ic_confidence(low_conf)]`

    견고성 가드: 평가 예외 → failed-safe (3 reason code entry 모두 반환).
    """
    try:
        entries: list[ReasonCodeEntry] = []

        # 1. insufficient_stride (failed) — 전체 ICResult 카운트 (lock 8-D-15 α)
        if len(ic_confidences) < cfg.min_total_ic:
            entries.append(
                ReasonCodeEntry(
                    reason_code="insufficient_stride",
                    severity=REASON_CODE_SEVERITY["insufficient_stride"],
                )
            )

        # 2. low_ic_confidence (low_confidence) — high/medium 카운트
        high_medium_count = sum(
            1 for c in ic_confidences if c in ("high", "medium")
        )
        if high_medium_count < cfg.min_high_medium_confidence_ic:
            entries.append(
                ReasonCodeEntry(
                    reason_code="low_ic_confidence",
                    severity=REASON_CODE_SEVERITY["low_ic_confidence"],
                )
            )

        # 3. insufficient_window (low_confidence) — 단일 stride 위반 트리거 (lock 8-D-3 α)
        # 빈 list → skip (lock 8-D-9 A, false positive 회피)
        if trunk_window_valid_ratios:
            if any(
                r < cfg.trunk_lean_window_min_valid_ratio
                for r in trunk_window_valid_ratios
            ):
                entries.append(
                    ReasonCodeEntry(
                        reason_code="insufficient_window",
                        severity=REASON_CODE_SEVERITY["insufficient_window"],
                    )
                )

        return entries
    except Exception:
        logger.exception(
            "evaluate_ic_validation 예외 (swallow, 3 reason code entry 반환)"
        )
        return [
            ReasonCodeEntry(
                reason_code="insufficient_stride",
                severity=REASON_CODE_SEVERITY["insufficient_stride"],
            ),
            ReasonCodeEntry(
                reason_code="low_ic_confidence",
                severity=REASON_CODE_SEVERITY["low_ic_confidence"],
            ),
            ReasonCodeEntry(
                reason_code="insufficient_window",
                severity=REASON_CODE_SEVERITY["insufficient_window"],
            ),
        ]


# ============================================================
# §4 추적 안정성 sliding window 평가 (Phase 8-E scope γ — visibility 1 신호 2 reason_code)
# ============================================================


def evaluate_tracking_stability(
    visibility_per_frame: list[float],
    fps: float,
    cfg: TrackingStabilityConfig,
) -> list[ReasonCodeEntry]:
    """sliding window 추적 안정성 평가 (docs/2-3-5 §4 scope γ).

    docs §4 4 reason_code 중 scope γ (Phase 8-E lock 8-E-1) 2 reason_code 산출:
    1. `target_lost` (failed, docs §8-4): 1초 sliding window 평균 visibility가
       5초 이상 연속 < 0.4 (lock 8-E-6 α 정확 해석 — window of windows).
    2. `background_person_interference` (low_confidence, docs §8-4): visibility
       borderline (0.4 ≤ v < 0.6) frame 비율 ≥ 30% (lock 8-E-7 α simplification —
       scope γ는 scale 신호 미산출).

    ⚠️ Phase 9 분할 anchor (lock 8-E-18, 8-E-2 β):
    docs §4 4 reason_code 중 2개 Phase 9 분리:
    - `target_switch_detected`: pelvis 잔차 spike + scale spike + visibility 일시
      붕괴 3 신호 동시 AND. scale 산출 부재 catch 7-2.
    - `unstable_landmark_sequence`: heel/foot mid-stance 잔차. mid-stance 시점
      정의 어려움 catch 7-3.
    scope γ 채택으로 catch 7-2/7-3 해소 — visibility 1 신호로 단순화.

    ⚠️ docs §4-3 simplification (catch 7-7, lock 8-E-17):
    background_person_interference 직역: visibility borderline AND scale 미세 변동.
    scope γ는 visibility borderline만 — scale 신호 미사용. 파일럿 5~10영상 후
    scale 신호 추가 검토 후보. false positive 우려.

    ⚠️ docs §4-2-1 평활화 원칙 (lock 8-E-19):
    scope γ는 sliding window만 사용 — docs §4-2-1 잔차 평활화 미적용. 파일럿 후
    적용 검토 후보.

    visibility_per_frame 산출 정의 (lock 8-E-5 δ, 호출자 책임):
    - 각 frame의 주요 12 LandmarkPair (shoulder/hip/knee/ankle/heel/foot_index 양측)
      visibility 평균
    - §5-1 overall_avg 패턴 일관
    - 호출자(Phase 8-I)가 landmarks_series에서 산출

    ⚠️ sliding window 정확 해석 (lock 8-E-6 α, catch 7-5):
    - α (본 구현): 1초 window 평균 < 0.4가 5초 연속 (window of windows)
    - β: visibility frame < 0.4가 5*fps 연속 (단순)
    α 채택 — docs 직역 더 정확.

    ⚠️ Phase 8-I integration anchor (catch 7-6, anchor 누적 ↓):
    landmarks_series 보존(Phase 8-C anchor) 시 visibility_per_frame 호출자 산출
    가능. 추가 필드 X. Phase 8-C/8-D anchor와 통합:
    - Phase 8-C: landmarks_series (Phase 8-E 신호도 추출)
    - Phase 8-D: ICResult list + TrunkLeanResult list
    - Phase 8-E: visibility_per_frame (landmarks_series 파생)

    Args:
        visibility_per_frame: 각 frame의 주요 12점 visibility 평균 list.
        fps: 영상 fps. ≤ 1e-6 시 30.0 fallback (lock 8-E-14).
        cfg: TrackingStabilityConfig.
            - `visibility_window_seconds` (default 1.0)
            - `target_lost_seconds` (default 5.0)
            - `target_lost_visibility_threshold` (default 0.4)
            - `visibility_borderline_low` (default 0.4)
            - `visibility_borderline_high` (default 0.6)
            - `visibility_borderline_violation_ratio_max` (default 0.30)

    Returns:
        list[ReasonCodeEntry]:
        - 모두 통과 또는 빈 입력 → `[]` (lock 8-E-13)
        - target_lost 트리거 → `[ReasonCodeEntry('target_lost', 'failed')]` 추가
        - background_person_interference 트리거 →
          `[ReasonCodeEntry('background_person_interference', 'low_confidence')]` 추가
        - 둘 다 트리거 → 둘 다 entry (severity 혼합)
        - len(visibility_per_frame) < window_5s_frames → target_lost 산출 X
          (충분한 frame 없음)

    견고성 가드: 평가 예외 → failed-safe (2 reason code entry 모두 반환).
    """
    try:
        if not visibility_per_frame:
            return []

        # lock 8-E-14: fps_safe fallback (Phase 7-A 패턴)
        fps_safe = fps if fps > 1e-6 else 30.0

        entries: list[ReasonCodeEntry] = []
        n = len(visibility_per_frame)

        # ── 1. target_lost (lock 8-E-6 α: window of windows) ────────
        # window frame 수 (1초 / 5초)
        window_1s_frames = max(1, int(fps_safe * cfg.visibility_window_seconds))
        window_5s_frames = max(1, int(fps_safe * cfg.target_lost_seconds))

        # 1초 window 평균 visibility 시리즈 (manual loop)
        if n >= window_1s_frames:
            window_avgs: list[float] = []
            for i in range(n - window_1s_frames + 1):
                avg = sum(visibility_per_frame[i:i + window_1s_frames]) / window_1s_frames
                window_avgs.append(avg)

            # 5초 연속 < threshold 검사 (1초 window 평균 시리즈에서 5*fps 연속)
            threshold = cfg.target_lost_visibility_threshold
            consecutive_count = 0
            for avg in window_avgs:
                if avg < threshold:
                    consecutive_count += 1
                    if consecutive_count >= window_5s_frames:
                        entries.append(
                            ReasonCodeEntry(
                                reason_code="target_lost",
                                severity=REASON_CODE_SEVERITY["target_lost"],
                            )
                        )
                        break
                else:
                    consecutive_count = 0

        # ── 2. background_person_interference (lock 8-E-7 α scope γ) ────
        # visibility borderline (low ≤ v < high) frame 비율
        lo = cfg.visibility_borderline_low
        hi = cfg.visibility_borderline_high
        borderline_count = sum(1 for v in visibility_per_frame if lo <= v < hi)
        borderline_ratio = borderline_count / n
        if borderline_ratio >= cfg.visibility_borderline_violation_ratio_max:
            entries.append(
                ReasonCodeEntry(
                    reason_code="background_person_interference",
                    severity=REASON_CODE_SEVERITY["background_person_interference"],
                )
            )

        return entries
    except Exception:
        logger.exception(
            "evaluate_tracking_stability 예외 (swallow, 2 reason code entry 반환)"
        )
        return [
            ReasonCodeEntry(
                reason_code="target_lost",
                severity=REASON_CODE_SEVERITY["target_lost"],
            ),
            ReasonCodeEntry(
                reason_code="background_person_interference",
                severity=REASON_CODE_SEVERITY["background_person_interference"],
            ),
        ]


# ============================================================
# §6 status 분기 + §8-7 primary_reason_code 우선순위 (Phase 8-F SoT)
# ============================================================


REASON_CODE_PRIORITY: list[tuple[ReasonCode, "Severity"]] = [
    # === failed 그룹 (1~5, docs §8-7-2) ===
    # 그룹 1: 분석 대상자 추적 실패
    # ⚠️ target_switch_detected (failed) — Phase 9 미구현 (scale 산출 정의 필요)
    ("target_lost", "failed"),
    # 그룹 2: 발 가시성 실패
    ("foot_out_of_frame", "failed"),
    ("foot_not_visible", "failed"),
    # 그룹 3: 전신 가시성 실패
    ("body_not_fully_visible", "failed"),
    # 그룹 4: 측면 구도 실패 (invalid_view context-dependent failed 강도)
    ("invalid_view", "failed"),
    # 그룹 5: 메타데이터/IC 실패
    # ⚠️ 메타데이터 3종 (too_short > low_resolution > low_fps) — docs/2-3-1 영역 미구현
    ("insufficient_stride", "failed"),

    # === low_confidence 그룹 (6~10, docs §8-7-3) ===
    # 그룹 6: 추적 borderline
    # ⚠️ unstable_landmark_sequence (low_confidence) — Phase 9 미구현 (mid-stance 정의 필요)
    ("background_person_interference", "low_confidence"),
    # 그룹 7: 측면 구도 borderline (invalid_view context-dependent low_confidence 강도)
    ("invalid_view", "low_confidence"),
    # 그룹 8: 일반 품질 저하
    ("camera_unstable", "low_confidence"),
    ("low_landmark_visibility", "low_confidence"),
    ("lower_body_not_visible", "low_confidence"),
    ("upper_body_not_visible", "low_confidence"),
    # 그룹 9: IC 신뢰도 부족
    ("low_ic_confidence", "low_confidence"),
    ("insufficient_window", "low_confidence"),
    # 그룹 10: 지표 변동성
    ("unstable_foot_angle", "low_confidence"),
    ("unstable_knee_angle", "low_confidence"),
    ("unstable_trunk_angle", "low_confidence"),
]
"""docs/2-3-5 §8-7-2 + §8-7-3 primary_reason_code 우선순위 단일 정답 (Phase 8-F SoT).

본 list는 docs §8-7-2 (failed 그룹 1~5) + §8-7-3 (low_confidence 그룹 6~10) 그룹
우선순위 + 그룹 내 우선순위를 flat list로 구조화 — 17 entry.

자료구조 채택 (Phase 8-F lock 8-F-2 α):
- `list[tuple[ReasonCode, Severity]]` flat — `invalid_view` context-dependent 2 entry
  (failed/low_confidence 별도). linear search 단순.
- 그룹 1~5 = failed (위), 그룹 6~10 = low_confidence (아래) — docs §8-7-1 정합.

그룹 내 우선순위 — docs §8-7-1 "사용자 즉시 해결 가능 코드 > 추상 코드":
- 그룹 2: foot_out_of_frame ("카메라 멀리") > foot_not_visible ("발 안 보임")
- 그룹 8: camera_unstable ("카메라 흔들림") > low_landmark_visibility ("신뢰도 낮음")
  > lower_body_not_visible > upper_body_not_visible
- 그룹 9: low_ic_confidence > insufficient_window
- 그룹 10: foot > knee > trunk (지표 변동성)

미등록 reason_code (lock 8-F-7 β YAGNI):
- Phase 9 진입 시: target_switch_detected (그룹 1) / unstable_landmark_sequence (그룹 6)
- 메타데이터 진입 시: too_short / low_resolution / low_fps (그룹 5 앞쪽)

docs §8-7-4 5 예시 정합 (sanity/pytest 박힘):
- foot_out_of_frame + foot_not_visible + low_landmark_visibility → foot_out_of_frame (그룹 2)
- target_switch_detected + low_landmark_visibility → target_switch_detected (그룹 1) ⚠️ Phase 9
- body_not_fully_visible + low_landmark_visibility → body_not_fully_visible (그룹 3)
- unstable_foot + unstable_knee + low_ic_confidence → low_ic_confidence (그룹 9 > 10)
- camera_unstable + unstable_landmark_sequence → unstable_landmark_sequence (그룹 6 > 8) ⚠️ Phase 9
"""


@dataclass(frozen=True)
class ResponseStatusResult:
    """docs/2-3-7 §6 status + docs §8-7 primary_reason_code 산출 결과.

    Phase 8-F lock 8-F-5 α — dataclass 채택 (가독성 + 향후 확장 여지).
    Phase 8-B-1 ReasonCodeEntry 패턴 일관 — frozen=True (불변 보장).

    Phase 7-A AnalysisResultMessage schema 정합:
    - status → `AnalysisResultMessage.status` (Literal['success', 'low_confidence', 'failed'])
    - primary_reason_code → `AnalysisResultMessage.primary_reason_code` (Optional[str])
    - reason_codes → `AnalysisResultMessage.reason_codes` (list[str])

    Phase 8-I integration 시 Pipeline에서 사용:
    1. evaluate_* 함수 호출 → list[ReasonCodeEntry] 누적
    2. compute_response_status(entries) → ResponseStatusResult
    3. AnalysisResultMessage 조립 (status/primary/reason_codes 필드 채움)

    Phase 8-G feedback_engine 입력으로도 사용:
    - status: docs/2-3-6 §3-4 status별 피드백 출력 정책 분기
    - primary_reason_code: docs §8-3 system_info 메시지 매핑
    - reason_codes: feedback context

    Attributes:
        status: 'success' / 'low_confidence' / 'failed'.
        primary_reason_code: 사용자 노출 대표 reason code (1개). success 시 None.
        reason_codes: dedup + PRIORITY 정렬된 reason_code list. success 시 빈 list.
    """

    status: AnalysisStatus
    primary_reason_code: Optional[ReasonCode]
    reason_codes: list[ReasonCode]


def compute_response_status(
    entries: list[ReasonCodeEntry],
) -> ResponseStatusResult:
    """누적 list[ReasonCodeEntry] → ResponseStatusResult 산출.

    docs/2-3-5 §6 status 분기 + docs §8-7 primary_reason_code 우선순위 통합 산출.
    Phase 8 묶음 중 가장 중요한 SoT 결정 — 발표 Q&A 핵심 답안지.

    ⚠️ docs §6 status 분기 (3 단계 우선순위):
    1. failed severity entry ≥ 1 → status='failed'
    2. low_confidence severity entry ≥ 1 (failed 0) → status='low_confidence'
    3. 빈 list → status='success' (모든 검사 통과)

    ⚠️ docs §8-7 primary_reason_code 우선순위 (lock 8-F-2 α):
    - `REASON_CODE_PRIORITY` linear search (17 entry 순회, lock 8-F-13)
    - (reason_code, severity) 튜플 기반 — `invalid_view` context-dependent 정확 처리
    - 그룹 1~5 failed > 그룹 6~10 low_confidence > 그룹 내 우선순위
    - 빈 list → primary_reason_code=None

    ⚠️ reason_codes 배열 (lock 8-F-3/4 α):
    - dedup (set 변환, 중복 reason_code 제거 — invalid_view 2 severity 동시 트리거 시)
    - PRIORITY 순서 정렬 (primary와 일관)
    - 빈 list 가능 (success 시)

    ⚠️ fallback 동작 (방어적, lock 8-F 미언급):
    - PRIORITY 미등록 reason_code 입력 시 (현재 16 ReasonCode 모두 등록되어 발생 불가):
      · primary: entries[0].reason_code fallback
      · reason_codes: PRIORITY 정렬 후 미등록 코드 끝에 append
    - Phase 9 + 메타데이터 진입 시 PRIORITY 확장으로 자연 해소

    ⚠️ docs §8-7-4 5 예시 정합 (sanity/pytest 박힘 — 발표 Q&A 답안지):
    예시 1: [foot_out_of_frame, foot_not_visible, low_landmark_visibility]
            → primary=foot_out_of_frame (그룹 2 첫)
    예시 2: [target_switch_detected, low_landmark_visibility] ⚠️ Phase 9
            → primary=target_switch_detected (그룹 1)
    예시 3: [body_not_fully_visible, low_landmark_visibility]
            → primary=body_not_fully_visible (그룹 3)
    예시 4: [unstable_foot_angle, unstable_knee_angle, low_ic_confidence]
            → primary=low_ic_confidence (그룹 9 > 10)
    예시 5: [camera_unstable, unstable_landmark_sequence] ⚠️ Phase 9
            → primary=unstable_landmark_sequence (그룹 6 > 8)

    Args:
        entries: 누적 list[ReasonCodeEntry] (Phase 8-A~8-E 산출 + Pipeline integration).

    Returns:
        ResponseStatusResult — status / primary_reason_code / reason_codes.

    견고성 가드: 빈 list → success 결과. PRIORITY 미등록 코드 fallback 처리.
    """
    # 빈 entries → success (lock 8-F-6)
    if not entries:
        return ResponseStatusResult(
            status="success",
            primary_reason_code=None,
            reason_codes=[],
        )

    # ── docs §6 status 분기 (3 단계 우선순위) ─────────────
    has_failed = any(e.severity == "failed" for e in entries)
    status: AnalysisStatus = "failed" if has_failed else "low_confidence"

    # ── docs §8-7 primary_reason_code linear search ─────
    # entry set (code, severity) tuple — invalid_view context-dependent 정확 lookup
    entry_set = {(e.reason_code, e.severity) for e in entries}

    primary_reason_code: Optional[ReasonCode] = None
    for code, sev in REASON_CODE_PRIORITY:
        if (code, sev) in entry_set:
            primary_reason_code = code
            break
    # fallback (방어적 — PRIORITY 미등록 reason_code 입력 시)
    if primary_reason_code is None:
        primary_reason_code = entries[0].reason_code

    # ── reason_codes dedup + PRIORITY 정렬 (lock 8-F-3/4) ─
    unique_codes = {e.reason_code for e in entries}
    reason_codes: list[ReasonCode] = []
    seen: set[ReasonCode] = set()
    # PRIORITY 순서로 정렬 (등록된 코드)
    for code, _sev in REASON_CODE_PRIORITY:
        if code in unique_codes and code not in seen:
            reason_codes.append(code)
            seen.add(code)
    # fallback (PRIORITY 미등록 코드는 끝에 append, 방어적)
    for code in unique_codes:
        if code not in seen:
            reason_codes.append(code)
            seen.add(code)

    return ResponseStatusResult(
        status=status,
        primary_reason_code=primary_reason_code,
        reason_codes=reason_codes,
    )


# ============================================================
# §4-3 target_switch_detected (Phase 9-A — 3 신호 AND sliding window)
# ============================================================
#
# docs/2-3-5 §4-3 트리거 조건 (모두 동시):
# - pelvis 잔차 spike > 0.15 (cfg.pelvis_residual_spike)
# - |scale 변동률| > 0.20 (cfg.scale_spike)
# - visibility 일시 붕괴 < 0.4 (cfg.target_lost_visibility_threshold)
#
# Phase 9-A heuristic (lock B-1/B-2 anchor): 위 3 조건이 본 frame 수 이상 연속
# 발생 시 발화 (cfg.target_switch_consecutive_frames, default 5). docs §4-3 본문은
# "일시 붕괴"만 명시 — frame 수 정책 docs 보강 후보.
#
# 계산식 lock (anchor B-3):
# - pelvis 잔차: α sliding window 평균 대비 |차이|
# - scale: γ hip-shoulder 수직 거리 (Phase 8-B-2 torso_yaw_proxy 분모 재사용, 측면 robust)
# - window 산출: α 단순 산술 평균 (Phase 8-E `evaluate_tracking_stability` 일관)


def _compute_pelvis_residual_series(
    landmarks_series: list[Optional[PoseLandmarks]],
    window_frames: int,
) -> list[float]:
    """각 frame pelvis_x의 sliding window 평균 대비 |잔차| (anchor B-3 결정 1 α).

    pelvis_x = (hip.L.x + hip.R.x) / 2 — Phase 8-B-2 torso_yaw 분자 패턴 재사용.
    window: leading (t - window_frames + 1 ~ t).
    None frame: pelvis_x = 0.0 (별도 검사 X — visibility check가 자연스럽게 잡음).

    Args:
        landmarks_series: list[PoseLandmarks | None] (Phase 8-C/D/E 패턴).
        window_frames: sliding window 크기 (frame). cfg.pelvis_window_seconds * fps.

    Returns:
        list[float] (n=len(landmarks_series)) — 각 frame |residual|.
    """
    pelvis_x_list: list[float] = []
    for pl in landmarks_series:
        if pl is None:
            pelvis_x_list.append(0.0)
        else:
            pelvis_x_list.append((pl.hip.left.x + pl.hip.right.x) / 2.0)

    n = len(pelvis_x_list)
    residual: list[float] = [0.0] * n
    for i in range(n):
        start = max(0, i - window_frames + 1)
        window = pelvis_x_list[start:i + 1]
        if not window:
            continue
        window_mean = sum(window) / len(window)
        residual[i] = abs(pelvis_x_list[i] - window_mean)
    return residual


def _compute_scale_series(
    landmarks_series: list[Optional[PoseLandmarks]],
    window_frames: int,
) -> list[float]:
    """각 frame hip-shoulder 수직 거리의 sliding window 평균 대비 변동률 (anchor B-3 결정 2 γ).

    scale[t] = |mean(hip.y) - mean(shoulder.y)| — Phase 8-B-2 `torso_yaw_proxy`
    분모 패턴 재사용 (측면 robust, 키에 비례 안정 척도).
    변동률: (scale[t] - window_mean) / window_mean (signed).
    epsilon 가드: window_mean < 1e-6 → 변동률 0.0 (zero-division 회피).

    Args:
        landmarks_series: list[PoseLandmarks | None].
        window_frames: sliding window 크기 (frame). cfg.scale_window_seconds * fps.

    Returns:
        list[float] (n=len) — signed 변동률 (호출자 abs() 검사).
    """
    scale_list: list[float] = []
    for pl in landmarks_series:
        if pl is None:
            scale_list.append(0.0)
        else:
            hip_y = (pl.hip.left.y + pl.hip.right.y) / 2.0
            shoulder_y = (pl.shoulder.left.y + pl.shoulder.right.y) / 2.0
            scale_list.append(abs(hip_y - shoulder_y))

    n = len(scale_list)
    variation: list[float] = [0.0] * n
    for i in range(n):
        start = max(0, i - window_frames + 1)
        window = scale_list[start:i + 1]
        if not window:
            continue
        window_mean = sum(window) / len(window)
        if abs(window_mean) < 1e-6:
            continue  # 변동률 0.0 fallback
        variation[i] = (scale_list[i] - window_mean) / window_mean
    return variation


def _compute_visibility_per_frame_local(
    landmarks_series: list[Optional[PoseLandmarks]],
) -> list[float]:
    """각 frame 12 LandmarkPair visibility 평균 (frame-level, window X).

    ⚠️ Phase 9-A anchor B-4 lock — visibility는 frame-level 절대값 평가
    (붕괴 신호). pelvis/scale은 변동성 신호 (window baseline 필요)와 의미 분화.

    ⚠️ Phase 8-H `pipeline._compute_visibility_per_frame`와 동일 시그니처/계산식.
    향후 두 helper 통합 anchor (현재 quality_gate / pipeline 양쪽 중복).

    None frame: visibility = 0.0 (붕괴 시그널 자연 반영, 5 frame 연속 trigger
    조건에 자연스럽게 기여).

    Args:
        landmarks_series: list[PoseLandmarks | None].

    Returns:
        list[float] (n=len) — 각 frame 12점 평균 visibility (0.0~1.0).
    """
    vis_per_frame: list[float] = []
    for pl in landmarks_series:
        if pl is None:
            vis_per_frame.append(0.0)
            continue
        vis_sum = (
            pl.shoulder.left.visibility + pl.shoulder.right.visibility
            + pl.hip.left.visibility + pl.hip.right.visibility
            + pl.knee.left.visibility + pl.knee.right.visibility
            + pl.ankle.left.visibility + pl.ankle.right.visibility
            + pl.heel.left.visibility + pl.heel.right.visibility
            + pl.foot_index.left.visibility + pl.foot_index.right.visibility
        ) / 12.0
        vis_per_frame.append(vis_sum)
    return vis_per_frame


def evaluate_target_switch(
    landmarks_series: list[Optional[PoseLandmarks]],
    fps: float,
    cfg: TrackingStabilityConfig,
) -> Optional[ReasonCodeEntry]:
    """docs/2-3-5 §4-3 target_switch_detected 평가 (Phase 9-A).

    3 신호 AND 조건 + 연속 frame 정책:
    - pelvis 잔차 (sliding window 평균 대비 |차이|) > `cfg.pelvis_residual_spike` (0.15)
    - |scale 변동률| (hip-shoulder 수직 거리, window 평균 대비) > `cfg.scale_spike` (0.20)
    - visibility window 평균 < `cfg.target_lost_visibility_threshold` (0.4)
    - 위 3 조건이 `cfg.target_switch_consecutive_frames` (5) 이상 연속 발생

    ⚠️ docs §4-3 본문은 "동시 발생" + "일시 붕괴"만 명시. 5 frame 연속 정책은
    Phase 9-A heuristic (false positive 방지: 옆 사람 통과 4 frame은 trigger X,
    옆 사람 정착 10 frame trigger O). docs 보강 후보 anchor (Phase D 패턴).

    ⚠️ 계산식 lock (anchor B-3):
    - pelvis 잔차 α: sliding window 평균 대비 |차이|
    - scale γ: hip-shoulder 수직 거리 (Phase 8-B-2 torso_yaw_proxy 분모, 측면 robust)
    - window 산출 α: 단순 산술 평균 (Phase 8-E `evaluate_tracking_stability` 일관)

    Args:
        landmarks_series: list[PoseLandmarks | None] (Phase 8-C/D/E 패턴).
        fps: 영상 fps. ≤ 1e-6 시 30.0 fallback (Phase 7-A / 8-E 패턴 일관).
        cfg: TrackingStabilityConfig (Phase 8-E 동일 cfg, Step 1 lock β).
            - pelvis_residual_spike (0.15)
            - scale_spike (0.20)
            - target_lost_visibility_threshold (0.4)
            - pelvis_window_seconds (0.5) / scale_window_seconds (1.0) /
              visibility_window_seconds (1.0)
            - target_switch_consecutive_frames (5, Phase 9-A 신규)

    Returns:
        Optional[ReasonCodeEntry]:
        - 3 AND 조건이 5 frame 이상 연속 → ReasonCodeEntry('target_switch_detected', 'failed')
        - 빈 입력 또는 trigger 안 됨 → None

    견고성 가드: 평가 예외 → logger.exception + failed-safe (target_switch_detected
    entry 반환).
    """
    try:
        if not landmarks_series:
            return None

        fps_safe = fps if fps > 1e-6 else 30.0

        # Phase 9-A anchor B-4: visibility는 frame-level 절대값 평가 (window X).
        # pelvis/scale은 변동성 신호 (window baseline 필요), visibility는 붕괴 신호.
        # cfg.visibility_window_seconds 미사용 (Phase 8-E evaluate_tracking_stability
        # target_lost reason_code에서 활용 중, cfg 필드 제거 X).
        pelvis_window = max(1, int(fps_safe * cfg.pelvis_window_seconds))
        scale_window = max(1, int(fps_safe * cfg.scale_window_seconds))

        pelvis_residual = _compute_pelvis_residual_series(
            landmarks_series, pelvis_window
        )
        scale_variation = _compute_scale_series(landmarks_series, scale_window)
        visibility_per_frame = _compute_visibility_per_frame_local(landmarks_series)

        consecutive = 0
        for i in range(len(landmarks_series)):
            pelvis_spike = pelvis_residual[i] > cfg.pelvis_residual_spike
            scale_spike_violation = abs(scale_variation[i]) > cfg.scale_spike
            visibility_collapse = (
                visibility_per_frame[i] < cfg.target_lost_visibility_threshold
            )

            if pelvis_spike and scale_spike_violation and visibility_collapse:
                consecutive += 1
                if consecutive >= cfg.target_switch_consecutive_frames:
                    return ReasonCodeEntry(
                        reason_code="target_switch_detected",
                        severity=REASON_CODE_SEVERITY["target_switch_detected"],
                    )
            else:
                consecutive = 0

        return None
    except Exception:
        logger.exception(
            "evaluate_target_switch 예외 (swallow, target_switch_detected 반환)"
        )
        return ReasonCodeEntry(
            reason_code="target_switch_detected",
            severity=REASON_CODE_SEVERITY["target_switch_detected"],
        )

