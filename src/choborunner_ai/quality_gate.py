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

from choborunner_ai.config import VisibilityCheckConfig
from choborunner_ai.pose_extractor import PoseLandmarks

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
]
"""docs/2-3-5 §8-2 visibility/geometry 카테고리 reason code 6종.

§5-1 (Phase 4) — `lower_body_not_visible` / `foot_not_visible` /
`upper_body_not_visible` / `low_landmark_visibility`.

§5-2 / §5-3 (Phase 8-A) — `body_not_fully_visible` (전신 미포함, 머리/발끝
잘림 의심) / `foot_out_of_frame` (발 화면 하단 잘림 의심).

각 코드 강도(failed vs low_confidence)는 REASON_CODE_SEVERITY 참조.
사용자 메시지는 docs §8-2 표 (SoT). 응답 메시지 매핑 + 우선순위 1개 선택은
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
}
"""docs §8-2 visibility/geometry reason code 강도 매핑.

§5-1: `foot_not_visible`만 failed — 발 미가시 시 IC 검출 불가, 핵심 지표 산출
자체 불가. 나머지 3종은 low_confidence — 지표 산출은 가능하나 신뢰도 낮음.

§5-2 / §5-3 (Phase 8-A 추가): `body_not_fully_visible` / `foot_out_of_frame`
둘 다 failed (docs §8-2 표) — 전신/발 미포함은 분석 자체가 의미 없음.

사용 위치는 §6 status 분기 별도 Phase (Phase 8-F). 본 dict 정의는 reason code
사전과 함께 본 모듈에 위치 (SoT 정합).
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


# ============================================================
# 누적 visibility 평가 (docs §5-1 5번 유효 frame 비율)
# ============================================================


def evaluate_visibility_accumulation(
    results: list[FrameVisibilityResult],
    cfg: VisibilityCheckConfig,
) -> list[ReasonCode]:
    """누적 평가 — 유효 frame 비율 (docs/2-3-5 §5-1 5번).

    유효 frame 정의: `FrameVisibilityResult.is_valid=True` (4 카테고리 모두
    통과). docs §3-1의 "유효 frame" 정의(분석측 5 landmark 평균 임계 통과)와
    다름 — 본 모듈 §5-1 5번 한정 정의 (Phase 4 decision 7).

    모집단: 입력 list 전체. 빈 list → 빈 list 반환 (모듈 경계, decision 5:
    frame 부족은 docs/2-3-1 `too_short` trigger 책임, 본 모듈 책임 외).

    Args:
        results: list[FrameVisibilityResult] (Phase 4-A 출력 누적).
        cfg: VisibilityCheckConfig. `valid_frame_ratio_min` 사용 (default 0.6).

    Returns:
        list[ReasonCode]:
        - 유효 비율 ≥ `valid_frame_ratio_min` 또는 빈 입력 → `[]`
        - 유효 비율 < `valid_frame_ratio_min` → `["low_landmark_visibility"]`
        단일 코드라도 list 형식 유지 (호출자 일관성, 향후 §5-2~5-7 누적 함수
        확장 시 동일 시그니처).

    견고성 가드:
    - 빈 list → `[]` (zero division 가드, 모듈 경계).
    - 평가 예외 → `logger.exception` + `["low_landmark_visibility"]` failed-safe
      ("평가 못 함" 시그널, 메인 분석 흐름 보호).
    """
    try:
        if not results:
            return []
        valid_count = sum(1 for r in results if r.is_valid)
        ratio = valid_count / len(results)
        if ratio < cfg.valid_frame_ratio_min:
            return ["low_landmark_visibility"]
        return []
    except Exception:
        logger.exception(
            "evaluate_visibility_accumulation 예외 "
            "(swallow, low_landmark_visibility 반환)"
        )
        return ["low_landmark_visibility"]


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
) -> Optional[ReasonCode]:
    """§5-2 body 포함 누적 평가 — 통과 frame 비율 (docs/2-3-5 §5-2 frame 비율 60%).

    유효 frame 정의: `FrameGeometryResult.is_valid=True` (body_visibility AND
    body_coords 통과). 본 함수는 `evaluate_frame_body_inclusion` 출력 누적만
    입력으로 받음 — `evaluate_frame_foot_cutoff` 출력 섞으면 의미 깨짐.

    모집단: 입력 list 전체. 빈 list → None (frame 부족은 docs/2-3-1 `too_short`
    책임, 본 모듈 책임 외 — §5-1 패턴 일관).

    Args:
        results: list[FrameGeometryResult] (`evaluate_frame_body_inclusion` 누적).
        cfg: VisibilityCheckConfig. `body_inclusion_frame_ratio_min` (default 0.6).

    Returns:
        Optional[ReasonCode]:
        - 통과 비율 ≥ `body_inclusion_frame_ratio_min` 또는 빈 입력 → None
        - 통과 비율 < `body_inclusion_frame_ratio_min` → 'body_not_fully_visible'

        ⚠️ 시그니처 catch (Phase 8-A lock 5-6 α 정합):
        본 함수 반환은 `Optional[ReasonCode]` — §5-1 `evaluate_visibility_accumulation`
        의 `list[ReasonCode]`와 다름. §5-1 docstring은 "향후 §5-2~5-7 누적 함수 확장
        시 동일 시그니처" 의도였으나, Phase 8-A에서 단일 reason code 함수당 1개
        대응 명확화를 위해 Optional 채택. 향후 §6 status 분기 단계 (Phase 8-F)에서
        list 통일 검토.

    견고성 가드: 빈 list → None (zero division), 평가 예외 → failed-safe
    ('body_not_fully_visible' 반환).
    """
    try:
        if not results:
            return None
        valid_count = sum(1 for r in results if r.is_valid)
        ratio = valid_count / len(results)
        if ratio < cfg.body_inclusion_frame_ratio_min:
            return "body_not_fully_visible"
        return None
    except Exception:
        logger.exception(
            "evaluate_body_inclusion_accumulation 예외 "
            "(swallow, body_not_fully_visible 반환)"
        )
        return "body_not_fully_visible"


def evaluate_foot_cutoff_accumulation(
    results: list[FrameGeometryResult],
    cfg: VisibilityCheckConfig,
) -> Optional[ReasonCode]:
    """§5-3 발 잘림 누적 평가 — 통과 frame 비율 (docs/2-3-5 §5-3 frame 비율 60%).

    유효 frame 정의: `FrameGeometryResult.is_valid=True` (foot_cutoff 통과).
    본 함수는 `evaluate_frame_foot_cutoff` 출력 누적만 입력으로 받음.

    모집단: 입력 list 전체. 빈 list → None (§5-1 패턴 일관, frame 부족 책임 외).

    Args:
        results: list[FrameGeometryResult] (`evaluate_frame_foot_cutoff` 누적).
        cfg: VisibilityCheckConfig. `foot_cutoff_frame_ratio_min` (default 0.6).

    Returns:
        Optional[ReasonCode]:
        - 통과 비율 ≥ `foot_cutoff_frame_ratio_min` 또는 빈 입력 → None
        - 통과 비율 < `foot_cutoff_frame_ratio_min` → 'foot_out_of_frame'

    견고성 가드: 빈 list → None, 평가 예외 → failed-safe ('foot_out_of_frame').
    """
    try:
        if not results:
            return None
        valid_count = sum(1 for r in results if r.is_valid)
        ratio = valid_count / len(results)
        if ratio < cfg.foot_cutoff_frame_ratio_min:
            return "foot_out_of_frame"
        return None
    except Exception:
        logger.exception(
            "evaluate_foot_cutoff_accumulation 예외 "
            "(swallow, foot_out_of_frame 반환)"
        )
        return "foot_out_of_frame"
