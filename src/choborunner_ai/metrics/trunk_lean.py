"""docs/2-3-4 §7 Trunk Lean 산출 — 정식판 (Phase 5-A-1).

본 모듈은 docs/2-3-4 §7 (Trunk Lean 산출 로직) 단일 정답 구현.

scope (Phase 5-A 결정 6건, Day 5 잠금):
- (i) trunk_lean.py 본격 구현 (rename X, metrics 폴더 명명 일관)
- (ii) PoseLandmarks 단독 인터페이스 (Phase 3 정식)
- (iii) 모든 함수 포함, IC ± window 평균은 ic_indices 외부 입력
       (IC detector 미완성에서도 정식판 완성 가능)
- (iv) dataclass TrunkLeanResult
- (v) 5-A-1 (코드) / 5-A-2 (pytest) / 5-A-3 (Vertical Slice 폐기) 분할
- (vi) demo_trunk.py 정식판 호출 변경 (Phase 5-A-3)

좌표계 (docs §3-2):
- x 좌→우, y 화면 아래 + (MediaPipe convention)

부호 (docs §7-4):
- v_trunk = shoulder_center - hip_center
- theta = atan2(v.x, -v.y) -> 전방 = +, 후방 = -
- demo2 (legacy/demo_02) 절대값화 폐기 (CLAUDE.md §5-6 부호 보존)

visibility (docs §7-2):
- shoulder L/R + hip L/R 4점 visibility 모두 cfg.visibility_min 통과 시 계산
- 미달 시 NaN 반환

IC ± window 평균 (docs §7-3 + §7-2):
- IC ± cfg.ic_window_offset frame 평균 (default 2 → 5 frame window)
- window 내 유효 frame 비율 >= 0.5 시 평균 계산
  미달 시 stride 제외 (TrunkLeanResult.is_valid=False, deg=NaN) — docs §7-2 인용

분류 (docs §7-6):
- near_vertical: theta < cfg.near_vertical_below_deg (default 5°)
- forward_lean: near_vertical_below_deg <= theta <= cfg.forward_above_deg (5~10°)
- above_typical: theta > forward_above_deg (>10°)

참고문헌:
- Teng, H. L., & Powers, C. M. (2014). Sagittal plane trunk posture influences
  patellofemoral joint stress during running. Medicine & Science in Sports &
  Exercise, 46(9), 1739-1747.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Optional

from choborunner_ai.config import TrunkLeanConfig
from choborunner_ai.pose_extractor import PoseLandmarks

logger = logging.getLogger(__name__)


# ============================================================
# Classification Literal (docs §7-6)
# ============================================================


TrunkLeanClassification = Literal[
    "near_vertical",
    "forward_lean",
    "above_typical",
]
"""docs/2-3-4 §7-6 3 분류 (Day 5 lock).

- 'near_vertical': θ < cfg.near_vertical_below_deg (default 5°)
- 'forward_lean': near_vertical_below_deg <= θ <= forward_above_deg (5~10°)
- 'above_typical': θ > forward_above_deg (>10°)

응답 메시지(2-3-7) 매핑은 별도 Phase. 본 Literal은 단일 정답.
"""


# ============================================================
# TrunkLeanResult dataclass (IC ± window 평균 1 stride 결과)
# ============================================================


@dataclass
class TrunkLeanResult:
    """IC ± window 평균 1 stride trunk lean 결과 (docs §7-3, §7-7).

    Attributes:
        deg: IC ± window 평균 trunk lean 각도 (°). window 유효 frame 비율 미달
            시 NaN. 부호 보존: 전방=+, 후방=- (CLAUDE.md §5-6).
        classification: 3 분류. deg가 NaN이면 None.
        ic_frame_index: 본 stride의 IC frame index (호출자 입력 보존).
        window_valid_count: window 내 유효 (visibility 통과 + finite) frame 수.
        window_total_count: window 후보 frame 수 (IC ± offset, 영상 경계 reach
            제외 후 실제 가용 frame 수).
        is_valid: window_valid_count / window_total_count >= 0.5
            (docs §7-2 인용). False 시 본 stride는 stride 통계 산출에서 제외.
    """

    deg: float
    classification: Optional[TrunkLeanClassification]
    ic_frame_index: int
    window_valid_count: int
    window_total_count: int
    is_valid: bool


# ============================================================
# 단일 frame trunk lean (docs §7-2, §7-4)
# ============================================================


def trunk_lean_deg(pl: PoseLandmarks, cfg: TrunkLeanConfig) -> float:
    """단일 frame trunk lean 각도 (°) — 부호 보존 (전방=+, 후방=-).

    docs §7-2 visibility 가드: shoulder L/R + hip L/R 4점 visibility 모두
    cfg.visibility_min (default 0.6) 통과 시 계산, 미달 시 NaN.

    docs §7-4 계산:
        shoulder_center = (shoulder.left + shoulder.right) / 2
        hip_center      = (hip.left + hip.right) / 2
        v_trunk         = shoulder_center - hip_center
        theta           = atan2(v.x, -v.y)
    MediaPipe y축 화면 아래 + 정합 (-v.y로 부호 반전 → 전방 기울기 +).

    Args:
        pl: PoseLandmarks (Phase 3 6 LandmarkPair).
        cfg: TrunkLeanConfig.

    Returns:
        Trunk lean 각도 (°). visibility 미달 또는 vector 길이 ~0 시 NaN.
    """
    visibilities = (
        pl.shoulder.left.visibility,
        pl.shoulder.right.visibility,
        pl.hip.left.visibility,
        pl.hip.right.visibility,
    )
    if any(not math.isfinite(v) or v < cfg.visibility_min for v in visibilities):
        return float("nan")

    sx = (pl.shoulder.left.x + pl.shoulder.right.x) * 0.5
    sy = (pl.shoulder.left.y + pl.shoulder.right.y) * 0.5
    hx = (pl.hip.left.x + pl.hip.right.x) * 0.5
    hy = (pl.hip.left.y + pl.hip.right.y) * 0.5

    vx = sx - hx
    vy = sy - hy

    if math.hypot(vx, vy) < 1e-9:  # 순수 수학 가드 (CLAUDE.md §8 인라인 허용)
        return float("nan")

    theta_rad = math.atan2(vx, -vy)
    return math.degrees(theta_rad)


# ============================================================
# 시계열 (frame 단위 시퀀스)
# ============================================================


def compute_series(
    landmarks_series: list[Optional[PoseLandmarks]],
    cfg: TrunkLeanConfig,
) -> list[float]:
    """프레임 시퀀스 trunk lean 시계열 (Vertical Slice 호환 인터페이스).

    Args:
        landmarks_series: list[PoseLandmarks | None]. None 또는 visibility 미달
            frame은 NaN.
        cfg: TrunkLeanConfig.

    Returns:
        frame별 trunk lean 각도 list (same length as input).
    """
    out: list[float] = []
    for pl in landmarks_series:
        if pl is None:
            out.append(float("nan"))
        else:
            out.append(trunk_lean_deg(pl, cfg))
    return out


# ============================================================
# 분류 (docs §7-6)
# ============================================================


def classify(deg: float, cfg: TrunkLeanConfig) -> Optional[TrunkLeanClassification]:
    """docs §7-6 3 분류. NaN -> None.

    Args:
        deg: trunk lean 각도 (°).
        cfg: TrunkLeanConfig (near_vertical_below_deg / forward_above_deg).

    Returns:
        'near_vertical' | 'forward_lean' | 'above_typical' | None (NaN 시).

    경계 처리: docs §7-6 표 정합.
    - θ < near_vertical_below_deg          → near_vertical
    - near_vertical_below_deg <= θ <= forward_above_deg → forward_lean
    - θ > forward_above_deg                → above_typical
    """
    if not math.isfinite(deg):
        return None
    if deg < cfg.near_vertical_below_deg:
        return "near_vertical"
    if deg <= cfg.forward_above_deg:
        return "forward_lean"
    return "above_typical"


# ============================================================
# IC ± window 평균 (docs §7-3, §7-2 — Day 5 decision iii)
# ============================================================


def compute_at_ic(
    landmarks_series: list[Optional[PoseLandmarks]],
    ic_indices: list[int],
    cfg: TrunkLeanConfig,
) -> list[TrunkLeanResult]:
    """IC ± window 평균 trunk lean (docs §7-3 + §7-2).

    각 IC frame index 주변 ± cfg.ic_window_offset frame 평균.
    window 내 유효 (visibility 통과 + finite) frame 비율 50% 미만 시 본 stride
    제외 (deg=NaN, is_valid=False, classification=None) — docs §7-2 정합.

    경계 처리: IC가 영상 시작·종료에 가까워 window 일부가 series 범위 밖이면
    범위 안 frame만 사용 (window_total_count 축소). 50% 비율은 축소된 모집단
    기준 — 영상 경계 frame이 부당하게 제외되는 것 회피.

    Args:
        landmarks_series: list[PoseLandmarks | None] (전체 영상 시퀀스).
        ic_indices: IC frame index list. 호출자 입력 (Day 5 decision iii — IC
            detector 미완성에서도 정식판 완성 가능).
        cfg: TrunkLeanConfig.

    Returns:
        list[TrunkLeanResult] (len == len(ic_indices)). 입력 ic_indices 순서 보존.
    """
    results: list[TrunkLeanResult] = []
    offset = cfg.ic_window_offset
    total_len = len(landmarks_series)

    for ic in ic_indices:
        start = max(0, ic - offset)
        end = min(total_len, ic + offset + 1)
        window = landmarks_series[start:end]
        window_total = len(window)

        finite_vals: list[float] = []
        for pl in window:
            if pl is None:
                continue
            v = trunk_lean_deg(pl, cfg)
            if math.isfinite(v):
                finite_vals.append(v)

        window_valid = len(finite_vals)
        is_valid = window_total > 0 and (window_valid / window_total) >= 0.5

        if is_valid and finite_vals:
            mean_deg = sum(finite_vals) / len(finite_vals)
            cls = classify(mean_deg, cfg)
        else:
            mean_deg = float("nan")
            cls = None

        results.append(
            TrunkLeanResult(
                deg=mean_deg,
                classification=cls,
                ic_frame_index=ic,
                window_valid_count=window_valid,
                window_total_count=window_total,
                is_valid=is_valid,
            )
        )

    return results
