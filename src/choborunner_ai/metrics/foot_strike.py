"""docs/2-3-4 §5 Foot Strike Pattern 산출 — 정식판 (Phase 5-D-1).

본 모듈은 docs/2-3-4 §5 (Foot Strike Pattern 산출 로직) 단일 정답 구현.
Phase 5-C knee_flexion 패턴 재사용 + **direction 인자 추가** (docs §3-4 catch).

Phase 5-D scope (Day 5 결정 7건, 압축 모드):
- (i)   direction: Literal["left_to_right","right_to_left"] 함수 인자
- (ii)  visibility 가드: heel + foot_index 2점 (ankle 제외, docs §5-7)
- (iii) FootStrikeConfig.visibility_min 신규 필드 (default 0.6)
- (iv)  분류 경계 docs §5-5 strict (RFS: θ ≥ +5° / MFS: -5° < θ < +5° / FFS: θ ≤ -5°)
- (v)   Uncertain → classification=None (trunk/knee 패턴 일관)
- (vi)  히스테리시스 ±3° + 누적 최빈값 scope 외 (Phase 7 또는 별도 anchor)
- (vii) 2 sub-phase (5-D-1 코드 + 5-D-2 pytest)

좌표계 (docs §3-2): x 좌→우, y 화면 아래 + (MediaPipe convention).

계산 (docs §5-4):
    direction_sign = +1 if 'left_to_right' else -1
    vx_signed = direction_sign * (toe.x - heel.x)
    vy        = toe.y - heel.y
    θ_foot    = atan2(-vy, vx_signed)  # y 부호 반전 (화면 위쪽이 양수)

부호 (docs §5-4):
- 양수 (+): 발끝 들림 (dorsiflexion 성향, RFS 경향)
- 음수 (-): 발끝 내려감 (plantarflexion 성향, FFS 경향)

⚠️ direction 인자 catch (docs §3-4):
- "진행 방향 부호는 Foot Strike Pattern 분류와 직결되므로(잘못 정렬되면
  RFS / FFS 분류가 뒤집힘) 안정적인 결정이 매우 중요"
- MVP 정책: 사용자 선택 (촬영 시작 전 카메라 위치 좌/우 선택)
- analysis_side + direction 2개 인자 — trunk/knee와 인터페이스 차이

분류 (docs §5-5, decision iv strict):
- rfs: θ_foot ≥ cfg.rfs_above_deg (default +5°)
- mfs: cfg.ffs_below_deg < θ_foot < cfg.rfs_above_deg (-5 < θ < +5)
- ffs: θ_foot ≤ cfg.ffs_below_deg (default -5°)

⚠️ catch (trunk/knee와 차이, 압축 모드라도 박힐 자산):
- 분류 경계 패턴 3가지 다름:
  · trunk §7-6: `<` / `<=` / `>` (양쪽 inclusive)
  · knee §6-6: `<` / `<` / `>=` (한쪽 exclusive)
  · foot §5-5: `<=` / `< & <` / `>=` (양극 inclusive, 중간 strict)
- visibility 가드 2점 (heel + foot_index 분석측만, ankle 제외)
  · trunk: 4점 (shoulder L/R + hip L/R)
  · knee: 3점 (hip + knee + ankle 분석측)
  · foot: 2점 (heel + foot_index 분석측, §5-7)
- vector 길이 ~0 가드: 1 vector (toe - heel)
- ic_window_offset: 1 (3 frame window, knee 5 frame과 다름, docs §5-3 "1~2 frame")
- 단일 IC 외부 노출 금지 (docs §5-6): 본 compute_at_ic 결과는 호출자가 누적
  최빈값으로 변환 (별도 Phase). 본 모듈은 compute_at_ic까지만.
- 히스테리시스 ±3° (cfg.hysteresis_deg): 본 Phase 5-D scope 외, 별도.

Uncertain 정책 (docs §5-7):
- heel/foot_index visibility 미달 → NaN → classification=None
- vector 길이 ~0 → NaN → classification=None
- IC ± window 유효 비율 < 0.5 → is_valid=False, deg=NaN, classification=None
- IC 신뢰도 'low' (docs §4-3) → 호출자가 ic_indices 입력 시 제외 (5-C-2 패턴)
- 호출자가 None을 'uncertain' 라벨로 변환 (응답 메시지 단계, 별도 Phase)

참고문헌:
- Knorz, S., et al. (2017). Three-dimensional biomechanical analysis of rearfoot
  and forefoot running. Journal of Visualized Experiments, (122), e54818.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Optional

from choborunner_ai.config import FootStrikeConfig
from choborunner_ai.pose_extractor import PoseLandmarks

logger = logging.getLogger(__name__)


# ============================================================
# Classification Literal (docs §5-5)
# ============================================================


FootStrikeClassification = Literal["rfs", "mfs", "ffs"]
"""docs/2-3-4 §5-5 3 분류 (Day 5 lock, decision iv strict).

- 'rfs' (Rearfoot Strike): θ_foot ≥ cfg.rfs_above_deg (+5°)
- 'mfs' (Midfoot Strike): cfg.ffs_below_deg < θ_foot < cfg.rfs_above_deg (-5 < θ < +5)
- 'ffs' (Forefoot Strike): θ_foot ≤ cfg.ffs_below_deg (-5°)

Uncertain은 classification=None (docs §5-7 정합, decision v).
호출자가 None을 'uncertain' 라벨로 변환 (응답 메시지 단계).
"""


# ============================================================
# FootStrikeResult dataclass
# ============================================================


@dataclass
class FootStrikeResult:
    """IC ± window 평균 1 stride foot strike 결과 (docs §5-3, §5-8).

    KneeFlexionResult 패턴 재사용 (decision iii). direction은 보존하지 않음
    (호출자 입력, stride별 변동 가정 X — MVP 사용자 선택은 영상 시작 시 고정).

    Attributes:
        deg: IC ± window 평균 foot angle (°). 부호: + dorsiflexion / - plantarflexion.
            window 유효 비율 미달 시 NaN.
        classification: 3 분류 또는 None (Uncertain, docs §5-7).
        ic_frame_index: 본 stride IC frame index.
        window_valid_count: window 내 유효 frame 수.
        window_total_count: window 후보 frame 수.
        is_valid: window_valid_count / window_total_count >= 0.5.
    """

    deg: float
    classification: Optional[FootStrikeClassification]
    ic_frame_index: int
    window_valid_count: int
    window_total_count: int
    is_valid: bool


# ============================================================
# 단일 frame foot strike angle (docs §5-4)
# ============================================================


def foot_strike_deg(
    pl: PoseLandmarks,
    analysis_side: Literal["left", "right"],
    direction: Literal["left_to_right", "right_to_left"],
    cfg: FootStrikeConfig,
) -> float:
    """단일 frame foot angle (°). 부호: + dorsiflexion / - plantarflexion.

    docs §5-7 visibility 가드: heel + foot_index 분석측 2점 visibility 모두
    cfg.visibility_min (default 0.6) 통과 시 계산, 미달 시 NaN.

    docs §5-4 계산 + direction 부호 처리 (docs §3-4):
        direction_sign = +1 if 'left_to_right' else -1
        vx_signed = direction_sign * (toe.x - heel.x)
        vy        = toe.y - heel.y
        θ_foot    = atan2(-vy, vx_signed)  # y 부호 반전 (화면 위쪽 +)

    Args:
        pl: PoseLandmarks (Phase 3, 6 LandmarkPair).
        analysis_side: 'left' or 'right' (docs §5-2 분석측만).
        direction: 'left_to_right' or 'right_to_left' (docs §3-4 사용자 선택).
        cfg: FootStrikeConfig.

    Returns:
        Foot angle (°). visibility 미달 또는 vector 길이 ~0 시 NaN.

    Raises:
        ValueError: analysis_side 또는 direction이 잘못된 값일 시.
    """
    if analysis_side not in ("left", "right"):
        raise ValueError(
            f"analysis_side는 'left' 또는 'right'만 허용, got {analysis_side!r}"
        )
    if direction not in ("left_to_right", "right_to_left"):
        raise ValueError(
            f"direction은 'left_to_right' 또는 'right_to_left'만 허용, got {direction!r}"
        )

    heel = pl.heel.left if analysis_side == "left" else pl.heel.right
    foot = (
        pl.foot_index.left if analysis_side == "left" else pl.foot_index.right
    )

    # visibility 가드 (docs §5-7, heel + foot_index 2점, ankle 제외)
    if any(
        not math.isfinite(v) or v < cfg.visibility_min
        for v in (heel.visibility, foot.visibility)
    ):
        return float("nan")

    # direction 부호 처리 (docs §3-4)
    direction_sign = 1.0 if direction == "left_to_right" else -1.0
    vx_signed = direction_sign * (foot.x - heel.x)
    vy = foot.y - heel.y

    # vector 길이 ~0 가드 (CLAUDE.md §8 인라인)
    if math.hypot(vx_signed, vy) < 1e-9:
        return float("nan")

    # θ_foot = atan2(-vy, vx_signed) — y 부호 반전 (docs §5-4)
    theta_rad = math.atan2(-vy, vx_signed)
    return math.degrees(theta_rad)


# ============================================================
# 시계열
# ============================================================


def compute_series(
    landmarks_series: list[Optional[PoseLandmarks]],
    analysis_side: Literal["left", "right"],
    direction: Literal["left_to_right", "right_to_left"],
    cfg: FootStrikeConfig,
) -> list[float]:
    """프레임 시퀀스 foot angle 시계열.

    Args:
        landmarks_series: list[PoseLandmarks | None]. None 또는 visibility 미달
            frame은 NaN.
        analysis_side: 'left' or 'right'.
        direction: 'left_to_right' or 'right_to_left'.
        cfg: FootStrikeConfig.

    Returns:
        frame별 foot angle (°) list (same length).
    """
    out: list[float] = []
    for pl in landmarks_series:
        if pl is None:
            out.append(float("nan"))
        else:
            out.append(foot_strike_deg(pl, analysis_side, direction, cfg))
    return out


# ============================================================
# 분류 (docs §5-5, decision iv strict)
# ============================================================


def classify(
    deg: float, cfg: FootStrikeConfig
) -> Optional[FootStrikeClassification]:
    """docs §5-5 3 분류 (decision iv strict).

    경계:
    - rfs: θ ≥ cfg.rfs_above_deg (+5°, inclusive)
    - mfs: cfg.ffs_below_deg < θ < cfg.rfs_above_deg (-5 < θ < +5, 양쪽 strict)
    - ffs: θ ≤ cfg.ffs_below_deg (-5°, inclusive)

    ⚠️ trunk/knee와 경계 패턴 차이:
    - trunk §7-6: `<` / `<=` / `>`
    - knee §6-6: `<` / `<` / `>=`
    - foot §5-5: `<=` / `< & <` / `>=` (양극 inclusive, 중간 strict)

    NaN → None (Uncertain, docs §5-7).
    """
    if not math.isfinite(deg):
        return None
    if deg >= cfg.rfs_above_deg:
        return "rfs"
    if deg <= cfg.ffs_below_deg:
        return "ffs"
    return "mfs"


# ============================================================
# IC ± window 평균 (docs §5-3 + §7-2 정합)
# ============================================================


def compute_at_ic(
    landmarks_series: list[Optional[PoseLandmarks]],
    ic_indices: list[int],
    analysis_side: Literal["left", "right"],
    direction: Literal["left_to_right", "right_to_left"],
    cfg: FootStrikeConfig,
) -> list[FootStrikeResult]:
    """IC ± window 평균 foot angle (docs §5-3 + §7-2 정합).

    knee_flexion.compute_at_ic 패턴 + direction 인자.

    각 IC frame index 주변 ± cfg.ic_window_offset frame 평균.
    ic_window_offset default 1 → 3 frame window (knee 5 frame과 다름, docs §5-3
    "1~2 frame" 정합).

    window 내 유효 frame 비율 50% 미만 시 본 stride 제외 (deg=NaN, is_valid=False).

    경계 처리: IC가 영상 시작·종료에 가까워 window 일부가 series 범위 밖이면
    축소된 모집단 기준 50% 적용.

    ⚠️ docs §5-6: "단일 IC 분류 사용 금지" — 본 compute_at_ic 결과는 호출자가
    누적 최빈값으로 변환 (별도 Phase). 본 모듈은 compute_at_ic까지만.

    Args:
        landmarks_series: list[PoseLandmarks | None] (전체 영상 시퀀스).
        ic_indices: IC frame index list (호출자 입력, ic_detector + low 제외).
        analysis_side: 'left' or 'right'.
        direction: 'left_to_right' or 'right_to_left'.
        cfg: FootStrikeConfig.

    Returns:
        list[FootStrikeResult] (len == len(ic_indices)).
    """
    results: list[FootStrikeResult] = []
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
            v = foot_strike_deg(pl, analysis_side, direction, cfg)
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
            FootStrikeResult(
                deg=mean_deg,
                classification=cls,
                ic_frame_index=ic,
                window_valid_count=window_valid,
                window_total_count=window_total,
                is_valid=is_valid,
            )
        )

    return results
