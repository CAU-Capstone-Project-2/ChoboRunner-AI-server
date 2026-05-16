"""docs/2-3-4 §6 Initial Knee Flexion 산출 — 정식판 (Phase 5-C-1).

본 모듈은 docs/2-3-4 §6 (Initial Knee Flexion 산출 로직) 단일 정답 구현.
Phase 5-A trunk_lean 패턴 재사용 + analysis_side 입력 추가 (decision ii).

Phase 5-C scope (Day 5 결정 6건, 압축 모드):
- (i)   KneeFlexionConfig.visibility_min 신규 필드 추가 (default 0.6)
- (ii)  API: knee_flexion_deg(pl, analysis_side, cfg) — analysis_side 입력
- (iii) KneeFlexionResult dataclass (TrunkLeanResult 패턴 그대로)
- (iv)  분류 경계 docs §6-6 strict (`<` / `<` / `>=`)
- (v)   2 sub-phase (5-C-1 코드 + 5-C-2 pytest)
- (vi)  jaemin baseline metric (5-C-2)

좌표계 (docs §3-2): x 좌→우, y 화면 아래 + (MediaPipe convention).

계산 (docs §6-4):
    v1 = hip - knee  (분석측)
    v2 = ankle - knee (분석측)
    θ_inner = arccos(dot(v1, v2) / (|v1| × |v2|))
    knee_flexion = 180° − θ_inner

- 값이 작을수록 직선에 가까움 (덜 굽혀짐), 클수록 더 굽혀짐
- 2D 투영각 — 3D mocap 절대값과 차이 (docs §6-5, 파일럿 보정 필요)

visibility (docs §6-2):
- hip/knee/ankle 분석측 3점 visibility 가드 (cfg.visibility_min, default 0.6)
- 미달 시 NaN

분류 (docs §6-6, decision iv strict):
- below_typical: knee_flexion < cfg.below_typical_deg (default 15°)
- typical:       cfg.below_typical_deg ≤ kf < cfg.above_typical_deg (15~25°)
- above_typical: knee_flexion ≥ cfg.above_typical_deg (default 25°)

⚠️ catch (trunk_lean과 차이, 압축 모드라도 박을 자산):
- 분류 경계: trunk는 `<` / `<=` / `>` (양쪽 inclusive),
  knee는 `<` / `<` / `>=` (한쪽 exclusive)
- 사용 landmark: trunk 양측 평균 (shoulder + hip),
  knee 분석측만 (hip + knee + ankle)
- arccos vector 가드: trunk는 1 vector (shoulder-hip),
  knee는 2 vector (hip-knee + ankle-knee). 둘 다 가드.
- arccos 입력 [-1, 1] clamp (부동소수점 오차 가드)

IC ± window 평균 (docs §6-3 + §7-2 정합):
- IC ± cfg.ic_window_offset frame 평균 (default 2 → 5 frame window)
- window 유효 비율 >= 0.5 시 평균, 미달 시 stride 제외

참고문헌:
- Teng, H. L., & Powers, C. M. (2014). Sagittal plane trunk posture influences
  patellofemoral joint stress during running. Medicine & Science in Sports &
  Exercise, 46(9), 1739–1747.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Optional

from choborunner_ai.config import KneeFlexionConfig
from choborunner_ai.pose_extractor import PoseLandmarks

logger = logging.getLogger(__name__)


# ============================================================
# Classification Literal (docs §6-6)
# ============================================================


KneeFlexionClassification = Literal[
    "below_typical",
    "typical",
    "above_typical",
]
"""docs/2-3-4 §6-6 3 분류 (Day 5 lock, decision iv strict).

- 'below_typical': kf < cfg.below_typical_deg (default 15°)
- 'typical':       cfg.below_typical_deg ≤ kf < cfg.above_typical_deg (15~25°)
- 'above_typical': kf ≥ cfg.above_typical_deg (default 25°)
"""


# ============================================================
# KneeFlexionResult dataclass (IC ± window 평균 1 stride 결과)
# ============================================================


@dataclass
class KneeFlexionResult:
    """IC ± window 평균 1 stride knee flexion 결과 (docs §6-3, §6-7).

    TrunkLeanResult 패턴 그대로 (decision iii).

    Attributes:
        deg: IC ± window 평균 knee flexion 각도 (°). window 유효 frame 비율
            미달 시 NaN.
        classification: 3 분류. deg가 NaN이면 None.
        ic_frame_index: 본 stride의 IC frame index.
        window_valid_count: window 내 유효 frame 수.
        window_total_count: window 후보 frame 수.
        is_valid: window_valid_count / window_total_count >= 0.5.
    """

    deg: float
    classification: Optional[KneeFlexionClassification]
    ic_frame_index: int
    window_valid_count: int
    window_total_count: int
    is_valid: bool


# ============================================================
# 단일 frame knee flexion (docs §6-2, §6-4)
# ============================================================


def knee_flexion_deg(
    pl: PoseLandmarks,
    analysis_side: Literal["left", "right"],
    cfg: KneeFlexionConfig,
) -> float:
    """단일 frame knee flexion 각도 (°).

    docs §6-2 visibility 가드: hip/knee/ankle 분석측 3점 visibility 모두
    cfg.visibility_min (default 0.6) 통과 시 계산, 미달 시 NaN.

    docs §6-4 계산:
        v1 = hip - knee  (분석측)
        v2 = ankle - knee (분석측)
        θ_inner = arccos(dot(v1, v2) / (|v1| * |v2|))
        knee_flexion = 180° − θ_inner

    arccos 입력 [-1, 1] clamp (부동소수점 오차 가드).
    |v1| 또는 |v2| ~0 시 NaN (catch: trunk는 1 vector, knee는 2 vector 가드).

    Args:
        pl: PoseLandmarks (Phase 3, 6 LandmarkPair).
        analysis_side: 'left' or 'right' (docs §6-2 분석측만).
        cfg: KneeFlexionConfig.

    Returns:
        Knee flexion 각도 (°). visibility 미달 또는 vector 길이 ~0 시 NaN.

    Raises:
        ValueError: analysis_side가 'left'/'right' 아닐 시.
    """
    if analysis_side not in ("left", "right"):
        raise ValueError(
            f"analysis_side는 'left' 또는 'right'만 허용, got {analysis_side!r}"
        )

    hip = pl.hip.left if analysis_side == "left" else pl.hip.right
    knee = pl.knee.left if analysis_side == "left" else pl.knee.right
    ankle = pl.ankle.left if analysis_side == "left" else pl.ankle.right

    # visibility 가드 (docs §6-2)
    visibilities = (hip.visibility, knee.visibility, ankle.visibility)
    if any(not math.isfinite(v) or v < cfg.visibility_min for v in visibilities):
        return float("nan")

    # v1 = hip - knee, v2 = ankle - knee
    v1x = hip.x - knee.x
    v1y = hip.y - knee.y
    v2x = ankle.x - knee.x
    v2y = ankle.y - knee.y

    # |v1| 또는 |v2| ~0 가드 (catch: 2 vector 둘 다, trunk_lean과 차이)
    norm_v1 = math.hypot(v1x, v1y)
    norm_v2 = math.hypot(v2x, v2y)
    if norm_v1 < 1e-9 or norm_v2 < 1e-9:
        return float("nan")

    # θ_inner = arccos(dot / (|v1| * |v2|))
    dot = v1x * v2x + v1y * v2y
    cos_theta = dot / (norm_v1 * norm_v2)
    # arccos 입력 [-1, 1] clamp (부동소수점 오차 가드)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_inner_rad = math.acos(cos_theta)
    theta_inner_deg = math.degrees(theta_inner_rad)

    return 180.0 - theta_inner_deg


# ============================================================
# 시계열
# ============================================================


def compute_series(
    landmarks_series: list[Optional[PoseLandmarks]],
    analysis_side: Literal["left", "right"],
    cfg: KneeFlexionConfig,
) -> list[float]:
    """프레임 시퀀스 knee flexion 시계열.

    Args:
        landmarks_series: list[PoseLandmarks | None]. None 또는 visibility 미달
            frame은 NaN.
        analysis_side: 'left' or 'right'.
        cfg: KneeFlexionConfig.

    Returns:
        frame별 knee flexion 각도 list (same length).
    """
    out: list[float] = []
    for pl in landmarks_series:
        if pl is None:
            out.append(float("nan"))
        else:
            out.append(knee_flexion_deg(pl, analysis_side, cfg))
    return out


# ============================================================
# 분류 (docs §6-6, decision iv strict)
# ============================================================


def classify(
    deg: float, cfg: KneeFlexionConfig
) -> Optional[KneeFlexionClassification]:
    """docs §6-6 3 분류 (decision iv strict).

    경계:
    - below_typical: kf < below_typical_deg (15°)
    - typical:       below_typical_deg ≤ kf < above_typical_deg (15~25°)
    - above_typical: kf ≥ above_typical_deg (25°)

    ⚠️ trunk_lean.classify와 경계 차이 (trunk: `<` / `<=` / `>`).

    Args:
        deg: knee flexion 각도 (°).
        cfg: KneeFlexionConfig.

    Returns:
        'below_typical' | 'typical' | 'above_typical' | None (NaN 시).
    """
    if not math.isfinite(deg):
        return None
    if deg < cfg.below_typical_deg:
        return "below_typical"
    if deg < cfg.above_typical_deg:
        return "typical"
    return "above_typical"


# ============================================================
# IC ± window 평균 (docs §6-3 + §7-2 정합)
# ============================================================


def compute_at_ic(
    landmarks_series: list[Optional[PoseLandmarks]],
    ic_indices: list[int],
    analysis_side: Literal["left", "right"],
    cfg: KneeFlexionConfig,
) -> list[KneeFlexionResult]:
    """IC ± window 평균 knee flexion (docs §6-3 + §7-2 정합).

    trunk_lean.compute_at_ic 패턴 재사용 + analysis_side 인자.

    각 IC frame index 주변 ± cfg.ic_window_offset frame 평균.
    window 내 유효 (visibility 통과 + finite) frame 비율 50% 미만 시 본 stride
    제외 (deg=NaN, is_valid=False).

    경계 처리: IC가 영상 시작·종료에 가까워 window 일부가 series 범위 밖이면
    축소된 모집단 기준 50% 적용.

    Args:
        landmarks_series: list[PoseLandmarks | None] (전체 영상 시퀀스).
        ic_indices: IC frame index list (호출자 입력, ic_detector 결과 활용).
        analysis_side: 'left' or 'right'.
        cfg: KneeFlexionConfig.

    Returns:
        list[KneeFlexionResult] (len == len(ic_indices)).
    """
    results: list[KneeFlexionResult] = []
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
            v = knee_flexion_deg(pl, analysis_side, cfg)
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
            KneeFlexionResult(
                deg=mean_deg,
                classification=cls,
                ic_frame_index=ic,
                window_valid_count=window_valid,
                window_total_count=window_total,
                is_valid=is_valid,
            )
        )

    return results
