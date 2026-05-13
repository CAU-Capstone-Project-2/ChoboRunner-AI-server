"""Trunk Lean (상체 전경사) 산출 — Vertical Slice.

Trunk Lean은 상체가 수직축 대비 얼마나 앞으로 기울어져 있는지를 나타내는 지표.
러닝 효율 및 하지 부상 위험과의 연관이 보고됨 (Teng & Powers, 2014).

-------------------------------------------------------------------------------
좌표계·계산식
-------------------------------------------------------------------------------
좌표계: MediaPipe 정규화 영상 좌표 (x ∈ [0,1], y ∈ [0,1], y는 화면 아래쪽 +).

계산 (docs/2-3-4 §7-4):

    shoulder_center = (LEFT_SHOULDER + RIGHT_SHOULDER) / 2
    hip_center      = (LEFT_HIP + RIGHT_HIP) / 2
    v_trunk         = shoulder_center − hip_center        # (x, y) — y 아래 +
    θ_trunk         = atan2(v_trunk.x, -v_trunk.y)         # MediaPipe y축 부호 반전

부호 컨벤션: **전방 기울기 = 양수(+), 후방 = 음수(−)**.

-------------------------------------------------------------------------------
demo2 절대값화 폐기 정책 적용 (CLAUDE.md §5-6)
-------------------------------------------------------------------------------
legacy/demo_02/metrics.py:61-85의 `trunk_lean_deg_at_frame`은 `np.clip(c, 0.0, 1.0)`
로 cos값을 [0, 1]에 가두고 `arccos`을 취해 결과를 항상 [0, 90]° 절대값으로 만들었음.

본 구현에서는 그 절대값화를 폐기하고 atan2 기반으로 **부호를 보존**한다.

근거: 디버깅·재교정 시 전/후방 구분 정보가 사라지면 잘못된 자세 패턴 식별 불가.
docs/2-3-4 §7-4 계산식과 정합.

-------------------------------------------------------------------------------
⚠️ 본 모듈 미포함 — Knee Flexion / Foot Strike Pattern
-------------------------------------------------------------------------------
Vertical Slice는 Trunk Lean만 이식하며, 나머지 두 지표는 **의도적으로 제외**.

근거: demo2 회귀 분석(2026-05-10)에서 jaemin 영상의 Knee Flexion 42.2°가
4분류 v1 schema의 `excessive_tendency`로 오분류된 사례. 원인 4가지 규명:

  1. IC 검출 알고리즘이 true Zeni 2008이 아닌 단순 휴리스틱 (heel-pelvis x 피크)
  2. Fellin 2010 lookahead 보정 누락
  3. 4개 knee 임계값이 학술 인용 없이 박혀 있음 (3D mocap 출처를 2D에 직접 적용)
  4. Foot Strike Pattern은 IC 시점 의존이라 IC 결함이 그대로 전파

본 구현에서는 IC 알고리즘 재설계(true Zeni 2008 + Fellin 2010 lookahead,
docs/2-3-4 §4) 후 임계값 재교정을 별도 마일스톤으로 분리 (CLAUDE.md §11 참조).

Trunk Lean은 IC 시점 의존성이 0 (프레임별 독립 계산)이라 회귀 분석에서 정상
판정 받음. 안전하게 이식 대상.

-------------------------------------------------------------------------------
참고문헌
-------------------------------------------------------------------------------
- Teng, H. L., & Powers, C. M. (2014). Sagittal plane trunk posture influences
  patellofemoral joint stress during running. Medicine & Science in Sports &
  Exercise, 46(9), 1739–1747.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from choborunner_ai.pose_extractor import LM, FramePose

logger = logging.getLogger(__name__)


_TRUNK_LANDMARK_INDICES: tuple[int, ...] = (
    LM.LEFT_SHOULDER,
    LM.RIGHT_SHOULDER,
    LM.LEFT_HIP,
    LM.RIGHT_HIP,
)


def _visibility_ok(
    lm: np.ndarray, indices: tuple[int, ...], min_visibility: float
) -> bool:
    """주어진 인덱스의 visibility가 모두 임계 통과인지."""
    for i in indices:
        v = float(lm[i, 3])
        if not np.isfinite(v) or v < min_visibility:
            return False
    return True


def trunk_lean_deg_at_frame(lm: np.ndarray, min_visibility: float = 0.4) -> float:
    """한 프레임에서 trunk lean 각도 (°) — 부호 보존.

    Args:
        lm: (33, 4) MediaPipe Pose 정규화 좌표 + visibility.
        min_visibility: shoulder/hip 4점 모두 통과해야 할 visibility 임계.

    Returns:
        Trunk lean 각도 (degrees). **전방=+, 후방=−** (CLAUDE.md §5-6 부호 보존).
        visibility 미달 또는 trunk 벡터 길이 ~0 시 NaN.
    """
    if not _visibility_ok(lm, _TRUNK_LANDMARK_INDICES, min_visibility):
        return float("nan")
    ls = lm[LM.LEFT_SHOULDER, :2]
    rs = lm[LM.RIGHT_SHOULDER, :2]
    lh = lm[LM.LEFT_HIP, :2]
    rh = lm[LM.RIGHT_HIP, :2]
    shoulder_c = (ls + rs) * 0.5
    hip_c = (lh + rh) * 0.5
    v_trunk = shoulder_c - hip_c  # (x, y) — y는 화면 아래쪽 +
    if float(np.linalg.norm(v_trunk)) < 1e-9:
        return float("nan")
    # docs/2-3-4 §7-4: atan2(v.x, -v.y) — MediaPipe y축 부호 반전, 전방 기울기 +
    theta_rad = float(np.arctan2(float(v_trunk[0]), -float(v_trunk[1])))
    return float(np.degrees(theta_rad))


def compute_trunk_lean_series(
    frames: Iterable[FramePose], min_visibility: float = 0.4
) -> list[float]:
    """프레임 시퀀스에 대한 trunk lean 시계열.

    Args:
        frames: FramePose Iterable (예: `pose_extractor.extract_poses_from_frames` 출력).
        min_visibility: 각 프레임 `trunk_lean_deg_at_frame`로 전달.

    Returns:
        프레임별 trunk lean 각도 list. 포즈 미검출(landmarks=None) 또는 visibility
        미달 시 NaN.
    """
    series: list[float] = []
    for f in frames:
        if f.landmarks is None:
            series.append(float("nan"))
        else:
            series.append(trunk_lean_deg_at_frame(f.landmarks, min_visibility))
    return series
