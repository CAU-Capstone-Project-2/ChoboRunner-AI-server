"""ic_detector.py unit 테스트 (Phase 5-B-3).

5-B-1 sanity (Stage 1 단독) + 5-B-2 sanity (hybrid) pytest 승격 + edge case 보강.

카테고리 (10 case):
A. compute_ic_indices batch wrapper 정상 동작 (부록 D 합성 hybrid MAE baseline)
B. 입력 edge case (empty / all None / None 섞임 / low vis / warmup 부족)
C. MIN_IC_INTERVAL 가드 (짧은 stride)
D. analysis_side 가드 (ValueError)
E. _find_pos_to_neg helper 단독 (정상 / no crossing)

영상 의존 X — 합성 PoseLandmarks (tests/fixtures/synthetic_stride.py).
"""
from __future__ import annotations

from collections import Counter

import pytest

from choborunner_ai.config import ICDetectorConfig
from choborunner_ai.metrics.ic_detector import (
    ICDetector,
    ICResult,
    compute_ic_indices,
)
from choborunner_ai.pose_extractor import Landmark, LandmarkPair, PoseLandmarks
from tests.fixtures.synthetic_stride import (
    DEFAULT_FPS,
    generate_synthetic_stride_series,
)

# Phase 5-B-2 합성 baseline (본 합성 모델 한정, 부록 D anchor와 다름)
HYBRID_BASELINE_MAE_MS = 28.6
TOLERANCE_MS = 5.0


@pytest.fixture
def cfg() -> ICDetectorConfig:
    """default ICDetectorConfig (buffer_size=15, lookahead=3, min_ic_interval=15)."""
    return ICDetectorConfig()


def _match_and_mae(detected: list[int], gt: list[int], fps: int) -> float:
    matched = []
    for g in gt:
        if detected:
            best = min(detected, key=lambda d: abs(d - g))
            matched.append((g, best))
    errs = [abs(d - g) * 1000.0 / fps for g, d in matched]
    return sum(errs) / len(errs) if errs else float("inf")


def _make_static_pose(vis: float = 0.9) -> PoseLandmarks:
    """정적 합성 PoseLandmarks (motion 없음, vis 임의)."""
    def lm(x: float, y: float, v: float = vis) -> Landmark:
        return Landmark(x=x, y=y, visibility=v)

    return PoseLandmarks(
        shoulder=LandmarkPair(left=lm(0.45, 0.25), right=lm(0.55, 0.25)),
        hip=LandmarkPair(left=lm(0.45, 0.55), right=lm(0.55, 0.55)),
        knee=LandmarkPair(left=lm(0.5, 0.7), right=lm(0.5, 0.7)),
        ankle=LandmarkPair(left=lm(0.5, 0.85), right=lm(0.5, 0.85)),
        heel=LandmarkPair(left=lm(0.5, 0.9), right=lm(0.5, 0.9)),
        foot_index=LandmarkPair(left=lm(0.52, 0.92), right=lm(0.52, 0.92)),
    )


# ============================================================
# A. 정상 동작 (부록 D 합성 hybrid baseline)
# ============================================================


def test_synthetic_hybrid_mae_baseline(cfg: ICDetectorConfig):
    """부록 D 합성 hybrid MAE baseline 28.6ms (Phase 5-B-2 anchor) 재현."""
    series, gt = generate_synthetic_stride_series(noise_sigma=0.005)
    results = compute_ic_indices(series, "left", cfg)
    detected = [r.frame_index for r in results]
    assert len(results) == len(gt), f"검출 {len(results)} vs GT {len(gt)}"
    mae = _match_and_mae(detected, gt, DEFAULT_FPS)
    assert abs(mae - HYBRID_BASELINE_MAE_MS) <= TOLERANCE_MS, (
        f"MAE {mae:.1f}ms (baseline {HYBRID_BASELINE_MAE_MS}ms ±{TOLERANCE_MS}ms 외)"
    )
    # confidence 모두 high (Stage 2 zero-crossing 정상)
    confs = Counter(r.confidence for r in results)
    assert confs.get("high", 0) == len(results), f"confidence: {dict(confs)}"


# ============================================================
# B. 입력 edge case
# ============================================================


def test_empty_input(cfg: ICDetectorConfig):
    """빈 list 입력 → 빈 list 반환."""
    results = compute_ic_indices([], "left", cfg)
    assert results == []


def test_all_none_frames(cfg: ICDetectorConfig):
    """모든 frame None → 빈 list (buffer 비어있음, IC 검출 X)."""
    results = compute_ic_indices([None] * 30, "left", cfg)
    assert results == []


def test_mixed_none_frames(cfg: ICDetectorConfig):
    """일부 None 섞임 → None skip, 나머지 frame에서 정상 검출."""
    series, gt = generate_synthetic_stride_series(noise_sigma=0.005)
    # 5번째 frame을 None으로 교체 (warmup 영역, 검출 영향 작음)
    series[5] = None
    results = compute_ic_indices(series, "left", cfg)
    assert len(results) >= 1, "None skip 후 IC 검출 0"
    # frame_index는 enumerate 절대 인덱스 → 정상 frame_idx 유지
    for r in results:
        assert 0 <= r.frame_index < len(series)


def test_low_visibility_no_ic(cfg: ICDetectorConfig):
    """모든 landmark visibility 0.3 (< 0.6) → Stage 1 visibility 가드로 IC 0."""
    series = [_make_static_pose(vis=0.3) for _ in range(30)]
    results = compute_ic_indices(series, "left", cfg)
    assert results == []


def test_buffer_warmup_insufficient(cfg: ICDetectorConfig):
    """buffer warmup 미충족 (3 frame < lookahead*2+1=7) → IC 0."""
    series = [_make_static_pose() for _ in range(3)]
    results = compute_ic_indices(series, "left", cfg)
    assert results == []


# ============================================================
# C. MIN_IC_INTERVAL 가드
# ============================================================


def test_min_ic_interval_guard():
    """짧은 stride (stride_frames=10 < min_ic_interval=15) → 일부 IC skip.

    합성 stride 10 frame이면 GT IC 다수, MIN_IC_INTERVAL 가드로 일부만 검출.
    """
    cfg = ICDetectorConfig()
    series, gt = generate_synthetic_stride_series(
        stride_sec=10 / DEFAULT_FPS,  # 10 frame stride
        duration_sec=3.0,
        noise_sigma=0.003,
    )
    results = compute_ic_indices(series, "left", cfg)
    # GT는 짧은 stride라 더 많지만, MIN_IC_INTERVAL 가드로 검출 IC 수 < GT
    assert len(results) < len(gt), (
        f"검출 {len(results)} >= GT {len(gt)} — MIN_IC_INTERVAL 가드 미작동"
    )


# ============================================================
# D. analysis_side 가드
# ============================================================


def test_analysis_side_invalid_raises(cfg: ICDetectorConfig):
    """analysis_side='top' (잘못된 값) → ValueError."""
    series, _ = generate_synthetic_stride_series(noise_sigma=0.005)
    with pytest.raises(ValueError, match="analysis_side"):
        compute_ic_indices(series, "top", cfg)  # type: ignore[arg-type]


# ============================================================
# E. _find_pos_to_neg helper 단독
# ============================================================


def test_find_pos_to_neg_normal_crossing():
    """v[0]>0, v[1]<=0 정상 crossing → crossings=[0]."""
    velocities = [1.0, -0.5, -1.0, -0.3]
    crossings = ICDetector._find_pos_to_neg(velocities)
    assert crossings == [0]


def test_find_pos_to_neg_no_crossing():
    """양→음 전환 없음 (계속 양수) → 빈 list."""
    velocities = [1.0, 0.5, 0.8, 0.3]
    crossings = ICDetector._find_pos_to_neg(velocities)
    assert crossings == []


def test_find_pos_to_neg_multiple_crossings():
    """여러 crossing 존재 → 모든 인덱스 list (호출자가 min으로 가장 이른 t 채택)."""
    velocities = [1.0, -0.5, 0.3, -0.2, -0.5]
    crossings = ICDetector._find_pos_to_neg(velocities)
    # v[0]>0 + v[1]<=0 (i=0), v[2]>0 + v[3]<=0 (i=2)
    assert crossings == [0, 2]
