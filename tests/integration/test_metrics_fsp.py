"""foot_strike.py integration 테스트 (Phase 5-D-2).

jaemin.mp4 실측 foot_strike baseline 6 metric (5-C-2 패턴 일관).

direction='left_to_right' (본인 결정 박힘, docs §3-4 사용자 선택 정합).

흐름 (5-B-3, 5-C-2 패턴 그대로):
- PoseExtractor + iter_frames → list[PoseLandmarks | None]
- compute_ic_indices → list[ICResult]
- low IC 제외 (docs §4-3): 42 → 28 valid (5-C-2와 동일)
- foot_strike.compute_at_ic(..., "left_to_right", ...) → list[FootStrikeResult]

재현성 검증 PASS (deterministic, 1차/2차 완전 일치):
- VALID_IC_COUNT_BASELINE   = 28 (5-C-2 일치)
- FSP_VALID_COUNT_BASELINE  = 28 (window 모두 통과)
- MEAN_FA_BASELINE_DEG      = -0.4918 (MFS 영역, plantarflexion 살짝)
- STD_FA_BASELINE_DEG       = 2.7634 ← docs §5-6 "stride 표준편차 ~2.9°" 정합!
- 분류 분포: mfs 26 / ffs 2 / rfs 0
  · RFS_RATIO  = 0.0000
  · MFS_RATIO  = 0.9286 (dominant — jaemin은 MFS 러너)
  · FFS_RATIO  = 0.0714
- HEAD5 = [19, 86, 137, 160, 183]
- TAIL5 = [874, 899, 922, 945, 966]

⚠️ docs §5-6 정합 (test 설계 단위):
- "단일 IC 분류 사용 금지" — 본 test는 stride별 개별 분류 외부 노출 X
- fixture 내부에서만 개별 결과 보존, assertion은 통계 (count / mean / 최빈 비율)만
- 누적 최빈값 변환 + 히스테리시스는 별도 Phase (호출자 책임)

⚠️ docs §5-6 anchor 검증: std 2.76° ≈ docs ~2.9° 정합 (학습 자산)
- 본 합성 sanity (5-D-1) 합성 모델은 noise 없는 deterministic 자세
- 실측 (본 integration) std는 docs anchor와 정합 = MediaPipe Pose의 실측 변동성
  이 docs §5-6 가정과 정합 (Day 6 학습 자산)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from choborunner_ai.config import AppConfig
from choborunner_ai.metrics.foot_strike import compute_at_ic as fsp_compute_at_ic
from choborunner_ai.metrics.ic_detector import compute_ic_indices
from choborunner_ai.pose_extractor import PoseExtractor
from choborunner_ai.video_preprocessor import get_video_meta, iter_frames


# === Phase 5-D-2 jaemin.mp4 baseline anchor (재현성 PASS, deterministic) ===
VALID_IC_COUNT_BASELINE = 28
FSP_VALID_COUNT_BASELINE = 28
MEAN_FA_BASELINE_DEG = -0.4918
STD_FA_BASELINE_DEG = 2.7634
MFS_RATIO_BASELINE = 0.9286  # dominant — jaemin MFS 러너

TOLERANCE_COUNT = 1
TOLERANCE_MEAN_DEG = 0.1
TOLERANCE_RATIO = 0.05

JAEMIN_VIDEO_PATH = Path("legacy/demo_02/jaemin.mp4")
DIRECTION = "left_to_right"  # 본인 결정 박힘 (docs §3-4 사용자 선택)


@pytest.fixture(scope="module")
def fsp_measurement() -> dict:
    """jaemin.mp4 fsp 측정 1회 (module scope cache, 3 test 시간 절약).

    direction='left_to_right' + low IC 제외 (docs §4-3 정합).
    """
    if not JAEMIN_VIDEO_PATH.is_file():
        pytest.skip(f"jaemin.mp4 미존재: {JAEMIN_VIDEO_PATH.resolve()}")

    cfg = AppConfig()
    meta = get_video_meta(JAEMIN_VIDEO_PATH)
    fps_safe = meta.fps if meta.fps > 1e-6 else 30.0

    landmarks_series: list = []
    extractor = PoseExtractor(cfg.mediapipe_pose)
    try:
        for idx, frame in enumerate(iter_frames(JAEMIN_VIDEO_PATH)):
            ts_ms = int(idx * 1000.0 / fps_safe)
            pl = extractor.process_frame(frame, timestamp_ms=ts_ms)
            landmarks_series.append(pl)
    finally:
        extractor._landmarker.close()

    ic_results = compute_ic_indices(landmarks_series, "left", cfg.ic)
    valid_ic_indices = [
        r.frame_index for r in ic_results if r.confidence != "low"
    ]

    fsp_results = fsp_compute_at_ic(
        landmarks_series, valid_ic_indices, "left", DIRECTION, cfg.foot_strike
    )
    fsp_valid = [r for r in fsp_results if r.is_valid]
    fa_vals = [r.deg for r in fsp_valid]

    cls_counter = Counter(
        r.classification for r in fsp_valid if r.classification is not None
    )
    total_cls = sum(cls_counter.values())
    # 최빈 분류 비율 (docs §5-6: 통계만 외부, 개별 분류 X)
    mfs_ratio = cls_counter.get("mfs", 0) / total_cls if total_cls else 0.0

    return {
        "valid_ic_count": len(valid_ic_indices),
        "fsp_valid_count": len(fsp_valid),
        "mean_fa": float(np.mean(fa_vals)) if fa_vals else float("nan"),
        "std_fa": float(np.std(fa_vals)) if fa_vals else float("nan"),
        "mfs_ratio": mfs_ratio,
    }


def test_valid_ic_count_baseline(fsp_measurement: dict):
    """low 제외 후 IC 개수 baseline 28 ±1 (5-C-2 일치)."""
    count = fsp_measurement["valid_ic_count"]
    assert abs(count - VALID_IC_COUNT_BASELINE) <= TOLERANCE_COUNT, (
        f"valid IC {count} (baseline {VALID_IC_COUNT_BASELINE} ±{TOLERANCE_COUNT} 외)"
    )


def test_mean_foot_angle_baseline(fsp_measurement: dict):
    """평균 foot angle baseline -0.4918° ±0.1° (MFS 영역)."""
    mean = fsp_measurement["mean_fa"]
    assert abs(mean - MEAN_FA_BASELINE_DEG) <= TOLERANCE_MEAN_DEG, (
        f"mean fa {mean:.4f}° "
        f"(baseline {MEAN_FA_BASELINE_DEG}° ±{TOLERANCE_MEAN_DEG}° 외)"
    )


def test_dominant_classification_ratio_baseline(fsp_measurement: dict):
    """최빈 분류 (MFS) 비율 baseline 92.86% ±5%.

    docs §5-6 정합: 개별 stride 분류 결과 외부 노출 X, 분포 통계만.
    jaemin은 MFS 러너 (mean -0.49° = -5 < θ < +5 영역).
    """
    ratio = fsp_measurement["mfs_ratio"]
    assert abs(ratio - MFS_RATIO_BASELINE) <= TOLERANCE_RATIO, (
        f"MFS 비율 {ratio:.4f} "
        f"(baseline {MFS_RATIO_BASELINE} ±{TOLERANCE_RATIO} 외)"
    )
