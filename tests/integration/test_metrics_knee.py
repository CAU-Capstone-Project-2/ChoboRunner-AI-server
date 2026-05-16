"""knee_flexion.py integration 테스트 (Phase 5-C-2).

jaemin.mp4 실측 knee baseline 6 metric (5-A-2 + 5-B-3 패턴 결합):

흐름 (decision iii, docs §4-3 정합):
- PoseExtractor + iter_frames → list[PoseLandmarks | None]
- compute_ic_indices → list[ICResult]
- low IC 제외: ic_indices = [r.frame_index for r in ic_results if r.confidence != "low"]
  5-B-3 jaemin baseline 42 IC → 본 모듈 입력 28 IC (high+medium)
- knee_flexion.compute_at_ic(series, ic_indices, "left", cfg) → list[KneeFlexionResult]
- is_valid stride 필터 (window 50% 통과)

재현성 검증 PASS (deterministic, 1차/2차 완전 일치):
- VALID_IC_COUNT_BASELINE     = 28 (low 제외 후, docs §4-3 정합)
- KNEE_VALID_COUNT_BASELINE   = 27 (window 50% 통과 stride)
- MEAN_KF_BASELINE_DEG        = 27.7988
- STD_KF_BASELINE_DEG         = 5.8537
- 분류 분포: typical 8 / above_typical 18 / below_typical 1
- TYPICAL_RATIO_BASELINE      = 0.2963
- ABOVE_RATIO_BASELINE        = 0.6667 (above 우세, docs §6-5 임계 보정 필요 신호)
- HEAD5 = [19, 86, 137, 160, 183]
- TAIL5 = [874, 899, 922, 945, 966]

⚠️ 분류 분포 catch (학습 자산):
- jaemin 평균 kf 27.8°가 docs §6-6 default 임계 (above_typical ≥ 25°) 초과
- CLAUDE.md §5-2 v1 schema PoC 42.2° (단일 frame) vs 본 정식판 27.8° (IC±window 평균)
- 정식판이 v1 대비 훨씬 합리적 (14°↓ 개선)
- 단 docs §6-5 "2D vs 3D mocap 절대값 차이, 파일럿 보정 필요" 정확 케이스

조립 흐름 (Pipeline 미사용, Phase 5-A-2 + 5-B-3 패턴 일관):
- module-scope knee_measurement fixture (1회 측정 cache, 3 test 공유)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from choborunner_ai.config import AppConfig
from choborunner_ai.metrics.ic_detector import compute_ic_indices
from choborunner_ai.metrics.knee_flexion import compute_at_ic as knee_compute_at_ic
from choborunner_ai.pose_extractor import PoseExtractor
from choborunner_ai.video_preprocessor import get_video_meta, iter_frames


# === Phase 5-C-2 jaemin.mp4 knee baseline anchor (재현성 검증 완료) ===
VALID_IC_COUNT_BASELINE = 28
KNEE_VALID_COUNT_BASELINE = 27
MEAN_KF_BASELINE_DEG = 27.7988
TYPICAL_RATIO_BASELINE = 0.2963
ABOVE_RATIO_BASELINE = 0.6667

TOLERANCE_COUNT = 1
TOLERANCE_MEAN_DEG = 0.1
TOLERANCE_RATIO = 0.05

JAEMIN_VIDEO_PATH = Path("legacy/demo_02/jaemin.mp4")


@pytest.fixture(scope="module")
def knee_measurement() -> dict:
    """jaemin.mp4 knee 측정 1회 (module scope cache, 3 test 시간 절약).

    PoseExtractor + compute_ic_indices + low 제외 + knee_compute_at_ic 흐름.
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

    # IC detector + low 제외 (docs §4-3 정합)
    ic_results = compute_ic_indices(landmarks_series, "left", cfg.ic)
    valid_ic_indices = [
        r.frame_index for r in ic_results if r.confidence != "low"
    ]

    # knee compute_at_ic
    knee_results = knee_compute_at_ic(
        landmarks_series, valid_ic_indices, "left", cfg.knee_flexion
    )
    knee_valid = [r for r in knee_results if r.is_valid]
    kf_vals = [r.deg for r in knee_valid]

    cls_counter = Counter(
        r.classification for r in knee_valid if r.classification is not None
    )
    total_cls = sum(cls_counter.values())

    return {
        "valid_ic_count": len(valid_ic_indices),
        "knee_valid_count": len(knee_valid),
        "mean_kf": float(np.mean(kf_vals)) if kf_vals else float("nan"),
        "std_kf": float(np.std(kf_vals)) if kf_vals else float("nan"),
        "cls_counter": dict(cls_counter),
        "typical_ratio": (
            cls_counter.get("typical", 0) / total_cls if total_cls else 0.0
        ),
        "above_ratio": (
            cls_counter.get("above_typical", 0) / total_cls if total_cls else 0.0
        ),
    }


def test_valid_ic_count_baseline(knee_measurement: dict):
    """low 제외 후 IC 개수 baseline 28 ±1."""
    count = knee_measurement["valid_ic_count"]
    assert abs(count - VALID_IC_COUNT_BASELINE) <= TOLERANCE_COUNT, (
        f"valid IC {count} (baseline {VALID_IC_COUNT_BASELINE} ±{TOLERANCE_COUNT} 외)"
    )


def test_mean_knee_flexion_baseline(knee_measurement: dict):
    """평균 knee flexion baseline 27.7988° ±0.1°."""
    mean = knee_measurement["mean_kf"]
    assert abs(mean - MEAN_KF_BASELINE_DEG) <= TOLERANCE_MEAN_DEG, (
        f"mean kf {mean:.4f}° "
        f"(baseline {MEAN_KF_BASELINE_DEG}° ±{TOLERANCE_MEAN_DEG}° 외)"
    )


def test_classification_typical_ratio_baseline(knee_measurement: dict):
    """typical 비율 baseline 29.63% ±5% (분류 분포 합리성 검증)."""
    ratio = knee_measurement["typical_ratio"]
    assert abs(ratio - TYPICAL_RATIO_BASELINE) <= TOLERANCE_RATIO, (
        f"typical 비율 {ratio:.4f} "
        f"(baseline {TYPICAL_RATIO_BASELINE} ±{TOLERANCE_RATIO} 외)"
    )
