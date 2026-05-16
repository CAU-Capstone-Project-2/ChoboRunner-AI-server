"""ic_detector.py integration 테스트 (Phase 5-B-3).

jaemin.mp4 실측 IC baseline 6 metric (Phase 5-A-2 패턴 일관):
- IC 개수
- 평균 stride time (s) + 표준편차
- confidence 분포 (high / medium / low 비율)
- 시작 5 + 끝 5 IC frame_index

재현성 검증 (1차/2차 완전 일치, deterministic):
- IC_COUNT_BASELINE = 42
- MEAN_STRIDE_BASELINE_SEC = 0.7698 (±0.0464 std)
- CONF_HIGH_RATIO_BASELINE = 0.6667
- CONF_MEDIUM_RATIO_BASELINE = 0.0000 (Stage 2 zero-crossing 큰 offset 없음)
- CONF_LOW_RATIO_BASELINE = 0.3333 (실측 noise 영향, Stage 2 실패 33%)
- HEAD5 = [19, 41, 64, 86, 111]
- TAIL5 = [874, 899, 922, 945, 966]

실측 분포 합리성:
- 평균 stride 0.77s → 케이던스 ~156 SPM (러닝 합리적 150~180 SPM)
- IC 42개 / 32.73s = 1.28 IC/s
- confidence high 66.67% — Stage 2 zero-crossing 성공률, 실측 noise 영향
- low 33.33% — 누적 통계 제외 대상 (docs §4-3 정책)

조립 흐름 (Phase 5-A-2 패턴 일관, Pipeline 미사용):
- PoseExtractor + iter_frames + process_frame → list[PoseLandmarks | None]
- compute_ic_indices(landmarks, "left", cfg.ic) → list[ICResult]
- module-scope fixture로 1회 측정 cache (3 test 시간 절약)

통합 sanity (trunk_lean.compute_at_ic 연결 등)는 본 Phase 5-B-3 scope 밖
(decision v β) — Phase 5 통합 마일스톤 (5-E 또는 Pipeline 통합 anchor).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from choborunner_ai.config import AppConfig
from choborunner_ai.metrics.ic_detector import compute_ic_indices
from choborunner_ai.pose_extractor import PoseExtractor
from choborunner_ai.video_preprocessor import get_video_meta, iter_frames


# === Phase 5-B-3 jaemin.mp4 baseline anchor (재현성 검증 완료) ===
IC_COUNT_BASELINE = 42
MEAN_STRIDE_BASELINE_SEC = 0.7698
CONF_HIGH_RATIO_BASELINE = 0.6667

TOLERANCE_COUNT = 1
TOLERANCE_STRIDE_SEC = 0.05
TOLERANCE_RATIO = 0.05

JAEMIN_VIDEO_PATH = Path("legacy/demo_02/jaemin.mp4")


@pytest.fixture(scope="module")
def ic_measurement() -> dict:
    """jaemin.mp4 IC 측정 1회 (module scope cache, 3 test 공유 시간 절약).

    PoseExtractor 1개 + iter_frames + compute_ic_indices 흐름 (decision v β,
    Pipeline 미사용 — Phase 6 §5-1만 통합, IC 미통합).
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

    results = compute_ic_indices(landmarks_series, "left", cfg.ic)
    frame_indices = [r.frame_index for r in results]
    strides = (
        [
            (frame_indices[i + 1] - frame_indices[i]) / fps_safe
            for i in range(len(frame_indices) - 1)
        ]
        if len(frame_indices) >= 2
        else []
    )

    conf_counter = Counter(r.confidence for r in results)
    total = len(results)
    return {
        "results": results,
        "frame_indices": frame_indices,
        "total": total,
        "mean_stride_sec": float(np.mean(strides)) if strides else float("nan"),
        "std_stride_sec": float(np.std(strides)) if strides else float("nan"),
        "conf_counter": conf_counter,
        "conf_high_ratio": (
            conf_counter.get("high", 0) / total if total else 0.0
        ),
    }


def test_ic_count_baseline(ic_measurement: dict):
    """IC 개수 baseline 42 ±1 (deterministic)."""
    count = ic_measurement["total"]
    assert abs(count - IC_COUNT_BASELINE) <= TOLERANCE_COUNT, (
        f"IC 개수 {count} (baseline {IC_COUNT_BASELINE} ±{TOLERANCE_COUNT} 외)"
    )


def test_mean_stride_baseline(ic_measurement: dict):
    """평균 stride time baseline 0.7698s ±0.05s (러닝 합리적 0.7~0.8s)."""
    mean = ic_measurement["mean_stride_sec"]
    assert abs(mean - MEAN_STRIDE_BASELINE_SEC) <= TOLERANCE_STRIDE_SEC, (
        f"평균 stride {mean:.4f}s "
        f"(baseline {MEAN_STRIDE_BASELINE_SEC}s ±{TOLERANCE_STRIDE_SEC}s 외)"
    )


def test_confidence_high_ratio_baseline(ic_measurement: dict):
    """confidence 'high' 비율 baseline 66.67% ±5% (Stage 2 zero-crossing 성공률)."""
    ratio = ic_measurement["conf_high_ratio"]
    assert abs(ratio - CONF_HIGH_RATIO_BASELINE) <= TOLERANCE_RATIO, (
        f"high 비율 {ratio:.4f} "
        f"(baseline {CONF_HIGH_RATIO_BASELINE} ±{TOLERANCE_RATIO} 외)"
    )
