"""trunk_lean.py integration 테스트 (Phase 5-A-2).

jaemin.mp4 회귀 baseline (Phase 5-A-2 정식판 anchor):
- MEAN_BASELINE_DEG = 3.651° (정식판, visibility 0.6, docs §7-2 정합)
- TOLERANCE_DEG = 0.1°
- 재현성 검증 완료 (소수점 15자리까지 1차/2차 완전 일치, 영상 deterministic +
  첫 frame timeout 일관)

History anchor — Vertical Slice (Day 3 commit 359184b) Mean 3.9°:
- 측정 환경 visibility 0.4 (docs 미정합) — demo path 임시값.
- Phase 5-A-2 integration test가 docs §7-2 (visibility 0.6 정합) baseline으로
  갱신 (Day 5 catch). 3.9°는 git history에만 남고 본 모듈 anchor는 3.651°.

⚠️ 첫 frame timeout 1 frame 모집단 제외:
- MediaPipe LIVE_STREAM 첫 frame warmup ~0.6s > cfg.frame_timeout_sec=0.5s →
  process_frame None 반환 → landmarks_series에 None 포함 → compute_series가
  NaN 반환 → finite_vals 모집단에서 자동 제외 (Phase 6-C baseline 일관).

⚠️ Vertical Slice trunk.py는 5-A-3에서 폐기 예정. 본 test는 정식판 단독
anchor — Vertical Slice 의존 X (Day 5 decision γ).

⚠️ PoseExtractor는 with-as 미지원 (Pipeline만 with-as 지원). 본 test는
try/finally 패턴 — 사용자 명세 (vi) "PoseExtractor 인스턴스 1개 + close 보장"
의도로 해석. Pipeline 미사용 사유: Phase 6 Pipeline.run_on_video_file은
visibility 검증 결과만 반환, trunk lean 산출 미통합 (Phase 6 scope §5-1만).

조립 흐름 (Day 5 decision B, Pipeline 미사용):
- PoseExtractor 1개 인스턴스 + try/finally
- iter_frames(jaemin.mp4) iterate
- process_frame -> PoseLandmarks | None
- compute_series -> [float]
- Mean (finite_vals only)

실행 시간: 첫 frame warmup + 982 frame × ~110ms ≈ 75~110초.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from choborunner_ai.config import AppConfig
from choborunner_ai.metrics.trunk_lean import compute_series
from choborunner_ai.pose_extractor import PoseExtractor
from choborunner_ai.video_preprocessor import get_video_meta, iter_frames


# 정식판 (Phase 5-A-2, docs §7-2 visibility 0.6 정합) jaemin.mp4 Mean baseline.
# 재현성 검증 (소수점 15자리 1차/2차 일치, deterministic).
MEAN_BASELINE_DEG = 3.651
TOLERANCE_DEG = 0.1


def test_jaemin_mean_matches_baseline(
    jaemin_video_path: Path, app_cfg: AppConfig
):
    """jaemin.mp4 정식판 Mean이 baseline 3.651° ±0.1° 안인지 검증.

    첫 frame timeout 1 frame은 finite_vals 모집단에서 자동 제외.
    """
    video_meta = get_video_meta(jaemin_video_path)
    fps_safe = video_meta.fps if video_meta.fps > 1e-6 else 30.0

    landmarks_series: list = []
    extractor = PoseExtractor(app_cfg.mediapipe_pose)
    try:
        for idx, frame in enumerate(iter_frames(jaemin_video_path)):
            ts_ms = int(idx * 1000.0 / fps_safe)
            pl = extractor.process_frame(frame, timestamp_ms=ts_ms)
            landmarks_series.append(pl)
    finally:
        extractor._landmarker.close()

    series_deg = compute_series(landmarks_series, app_cfg.trunk_lean)
    finite_vals = [v for v in series_deg if math.isfinite(v)]
    assert finite_vals, "valid trunk lean frame 0개 (visibility 또는 입력 점검 필요)"

    mean_deg = float(np.mean(finite_vals))
    assert abs(mean_deg - MEAN_BASELINE_DEG) <= TOLERANCE_DEG, (
        f"Mean {mean_deg:.3f}° "
        f"(baseline {MEAN_BASELINE_DEG}° ±{TOLERANCE_DEG}° 범위 외)"
    )
