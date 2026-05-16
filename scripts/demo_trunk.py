"""Trunk Lean 정식판 시연 데모 (Phase 5-A-3).

PoseLandmarks 인터페이스 (Phase 3 정식) 기반, docs/2-3-4 §7 정합.
3개 핵심 지표 중 Trunk Lean 1개만 시연, Knee Flexion / Foot Strike Pattern은
별도 metrics Phase 진입 시 통합 예정.

이력:
- Day 3 Vertical Slice (commit 359184b): FramePose 33점 ndarray +
  extract_poses_from_frames + min_visibility 0.4 (docs §7-2 미정합) →
  Mean 3.9°. Phase 5-A-3에서 폐기.
- Phase 5-A-3 (현재): 정식판 호출 (PoseExtractor LIVE_STREAM + PoseLandmarks
  6 LandmarkPair + trunk_lean.compute_series) + visibility 0.6 (docs §7-2
  정합) → Mean 3.651° baseline.

조립 흐름 (Phase 5-A-2 integration test 패턴 일관):
- AppConfig DI (--model / --min-visibility CLI 인자로 sub-config override)
- PoseExtractor 1개 인스턴스 + try/finally (close 보장)
- iter_frames + process_frame → PoseLandmarks | None
- compute_series → list[float] (NaN 포함)
- classify → 분류 분포 (시연 자산)
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from choborunner_ai.config import AppConfig, MediaPipePoseConfig, TrunkLeanConfig
from choborunner_ai.metrics.trunk_lean import classify, compute_series
from choborunner_ai.pose_extractor import PoseExtractor
from choborunner_ai.video_preprocessor import get_video_meta, iter_frames

# PowerShell cp949 콘솔에서 한국어·°·— 깨짐 방지
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_INPUT = Path("legacy/demo_02/jaemin.mp4")
DEFAULT_MODEL = Path("assets/models/pose_landmarker_lite.task")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trunk Lean 정식판 시연 데모 (docs/2-3-4 §7).",
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"입력 영상 경로 (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=(
            f"PoseLandmarker .task 모델 경로 (default: {DEFAULT_MODEL}, "
            "Phase 3 정식판 default)"
        ),
    )
    p.add_argument(
        "--min-visibility",
        type=float,
        default=0.6,
        help="shoulder/hip 4점 visibility 임계 (default: 0.6, docs §7-2 정합)",
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="N frame마다 1 frame 처리 (default: 1, 모든 frame)",
    )
    return p.parse_args(argv)


def _fmt_list(xs: list[float], decimals: int = 1) -> str:
    return "[" + ", ".join(
        ("nan" if not np.isfinite(x) else f"{x:.{decimals}f}") for x in xs
    ) + "]"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.is_file():
        logger.error("입력 영상 파일이 없음: %s", args.input.resolve())
        return 1
    if not args.model.is_file():
        logger.error("PoseLandmarker 모델 파일이 없음: %s", args.model.resolve())
        return 1

    try:
        meta = get_video_meta(args.input)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("영상 메타 조회 실패: %s", e)
        return 1

    logger.info(
        "입력: %s — %dx%d, fps=%.2f, frame_count=%d, rotation=%d°",
        args.input,
        meta.width,
        meta.height,
        meta.fps,
        meta.frame_count,
        meta.rotation_degrees,
    )

    # cfg DI — CLI args로 sub-config override (다른 sub-config는 default 자동 채움)
    cfg = AppConfig(
        mediapipe_pose=MediaPipePoseConfig(model_path=args.model),
        trunk_lean=TrunkLeanConfig(visibility_min=args.min_visibility),
    )

    # PoseExtractor 1개 인스턴스 + try/finally (Phase 5-A-2 integration 패턴)
    fps_safe = meta.fps if meta.fps > 1e-6 else 30.0
    landmarks_series: list = []
    extractor = PoseExtractor(cfg.mediapipe_pose)
    try:
        for idx, frame in enumerate(
            iter_frames(args.input, frame_stride=args.frame_stride)
        ):
            ts_ms = int(idx * 1000.0 / fps_safe)
            pl = extractor.process_frame(frame, timestamp_ms=ts_ms)
            landmarks_series.append(pl)
    finally:
        extractor._landmarker.close()

    # 시리즈 계산
    series = compute_series(landmarks_series, cfg.trunk_lean)
    valid = [x for x in series if np.isfinite(x)]
    mean_val = float(np.mean(valid)) if valid else float("nan")

    # 분류 분포 (docs §7-6 시연 자산)
    classifications = [classify(deg, cfg.trunk_lean) for deg in series]
    cls_counter = Counter(c for c in classifications if c is not None)
    total_classified = sum(cls_counter.values())

    print()
    print("=== Trunk Lean 정식판 (Phase 5-A) ===")
    print(f"Input: {args.input.name} ({len(series)} frames, {meta.fps:.2f} fps)")
    print(f"Per-frame (head 5): {_fmt_list(series[:5])}")
    print(f"Per-frame (tail 5): {_fmt_list(series[-5:])}")
    print(
        f"Mean: {'nan' if not np.isfinite(mean_val) else f'{mean_val:.3f}°'}"
    )
    print(f"Valid frames: {len(valid)} / {len(series)}")
    print()
    print(
        f"분류 분포 (docs §7-6, visibility {args.min_visibility} 통과 frame 기준):"
    )
    for cls_name in ("near_vertical", "forward_lean", "above_typical"):
        count = cls_counter.get(cls_name, 0)
        ratio = (count / total_classified * 100.0) if total_classified else 0.0
        print(f"  {cls_name:<14}: {count:4d} ({ratio:5.1f}%)")
    invalid_count = len(series) - total_classified
    print(f"  (NaN / 미분류    : {invalid_count:4d})")
    print()
    print(
        "(Knee Flexion / Foot Strike Pattern 미포함 - 별도 metrics Phase 통합 예정)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
