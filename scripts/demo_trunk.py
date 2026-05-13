"""
Vertical Slice Demo — Trunk Lean only.

본 스크립트는 회의 시연용 최소 동작 데모.
7개 지표 중 Trunk Lean 1개만 시연하며,
Knee Flexion / Foot Strike Pattern은 의도적으로 제외.
이유: src/choborunner_ai/metrics/trunk.py docstring 참조.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from choborunner_ai import pose_extractor, video_preprocessor
from choborunner_ai.metrics import trunk

# PowerShell cp949 콘솔에서 한국어·°·— 깨짐 방지
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_INPUT = Path("legacy/demo_02/jaemin.mp4")
DEFAULT_MODEL = Path("legacy/demo_02/models/pose_landmarker_lite.task")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vertical Slice Demo — Trunk Lean only.",
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
        help=f"PoseLandmarker .task 모델 경로 (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--min-visibility",
        type=float,
        default=0.4,
        help="shoulder/hip 4점 visibility 임계 (default: 0.4)",
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
        meta = video_preprocessor.get_video_meta(args.input)
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

    frames_iter = video_preprocessor.iter_frames(
        args.input, frame_stride=args.frame_stride
    )

    try:
        poses = pose_extractor.extract_poses_from_frames(
            frames=frames_iter,
            fps=meta.fps,
            model_path=args.model,
        )
    except FileNotFoundError as e:
        logger.error("Pose 추출 실패: %s", e)
        return 1

    series = trunk.compute_trunk_lean_series(poses, min_visibility=args.min_visibility)

    valid = [x for x in series if np.isfinite(x)]
    mean_val = float(np.mean(valid)) if valid else float("nan")

    print()
    print("=== Trunk Lean Vertical Slice Demo ===")
    print(f"Input: {args.input.name} ({len(series)} frames, {meta.fps:.2f} fps)")
    print(f"Per-frame (head 5): {_fmt_list(series[:5])}")
    print(f"Per-frame (tail 5): {_fmt_list(series[-5:])}")
    print(f"Mean: {'nan' if not np.isfinite(mean_val) else f'{mean_val:.1f}°'}")
    print(f"Valid frames: {len(valid)} / {len(series)}")
    print("(Knee/FSP excluded — see metrics/trunk.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
