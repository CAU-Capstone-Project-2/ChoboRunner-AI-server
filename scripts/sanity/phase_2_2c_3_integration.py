"""Phase 2-2c-3 통합 검증 — mp4 frame -> PoseExtractor.process_frame end-to-end.

CLI 인자:
- --video PATH (default: legacy/demo_02/jaemin.mp4) — 본인 PC 로컬 검증용,
  영상 수집 후 새 영상으로 교체 자연
- --frame-index INT (default: 100)
- --timestamp-ms INT (default: 33) — 호출자 부여 (option C, PoseExtractor 시간 정책 미보유)

위치 결정: scripts/sanity/ (사람 실행 검증 + 발표 시연 자산).
이전 tmp/ 위치는 사용자 트리거 미스 catch 후 scripts/sanity/로 이전 — git 추적 단위
정착, legacy/demo_02 결합도는 --video 인자로 완화.

견고성 가드:
- 영상 미존재 -> FileNotFoundError + --video 안내
- cap.isOpened()/cap.read() 실패 -> 명확한 RuntimeError
- process_frame None -> "FAIL: 추출 실패" + 재시도 옵션 안내 (자동 재시도 X)
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import argparse
import logging
import time
from pathlib import Path

import cv2

from choborunner_ai.config import MediaPipePoseConfig
from choborunner_ai.pose_extractor import PoseExtractor

DEFAULT_VIDEO = Path("legacy/demo_02/jaemin.mp4")
DEFAULT_FRAME_INDEX = 100
DEFAULT_TIMESTAMP_MS = 33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2-2c-3 통합 검증 — mp4 frame -> PoseLandmarks end-to-end."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help=(
            f"검증 영상 경로 (default: {DEFAULT_VIDEO}). "
            "본인 PC 로컬 — 영상 수집 후 교체 자연."
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=DEFAULT_FRAME_INDEX,
        help=f"추출할 frame index (default: {DEFAULT_FRAME_INDEX}).",
    )
    parser.add_argument(
        "--timestamp-ms",
        type=int,
        default=DEFAULT_TIMESTAMP_MS,
        help=(
            f"process_frame timestamp_ms (default: {DEFAULT_TIMESTAMP_MS}). "
            "호출자 부여 정책 (option C)."
        ),
    )
    return parser.parse_args()


def extract_frame(video_path: Path, frame_index: int):
    if not video_path.is_file():
        raise FileNotFoundError(
            f"검증 영상 없음: {video_path.resolve()}\n"
            f"  --video PATH 인자로 본인 영상 경로 지정 가능."
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture 열기 실패: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"frame 추출 실패: frame_index={frame_index}, cap.read() False"
            )
        return frame
    finally:
        cap.release()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print(f"[1/3] {args.video} frame_index={args.frame_index} 추출")
    try:
        frame = extract_frame(args.video, args.frame_index)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"FAIL: {e}")
        return 1
    print(f"  frame shape={frame.shape}, dtype={frame.dtype}")

    print(f"\n[2/3] PoseExtractor 초기화 (cfg.frame_timeout_sec=0.5 default)")
    cfg = MediaPipePoseConfig()
    extractor = PoseExtractor(cfg)

    try:
        print(f"\n[3/3] process_frame(timestamp_ms={args.timestamp_ms}) 호출")
        t_start = time.perf_counter()
        pl = extractor.process_frame(frame, timestamp_ms=args.timestamp_ms)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if pl is None:
            print(f"\nFAIL: 추출 실패 - process_frame returned None")
            print(
                f"  elapsed={elapsed_ms:.1f}ms "
                f"(cfg.frame_timeout_sec={cfg.frame_timeout_sec}s)"
            )
            print(
                f"  원인 후보: (1) polling timeout 초과 "
                f"또는 (2) MediaPipe pose 미검출 (_convert_result None)"
            )
            print(
                f"  재시도 옵션: cfg.frame_timeout_sec=1.0 으로 재실행 "
                f"(본인 결정, 자동 재시도 X)"
            )
            return 1

        print(f"\nPASS: PoseLandmarks 추출 성공")
        print(
            f"  elapsed={elapsed_ms:.1f}ms "
            f"(cfg.frame_timeout_sec={cfg.frame_timeout_sec}s)"
        )
        print(
            f"\n핵심 4 landmark (피사체 좌측, docs §3-4 normalized 좌표 x,y in [0,1]):"
        )
        print(
            f"  shoulder.left : x={pl.shoulder.left.x:.4f}, "
            f"y={pl.shoulder.left.y:.4f}, vis={pl.shoulder.left.visibility:.4f}"
        )
        print(
            f"  hip.left      : x={pl.hip.left.x:.4f}, "
            f"y={pl.hip.left.y:.4f}, vis={pl.hip.left.visibility:.4f}"
        )
        print(
            f"  knee.left     : x={pl.knee.left.x:.4f}, "
            f"y={pl.knee.left.y:.4f}, vis={pl.knee.left.visibility:.4f}"
        )
        print(
            f"  ankle.left    : x={pl.ankle.left.x:.4f}, "
            f"y={pl.ankle.left.y:.4f}, vis={pl.ankle.left.visibility:.4f}"
        )
        print(
            f"\n  to_numpy shape: {pl.to_numpy().shape} (expected (12, 3))"
        )
        full = pl.landmarks_full
        print(
            f"  landmarks_full: {'None' if full is None else full.shape} "
            f"(debug_mode={cfg.debug_mode})"
        )
        return 0
    finally:
        extractor._landmarker.close()


if __name__ == "__main__":
    sys.exit(main())
