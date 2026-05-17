"""Phase 6-C 통합 sanity — Pipeline.run_on_video_file end-to-end.

CLI 인자:
- --video PATH                 (default legacy/demo_02/jaemin.mp4)
- --analysis-side {left|right} (default left)
- --max-frames INT             (default 30) — 빠른 PoC 시연용
- --full flag                  — max_frames=None으로 full run.
                                 --max-frames 동시 지정 시 --full 우선.

print 형식 (β 요약):
- args / video_meta / frame 평가 요약 / 카테고리 평균 / accumulation 결과
- elapsed_total_sec (시연 자산, MediaPipe 처리 속도 baseline)

견고성:
- --video 미존재 -> FAIL + 명확한 메시지
- --max-frames 음수 -> FAIL
- pose_landmarks_count == 0 (모두 pose 미검출) -> FAIL (입력 또는 모듈 점검 신호)
- accumulation FAIL은 sanity 자체 PASS 유지 ("동작 확인"이지 "통과 보장" X)

위치: scripts/sanity/ (Phase 2-2c-3 C3 결정 일관, 사람 실행 + 발표 자산).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import argparse
import logging
import time
from pathlib import Path

from choborunner_ai.config import AppConfig
from choborunner_ai.pipeline import Pipeline
from choborunner_ai.quality_gate import FrameVisibilityResult

DEFAULT_VIDEO = Path("legacy/demo_02/jaemin.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6-C 통합 sanity — Pipeline end-to-end (file path mode)."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"검증 영상 경로 (default: {DEFAULT_VIDEO}). 본인 PC 로컬.",
    )
    parser.add_argument(
        "--analysis-side", choices=["left", "right"], default="left"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="처리 frame 수 (default 30, 빠른 PoC). --full 동시 지정 시 무시.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="full run (max_frames=None). --max-frames보다 우선.",
    )
    return parser.parse_args()


def category_means(
    frame_results: list[FrameVisibilityResult],
) -> dict[str, float]:
    keys = ["lower_body", "foot", "upper_body", "overall_avg"]
    if not frame_results:
        return {k: 0.0 for k in keys}
    return {
        k: sum(r.category_averages[k] for r in frame_results) / len(frame_results)
        for k in keys
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.video.is_file():
        print(f"FAIL: video 미존재 {args.video.resolve()}")
        return 1
    if args.max_frames is not None and args.max_frames < 0:
        print(f"FAIL: --max-frames 음수 ({args.max_frames})")
        return 1

    max_frames_effective = None if args.full else args.max_frames

    print("[Phase 6-C Integration Sanity]")
    print(
        f"  args: video={args.video}, analysis_side={args.analysis_side}, "
        f"max_frames={max_frames_effective} (full={args.full})"
    )
    print()

    cfg = AppConfig()
    t_start = time.perf_counter()
    with Pipeline(cfg) as p:
        result = p.run_on_video_file(
            args.video,
            analysis_side=args.analysis_side,
            max_frames=max_frames_effective,
        )
    elapsed_total_sec = time.perf_counter() - t_start

    # video_meta
    vm = result.video_meta
    duration_sec = vm.frame_count / vm.fps if vm.fps > 0 else 0.0
    print("[Video Meta]")
    print(f"  resolution      : {vm.width} x {vm.height}")
    print(f"  fps             : {vm.fps:.3f}")
    print(f"  total_frame_count: {vm.frame_count}")
    print(f"  duration        : {duration_sec:.2f}s")
    print(f"  rotation_degrees: {vm.rotation_degrees}")
    print()

    # frame 평가
    processed = result.pose_landmarks_count + result.pose_not_detected_count
    pose_detected_ratio = (
        result.pose_landmarks_count / processed if processed else 0.0
    )
    valid_count = sum(1 for r in result.frame_results if r.is_valid)
    visibility_valid_ratio = (
        valid_count / result.pose_landmarks_count
        if result.pose_landmarks_count
        else 0.0
    )
    print("[Frame 평가 요약]")
    print(f"  Processed frames    : {processed}")
    print(
        f"  pose_detected       : {result.pose_landmarks_count} "
        f"({pose_detected_ratio*100:.1f}%)"
    )
    print(
        f"  pose_not_detected   : {result.pose_not_detected_count} "
        f"({(1-pose_detected_ratio)*100:.1f}%)"
    )
    if result.pose_landmarks_count > 0:
        print(
            f"  visibility valid    : {valid_count} "
            f"({visibility_valid_ratio*100:.1f}%)"
        )
        print(
            f"  visibility invalid  : {result.pose_landmarks_count - valid_count} "
            f"({(1-visibility_valid_ratio)*100:.1f}%)"
        )
    else:
        print("  visibility valid    : N/A (pose_landmarks_count=0)")
    print()

    # 카테고리 평균
    if result.frame_results:
        avgs = category_means(result.frame_results)
        print("[카테고리 평균 (pose_detected frame 전체)]")
        for cat, v in avgs.items():
            print(f"  {cat:<12}: {v:.4f}")
    else:
        print("[카테고리 평균] N/A (frame_results 빈 list)")
    print()

    # accumulation (Phase 8-H — accumulation_reasons → reason_code_entries 통합)
    # ⚠️ Phase 8-H lock 8-H-3 α: Phase 8-B-1 δ 시그니처 잔재 해소.
    #   기존 list[ReasonCode] 단일 §5-1 필드 → list[ReasonCodeEntry] 통합
    #   (§5-1~§5-7 + §4 모두 누적).
    print("[Reason Code Entries (Phase 8-A~8-E 누적)]")
    if not result.reason_code_entries:
        print("  판정: PASS (모든 reason code 통과)")
    else:
        print(f"  entry count: {len(result.reason_code_entries)}")
        for entry in result.reason_code_entries:
            print(f"    - {entry.reason_code} ({entry.severity})")
    print()

    # elapsed
    print("[Elapsed]")
    print(f"  total run time      : {elapsed_total_sec:.2f}s")
    if processed > 0:
        print(
            f"  per-frame avg       : "
            f"{elapsed_total_sec*1000/processed:.1f}ms/frame"
        )
    print()

    # 최종 PASS/FAIL — sanity 자체 "동작 확인" 기준
    if result.pose_landmarks_count == 0:
        print(
            "FAIL: pose_landmarks_count == 0 (모두 pose 미검출 - "
            "입력 또는 모듈 점검 필요)"
        )
        return 1
    print("PASS: Pipeline end-to-end 동작 확인")
    if result.reason_code_entries:
        print(
            "  (reason code entries 누적되었지만 sanity 자체는 '동작 확인'이라 PASS - "
            "임계값 보정 필요 신호일 수 있음)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
