"""Phase 4-C 통합 sanity — quality_gate end-to-end (evaluate_frame_visibility +
evaluate_visibility_accumulation).

합성 PoseLandmarks 시퀀스 -> 4-A N번 -> 4-B -> reason code 누적 결과.

CLI 인자:
- --frames-total INT       (default 40)
- --invalid-from INT       (default 30) — N번째 frame부터 invalid (0-indexed)
- --invalid-category       (default 'foot') — 'foot' | 'lower_body' | 'upper_body' | 'overall'
- --analysis-side          (default 'left') — 'left' | 'right'

invalid-category -> 영향 받는 landmark:
- foot       : heel + foot_index
- lower_body : hip + knee + ankle
- upper_body : shoulder (양측)
- overall    : 6 landmark 모두 (12점 전체)

invalid 구간 frame은 위 landmark visibility=0.3, 나머지는 0.9.

default 시나리오 의도 (40 / 30 / foot / left):
- 0~29 frame: 12 landmark vis 0.9 -> 4 카테고리 통과 (is_valid=True)
- 30~39 frame: heel/foot_index vis 0.3 -> foot 평균 0.3 < 0.6 -> foot_not_visible
- valid 30 / total 40 -> ratio 0.75 >= 0.6 -> evaluate_visibility_accumulation: []
- 의도: "충분히 통과" 시나리오 (invalid 10/40 만 있어도 누적 통과)

위치: scripts/sanity/ (Phase 2-2c-3 C3 결정 일관, 사람 실행 + 발표 자산).
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import argparse

from choborunner_ai.config import VisibilityCheckConfig
from choborunner_ai.pose_extractor import Landmark, LandmarkPair, PoseLandmarks
from choborunner_ai.quality_gate import (
    FrameVisibilityResult,
    evaluate_frame_visibility,
    evaluate_visibility_accumulation,
)

INVALID_VIS = 0.3
NORMAL_VIS = 0.9

CATEGORY_LANDMARKS: dict[str, set[str]] = {
    "foot": {"heel", "foot_index"},
    "lower_body": {"hip", "knee", "ankle"},
    "upper_body": {"shoulder"},
    "overall": {"shoulder", "hip", "knee", "ankle", "heel", "foot_index"},
}


def make_landmark(vis: float) -> Landmark:
    return Landmark(x=0.0, y=0.0, visibility=vis)


def make_pose(
    invalid_landmarks: set[str], invalid_vis: float, normal_vis: float
) -> PoseLandmarks:
    """6 LandmarkPair 합성 — invalid_landmarks 집합 내 이름은 invalid_vis 적용."""

    def pair(name: str) -> LandmarkPair:
        v = invalid_vis if name in invalid_landmarks else normal_vis
        return LandmarkPair(left=make_landmark(v), right=make_landmark(v))

    return PoseLandmarks(
        shoulder=pair("shoulder"),
        hip=pair("hip"),
        knee=pair("knee"),
        ankle=pair("ankle"),
        heel=pair("heel"),
        foot_index=pair("foot_index"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4-C 통합 sanity — evaluate_frame_visibility + accumulation "
            "end-to-end (합성 PoseLandmarks 시퀀스)."
        )
    )
    parser.add_argument("--frames-total", type=int, default=40)
    parser.add_argument(
        "--invalid-from",
        type=int,
        default=30,
        help="N번째 frame부터 invalid (0-indexed). frames-total보다 크면 invalid 0개.",
    )
    parser.add_argument(
        "--invalid-category",
        choices=["foot", "lower_body", "upper_body", "overall"],
        default="foot",
    )
    parser.add_argument(
        "--analysis-side", choices=["left", "right"], default="left"
    )
    return parser.parse_args()


def category_means(results: list[FrameVisibilityResult]) -> dict[str, float]:
    """구간 평균 — 빈 list면 0.0 default."""
    keys = ["lower_body", "foot", "upper_body", "overall_avg"]
    if not results:
        return {k: 0.0 for k in keys}
    return {
        k: sum(r.category_averages[k] for r in results) / len(results) for k in keys
    }


def main() -> int:
    args = parse_args()

    if args.frames_total <= 0:
        print(f"FAIL: frames-total > 0 필수, got {args.frames_total}")
        return 1
    if args.invalid_from < 0 or args.invalid_from > args.frames_total:
        print(
            f"FAIL: invalid-from은 [0, frames-total={args.frames_total}] 범위, "
            f"got {args.invalid_from}"
        )
        return 1

    cfg = VisibilityCheckConfig()
    invalid_lms = CATEGORY_LANDMARKS[args.invalid_category]

    # 합성 시퀀스 생성 + 4-A 평가
    frame_results: list[FrameVisibilityResult] = []
    for idx in range(args.frames_total):
        used = invalid_lms if idx >= args.invalid_from else set()
        pl = make_pose(used, INVALID_VIS, NORMAL_VIS)
        r = evaluate_frame_visibility(
            pl, analysis_side=args.analysis_side, cfg=cfg
        )
        frame_results.append(r)

    # 4-B 누적 평가
    final_reasons = evaluate_visibility_accumulation(frame_results, cfg)

    valid_count = sum(1 for r in frame_results if r.is_valid)
    invalid_count = len(frame_results) - valid_count
    ratio = valid_count / len(frame_results)

    print("[Phase 4-C Integration Sanity]")
    print(
        f"  args: frames_total={args.frames_total}, "
        f"invalid_from={args.invalid_from}, "
        f"invalid_category={args.invalid_category}, "
        f"analysis_side={args.analysis_side}"
    )
    print(f"  cfg.valid_frame_ratio_min={cfg.valid_frame_ratio_min}")
    print()
    print("[Frame 평가 요약]")
    print(f"  Total frames        : {len(frame_results)}")
    print(f"  Valid frames        : {valid_count} ({ratio*100:.1f}%)")
    print(f"  Invalid frames      : {invalid_count} ({(1-ratio)*100:.1f}%)")
    print()

    if args.invalid_from > 0:
        normal_results = frame_results[: args.invalid_from]
        normal_avg = category_means(normal_results)
        print(
            f"[카테고리 평균 - 정상 구간 (0 ~ {args.invalid_from-1}, "
            f"N={len(normal_results)})]"
        )
        for cat, v in normal_avg.items():
            print(f"  {cat:<12}: {v:.4f}")
    if args.invalid_from < args.frames_total:
        invalid_results = frame_results[args.invalid_from:]
        invalid_avg = category_means(invalid_results)
        print(
            f"[카테고리 평균 - Invalid 구간 ({args.invalid_from} ~ "
            f"{args.frames_total-1}, N={len(invalid_results)})]"
        )
        for cat, v in invalid_avg.items():
            print(f"  {cat:<12}: {v:.4f}")
    print()

    print("[Accumulation 결과]")
    print(f"  final reason codes  : {final_reasons}")
    if not final_reasons:
        print(f"  accumulation 판정    : PASS (visibility 누적 통과)")
    else:
        print(
            f"  accumulation 판정    : FAIL ({', '.join(final_reasons)})"
        )

    # 의도 일관성 검증 (자동)
    expected_pass = ratio >= cfg.valid_frame_ratio_min
    actual_pass = not final_reasons
    if expected_pass == actual_pass:
        comparison = ">=" if expected_pass else "<"
        print(
            f"\nPASS: 의도 일치 — ratio {ratio*100:.1f}% {comparison} "
            f"{cfg.valid_frame_ratio_min*100:.0f}% threshold"
        )
        return 0
    print(
        f"\nFAIL: 의도 불일치 — ratio {ratio*100:.1f}%이지만 "
        f"accumulation 판정 다름 (expected_pass={expected_pass}, actual_pass={actual_pass})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
