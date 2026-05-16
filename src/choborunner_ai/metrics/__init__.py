"""choborunner_ai.metrics — 핵심 자세 지표 산출 모듈 (docs/2-3-4).

각 지표는 별도 submodule에 위치 (docs/2-3-4 §1 단일 정답):
- trunk_lean.py     : Trunk Lean — 상체 전경사 (§7) [Phase 5-A 정식판]
- knee_flexion.py   : Initial Knee Flexion (§6) [placeholder]
- foot_strike.py    : Foot Strike Pattern (§5) [placeholder]
- ic_detector.py    : Initial Contact 검출 (§4) [placeholder]

공개 인터페이스는 각 submodule에서 직접 import 권장 (re-export X — submodule
간 결합 최소화):

    from choborunner_ai.metrics.trunk_lean import (
        TrunkLeanResult,
        trunk_lean_deg,
        compute_series,
        compute_at_ic,
        classify,
    )

호환 모드 A 정책 (CLAUDE.md §7):
- 정식판: PoseLandmarks 인터페이스 (Phase 3 6 LandmarkPair).
- demo path FramePose(33점 ndarray)는 metrics/trunk.py Vertical Slice — 단계적
  deprecate (Phase 5-A-3에서 폐기, demo_trunk.py는 정식판 호출로 변경).
"""
