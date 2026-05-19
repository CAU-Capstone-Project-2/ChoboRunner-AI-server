# Phase 9 분할 anchor (Day 8 진입 트리거 기반)

## 박힌 결정 (Day 7 5/17 commit 5809aa9 직후)

### A-1 β: 2 sub-phase 분리

- Phase 9-A: target_switch_detected
- Phase 9-B: unstable_landmark_sequence
- 근거: 회귀 catch 분산 + Phase 8-E/8-B 분할 패턴 일관 + commit history 가독성

### A-2 β: 별도 함수

- `evaluate_target_switch(landmarks_series, fps, cfg)`
- `evaluate_unstable_landmark(landmarks_series, ic_indices, fps, cfg)`
- 근거: 시그니처 명확, reason_code별 책임 분리, 입력 데이터 다름

### A-3: 계산식 옵션 (Day 8 2단계 진입 시 lock)

- pelvis 잔차: α sliding window 평균 대비 / β max-min / γ stddev
- scale: α shoulder-shoulder / β hip-hip / γ hip-shoulder
- mid-stance frame: α 단일 (IC[n]+IC[n+1])/2 / β window ±N frame
- 정책: docs §4 직역 + 주석 박음 + docs 보강 후보 (Phase D 패턴 학습 적용)

### A-4 α: 즉시 진입 (Day 8 첫 작업)

- docs §4 임계 명시됨, 백엔드 의존 X

### A-5 α: list[Optional[PoseLandmarks]] 입력 (Phase 8 패턴)

### A-6: sub-phase별 commit + sanity (Phase 8 패턴)

## Day 8 진입 분량 추정

- Phase 9-A: ~120 line, ~1.5h (sanity ~5 case + pytest 3~5 case)
- Phase 9-B: ~150 line, ~2h (mid-stance window 계산 추가)
- 합: ~3~3.5h

## ⚠️ 영향 받는 site

- quality_gate.py: ReasonCode +2 (Phase 8-E 보류 2건 박힘)
- feedback_engine.py: REASON_CODE_USER_MESSAGES 16 → 18
- result_serializer.py: REASON_CODE_PRIORITY 17 → 19
- pipeline.py: 호출 2 line

## ⚠️ Phase D vs Phase 9 차이 (학습 자산)

- Phase D: schema 자체 미정의 → 보류 (백엔드 미팅 후)
- Phase 9: 임계 명시 ✓, 계산식만 모호 → 진입 OK (옵션 선택 + 주석)


## 추가 결정 (Day 8 진입 후, Phase 9-A Step 1)

### B-1 β: config 단일 SoT — TrackingStabilityConfig 활용 (TargetSwitchConfig 신규 X)

- 결정: 기존 `TrackingStabilityConfig`에 `target_switch_consecutive_frames=5`
  1 필드만 추가 (Phase 8-E `evaluate_tracking_stability`와 동일 cfg)
- 근거: 임계 단일 SoT (pelvis_spike 0.15 / scale_spike 0.20 / visibility
  0.4 / 3 window seconds 모두 기존 활용), drift 위험 0, Phase 8-E 패턴 일관
- catch: 사용자 트리거 명시 "TargetSwitchConfig 신규 (~30 line)"는 잘못된 추정 —
  실제 β로 ~10 line

### B-2: 5 frame 연속 정책 docs 보강 후보

- docs §4-3 본문: "동시 발생" + "일시 붕괴"만 명시 — frame 수 정책 X
- Phase 9-A: 5 frame 연속 heuristic 박음 (옆 사람 통과 4 frame은 trigger X,
  옆 사람 정착 10 frame trigger O)
- docs §4-3 보강 후보 (Phase D 패턴 학습 적용) — 향후 docs 업데이트 시 본 anchor 참조
