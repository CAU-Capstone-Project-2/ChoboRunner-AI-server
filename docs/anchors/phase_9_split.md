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

### B-3: Step 2 결정 3건 lock (Phase 9-A 계산식)

- 결정 1 — pelvis 잔차: α (sliding window 평균 대비 |차이|)
  · anchor A-3 lock 그대로
- 결정 2 — scale: γ (hip-shoulder 수직 거리)
  · Phase 8-B-2 `torso_yaw_proxy` 분모 패턴 재사용 (측면 robust)
  · `|mean(hip.y) - mean(shoulder.y)|` (정규화 좌표, 키에 비례 안정 척도)
- 결정 3 — window 산출 정책: α (단순 산술 평균)
  · Phase 8-E `evaluate_tracking_stability` sliding window 패턴 일관
  · ⚠️ docs §4-2-1 "중앙값 기반 편차" 직역과 다름 — 향후 Phase 8-E 일관
    변경 시 같이 보강 후보 (미래의 본인 헷갈리지 않게 박음, Phase D 패턴 학습 적용)

### B-4: visibility 평가 정책 — frame-level 절대값 채택 (Step 2 catch 해소)

- 결정: visibility 조건은 frame-level 절대값 `visibility[t] < 0.4`
  (window 평균 사용 X)
- 근거:
  · pelvis / scale은 **변동성 신호** (window baseline 필요) — sliding window 평균 대비
  · visibility는 **붕괴 신호** (절대값 frame 단위) — 사람이 바뀌는 순간 자체 drop
  · 5 frame 연속 정책 (B-2)과 자연 정합 (window 1초 평균은 trigger 사실상 불가:
    위반 frame이 window 30 frame 중 22+ 필요)
- catch (Step 2 실측):
  · 초기 구현 window 1초 평균 채택 시 C 케이스 (10 frame 위반) trigger 실패
  · window 평균이 < 0.4 되려면 위반 frame이 window의 73% 이상 — 5 frame 연속과 모순
- ⚠️ docs §4-2 sliding window 1초 패턴과 의미 분화 — pelvis/scale는 변동성,
  visibility는 붕괴. docs §4-3에 본 의미 분화 보강 후보.
- ⚠️ TrackingStabilityConfig.visibility_window_seconds 필드는 본 함수 미사용 —
  Phase 8-E `evaluate_tracking_stability` (target_lost reason_code)에서 활용 중,
  cfg 필드 제거 X.
