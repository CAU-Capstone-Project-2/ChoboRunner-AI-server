# CLAUDE.md — ChoboRunner AI Server

Claude Code 및 다른 AI 도구가 이 레포에서 작업할 때 가장 먼저 읽는 파일.
프로젝트 컨벤션과 핵심 의사결정 기록.

> 📝 개인 로컬 규칙(작업자별 선호도 등)은 `CLAUDE.local.md`에 별도 보관 (gitignore).
> 본 파일은 팀 공유 컨텍스트만 다룬다.

---

## 1. 프로젝트 개요

**ChoboRunner**는 초보 러너를 위한 실시간 러닝 자세 분석 서비스 (캡스톤 디자인). 본 레포는 그중 **AI 서버** — Python 라이브러리 + FastAPI 진입점.

- **범위 (v1)**: 측면 단일 카메라, MediaPipe Pose, 2D 영상 좌표계 분석
- **출력**: 3개 핵심 지표 (Trunk Lean, Initial Knee Flexion, Foot Strike Pattern) + Rule-based 피드백
- **면책**: 의료 진단이 아닌 참고용 피드백. 모든 JSON 응답에 `reference_feedback_only: true` 필수

**팀 구성**:
- AI: Tae-young Kim (본 레포)
- Backend (Spring/Java): 재민 — `ChoboRunner-Backend` 레포
- Frontend (Android): 정우 — `ChoboRunner_Frontend` 레포
- AI ↔ Backend 연결은 WebSocket (설계문서 2-4-2). Backend가 영상 frame을 보내면 AI가 분석 결과 반환.

**legacy 컨텍스트**:
- 옛 레포 `ChoboRunner-AI`는 그대로 둠 (백엔드 연결용 작업이 push되어 있음)
- 본 레포는 학기 말 정리 시점에 옛 레포 처리 결정
- demo2 PoC 코드는 작업자 PC에 있음 → Step 5에서 `legacy/`로 이식

---

## 2. 기술 스택

| 영역 | 도구 |
|---|---|
| 언어 | Python 3.11.5 |
| 가상환경 | venv (`.venv/`) |
| Pose | MediaPipe Tasks (PoseLandmarker, lite model) |
| 영상 IO / 시각화 | OpenCV |
| 수학 | NumPy (1.x — 2.0과 mediapipe 호환성 이슈 회피) |
| 서버 | FastAPI |
| 검증 / 스키마 | Pydantic v2 + pydantic-settings |
| 패키지 매니저 | pip |
| 테스트 | pytest |
| 린트 / 포맷 | ruff (선택) |

**환경 메타**:
- OS: Windows
- Shell: PowerShell
- Python 위치: `.venv\Scripts\python.exe` (활성화 후)
- PowerShell 실행 정책: `RemoteSigned` 권장

---

## 3. 디렉토리 구조 (현재)

```
ChoboRunner-AI-server/
├── .venv/                          (gitignore됨, Python 3.11.5)
├── .gitignore                      (Python·venv·IDE·OS·secrets·media 제외)
├── README.md                       (프로젝트 소개)
├── CLAUDE.md                       (본 파일 - 팀 공유)
├── pyproject.toml                  (Step 2에서 작성)
│
├── docs/                           (Step 6에서 채움)
│   ├── 0-index.md                  설계문서 통합 인덱스
│   └── 2-3-{1..7}.md               7개 설계문서
│
├── src/choborunner_ai/             Python 패키지 (실제 라이브러리)
│   ├── __init__.py
│   ├── config.py                   2-3-1~7 통합 설정 (Step 3)
│   ├── input_validator.py          2-3-1
│   ├── video_preprocessor.py       2-3-2
│   ├── pose_extractor.py           2-3-3
│   ├── metrics/                    2-3-4 (큰 모듈이라 폴더)
│   │   ├── __init__.py
│   │   ├── ic_detector.py          Zeni 2008 + Fellin 2010 hybrid
│   │   ├── trunk_lean.py
│   │   ├── knee_flexion.py
│   │   └── foot_strike.py
│   ├── quality_gate.py             2-3-5
│   ├── feedback_engine.py          2-3-6
│   ├── result_serializer.py        2-3-7 (Pydantic 모델)
│   └── pipeline.py                 모듈 오케스트레이션
│
├── server/                         FastAPI 얇은 레이어
│   ├── __init__.py
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── analyze.py              POST /analyze (batch, 우선)
│   │   └── stream.py               WebSocket (재민과 인터페이스 협의 후)
│   └── schemas.py                  요청·응답 Pydantic
│
├── tests/
│   ├── __init__.py
│   ├── unit/                       각 모듈 단위 테스트
│   ├── integration/                파이프라인 통합 테스트
│   └── fixtures/                   합성 landmark, 테스트 영상
│
├── scripts/                        CLI 도구
│   ├── run_local.py                영상 1개 batch 분석
│   ├── benchmark_ic.py             IC hybrid 합성 데이터 검증
│   ├── visualize_overlay.py        결과 시각화
│   └── fetch_pose_model.py         demo2에서 이식 (모델 다운로드)
│
└── legacy/                         (Step 5에서 demo2 코드 이식 예정)
```

---

## 4. 설계문서 = 단일 정답 (Single Source of Truth)

모든 알고리즘·정책 결정은 `docs/2-3-{1..7}.md`에 정의됨. 코드는 이 문서를 충실히 구현한다.
**코드와 설계가 어긋나면 → 코드를 고치거나 설계문서를 업데이트한다. 무시하고 진행 금지.**

| 문서 | 모듈 | 단일 정답 |
|---|---|---|
| 2-3-1 | `input_validator.py` | 입력 영상 규격, 촬영 가이드 |
| 2-3-2 | `video_preprocessor.py` | 프레임 정규화, frame-level 품질 |
| 2-3-3 | `pose_extractor.py` | MediaPipe 사용, 6종 landmark |
| 2-3-4 | `metrics/` | 자세 지표 계산식, IC hybrid |
| 2-3-5 | `quality_gate.py` | Pose 후 품질 검사, 상태값, reason code |
| 2-3-6 | `feedback_engine.py` | 트리거 룰, 메시지 사전, 빈도 정책 |
| 2-3-7 | `result_serializer.py` | 응답 메시지 4종 |

설계문서들은 Step 6에서 docs/에 배치된다.

---

## 5. 핵심 구현 결정사항

### 5-1. IC 검출 — True Zeni 2008 + Fellin 2010 lookahead hybrid

**결정**: 설계문서 2-3-4 v2.1대로 **true Zeni 2008** (foot-sacrum AP 속도 zero-crossing) + **Fellin 2010 lookahead 보정** 구현.

**배경**: PoC(demo2)는 단순 "heel-pelvis x 피크" 방식이었음. 본 레포는 설계대로 새로 작성.

**재사용 가능 (demo2에서 그대로 이식)**:
- `find_local_maxima_indices`, `nms_peaks` — peak 유틸
- `_resolve_stance_side` — visibility 합으로 좌/우 결정
- `refine_representative_frame` — IC 주변 ±N 프레임 refine

**인용**:
- Zeni, J. A., Richards, J. G., & Higginson, J. S. (2008). *Gait & Posture, 27(4), 710–714.*
- Fellin, R. E., Rose, W. C., Royer, T. D., & Davis, I. S. (2010). *J Sci Med Sport, 13(6), 646–650.*

### 5-2. 임계값 — 모두 `config.py`, 재교정 필요는 ⚠️

**결정**: 분류 임계값은 `config.py`의 `ClassificationThresholds`에만 존재. interpretation·feedback에서 직접 박지 말 것.

**⚠️ 중요 — Knee Flexion 임계값**:
4개 임계값 (`knee_stiff_below_deg`, `knee_optimal_low_deg`, `knee_optimal_high_deg`, `knee_excessive_above_deg`)은 **3D mocap 기반**인데 측정은 **2D MediaPipe 정규화 좌표**. 체계적 mismatch 위험. PoC 테스트(jaemin 케이스, knee=42.2°)에서 평범한 러너가 `excessive_tendency`로 오분류됨.

**v1 PoC 정책**: 일단 demo2 임계값 유지 + 코드에 경고 주석. 영상 파일럿 5~10개 수집 후 재교정 (별도 마일스톤). 임의 변경 금지 — 작업자와 합의 후에만.

### 5-3. 좌표계 — 2D 영상 좌표만 (v1)

MediaPipe 정규화 영상 좌표 (x ∈ [0,1], y ∈ [0,1], y는 화면 아래쪽 +). World landmarks(3D) 사용 안 함. v2 평가 예정.

### 5-4. Quality Gate — demo2 score 폐기, 2-3-5 신규

demo2의 `quality_score` (`metrics.py` 가중치 합산)는 underutilized 확인 (4개 IC 점수 모두 4.158~4.163, 변별력 없음). 본 레포는 폐기하고 **2-3-5 설계의 단순 게이트 로직** 사용.

### 5-5. Knee Flexion — 윈도 평균 추가

demo2는 IC 단일 프레임 측정 → 한 프레임 튐에 취약. 본 레포는 **trunk와 동일하게 IC ±2 프레임 평균**.

### 5-6. Trunk Lean — 부호 보존

demo2는 `clip(c, 0, 1)`로 항상 [0, 90]° 절대값화. 본 레포는 **부호 보존** (전방=+, 후방=−). 디버깅·재교정 시 정보 손실 방지.

### 5-7. 출력 스키마 — schema 1.1 → 2.0, Pydantic 모델

demo2는 dict 기반 schema 1.1. 본 레포는 Pydantic `BaseModel` 기반 schema 2.0. FastAPI 자동 문서화 활용.

---

## 6. 코드 컨벤션

- **모듈화**: 설계문서 1개 = 모듈 1개. 한 파일 300줄 미만 유지.
- **로직과 오케스트레이션 분리**: `pipeline.py`는 조립만. 계산 로직은 각 모듈.
- **Pydantic 일관성**: 설정 = `BaseSettings`, API 스키마 = `BaseModel`. 모듈 경계에서 raw dict 금지.
- **매직넘버 금지**: 모든 상수는 `config.py`. 인용 또는 `# ⚠️ heuristic` 표시 필수.
- **`print()` 금지**: `logging.getLogger(__name__)`. CLI 스크립트만 예외.
- **타입 힌트 필수**: PEP-585 (`list[int]`, `dict[str, float]`). `from typing import List` 금지.
- **Docstring**: Google 스타일. 임계값·알고리즘 상수는 출처 인용 포함.
- **언어**: 코드 식별자 영어, 사용자 노출 메시지 한국어, 주석 한국어 가능.
- **Line length**: 100자 (ruff 설정).

---

## 7. demo2 마이그레이션 정책

| 처리 | 비율 | 항목 |
|---|---|---|
| **거의 그대로 이식** | ~70% | `video_io.py`, `smoothing.py`, `overlay.py` (POSE_CONNECTIONS, draw 함수), 각도 계산식 (`trunk_lean_deg_at_frame`, `knee_flexion_deg_at_frame`, `foot_strike_*`), `pose_extractor.py` 골격 (LM enum, FramePose), `scripts/fetch_*` |
| **재구성 후 이식** | ~20% | `landmark_prep.py` (파이프라인 컨셉 유지), `config.py` (Pydantic Settings + 매직넘버 흡수), `output_formatter.py` → `result_serializer.py` (Pydantic) |
| **새로 작성** | ~10% | `ic_detector.py` (Zeni + Fellin), `quality_gate.py` (2-3-5), `feedback_engine.py` (2-3-6), FastAPI 서버, 모든 테스트 |

**legacy/ 폴더**: Step 5에서 demo2 전체를 `legacy/demo_02/`에 복사. `.gitignore`에 추가하지 않음 (참고용으로 git에 포함). 단, 큰 영상 파일은 `legacy/demo_02/storage/` 등 제외.

---

## 8. 임계값 관리 원칙

코드에서 상수 발견 시:

1. `config.py`에 있는 값인가? ✅ OK.
2. 순수 수학 가드 (예: `1e-9` for division-by-zero)? ✅ 인라인 + 주석 OK.
3. 그 외? ❌ `config.py`로 이동, 인용 추가.

새 임계값 추가 시:

1. 해당 `Config` 모델에 `Field(...)` 추가
2. `description=` 에 학술 인용 OR `# ⚠️ heuristic, needs validation`
3. 논문 기반이면 모듈 docstring에 풀 인용

---

## 9. 테스트 정책

| 모듈 | 최소 커버리지 | 테스트 형태 |
|---|---|---|
| `metrics/` (각 지표) | 100% | 합성 landmark → 알려진 각도 |
| `metrics/ic_detector.py` | 80%+ | 합성 stride 시계열 (sin파 + 노이즈) |
| `quality_gate.py`, `feedback_engine.py` | 80%+ | 합성 입력 + 경계값 |
| 분류 임계값 | 100% (경계값) | 9.99 vs 10.0, 14.99 vs 15.0 |
| 파이프라인 통합 | 영상 1~2개 | demo2의 jaemin 영상 회귀 검증 |

demo2는 0건 → 본 레포는 **모든 PR에 테스트 동반**.

---

## 10. 모듈 완료 기준 (Definition of Done)

각 모듈이 "완료"되려면:

- [ ] 대응 설계문서 전체 구현
- [ ] 모든 임계값이 `config.py` (인용 또는 ⚠️ 동반)
- [ ] 단위 테스트 통과
- [ ] 타입 힌트 + Docstring
- [ ] 구현 파일에 매직넘버 없음
- [ ] demo2 회귀 검토 (잃은 동작·잘못된 동작 짚어보기)

---

## 11. Step 진행 상황 (2026-05-10 기준)

| Step | 내용 | 상태 |
|---|---|---|
| 0 | 레포 생성, 환경 셋업, git 초기화 | ✅ 완료 |
| 1 | 디렉토리 구조 + 빈 파일 36개 | ✅ 완료 |
| **2** | **`pyproject.toml` + 의존성 설치** | **다음** |
| 3 | `config.py` (Pydantic Settings) | 대기 |
| 4 | CLAUDE.md (본 파일) | ✅ 완료 |
| 5 | demo2 → `legacy/` 이식 | 대기 |
| 6 | 설계문서 7개 docs/에 배치 | 대기 |

**Step 2 진입 시 작업 가이드**:

1. `pyproject.toml` 작성 (PEP 621 표준, src/ 레이아웃 인식)
2. 핵심 의존성 8개:
   - `mediapipe>=0.10.9`
   - `opencv-python>=4.8.0`
   - `numpy>=1.24.0,<2.0.0` (mediapipe 호환성)
   - `pydantic>=2.5.0`
   - `pydantic-settings>=2.1.0`
   - `fastapi>=0.110.0`
   - `uvicorn[standard]>=0.27.0`
   - `python-multipart>=0.0.9`
3. dev 그룹 (선택): `pytest`, `pytest-cov`, `ruff`
4. `pip install -e .` 로 editable install
5. `python -c "import choborunner_ai"` 으로 import 검증

설치 시간: 5~15분 (mediapipe·opencv 큰 패키지). 인터넷·시간 확보 후 진행. Step 2도 작은 단계로 분할 권장 (예: 2-A pyproject.toml 작성, 2-B install, 2-C 검증).

---

## 12. 작업 중 막힐 때

- 설계문서 X, 코드 Y → 작업자에게 묻기. 가정해서 진행 금지.
- 임계값이 이상해 보임 → 플래그만 달고 변경 금지. 작업자와 논의.
- demo2엔 있는데 신규엔 없는 동작 → 7번 마이그레이션 표 먼저 확인. 의도적일 수 있음.
- 의료 면책 (`reference_feedback_only: true`) 제거 변경은 절대 금지.
- 환경 에러 (PowerShell, Python, pip) → 에러 메시지 그대로 작업자에게 공유 후 함께 해결.

---

## 13. 변경 이력

- 2026-05-10 v1: 초안. Step 0~1 완료 시점, Step 2 직전. demo2 분석 결과 + 작업자 결정사항.
- 2026-05-10 v2: 작업 모드 규칙 추가.
- 2026-05-10 v3: 개인 규칙은 `CLAUDE.local.md`로 분리. 본 파일은 팀 공유 컨텍스트만.
