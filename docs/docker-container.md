Docker Container. ChoboRunner AI 서버 컨테이너 설계

> **작성·소유: 백엔드 (재민).** docs/2-4-2와 마찬가지로 백엔드가 소유하는 인프라/배포 문서다. AI 서버(`server/` + `choborunner_ai` 라이브러리)를 **어떻게 컨테이너 이미지로 빌드하고 배포 환경에서 실행하는가**의 단일 정답이다. WebSocket transport 규약은 docs/2-4-2, 응답 메시지 본문은 docs/2-3-7이 단일 정답이며 본 문서는 그 둘을 바꾸지 않는다.

## 1. 이 문서를 한눈에

### 이 문서가 정의하는 것

- AI 서버를 담는 **Docker 이미지의 멀티 스테이지 빌드 구조** — Python 빌드 단계(builder)와 실행 단계(runtime)의 완전 분리
- 배포 환경에서 **이미지를 실행해 배포**하는 흐름 — 컨테이너 레지스트리 push/pull
- WebSocket 엔드포인트와 **별개의 healthcheck HTTP 엔드포인트** 설계
- 컨테이너 런타임 구성 — 포트, 환경변수 override, 비루트 실행, 리소스
- MediaPipe 모델 파일·OpenCV 의존성 등 컨테이너화에 필요한 **선행 조정 항목**

### 이 문서가 정의하지 않는 것

| 알고 싶은 것 | 가야 할 문서 |
| --- | --- |
| WebSocket binary frame wire format·`stop` 신호·세션 생명주기는? | **docs/2-4-2** |
| 응답 메시지 4종의 필드 스펙은? | **docs/2-3-7** |
| 자세 지표 계산·IC 검출·품질 게이트는? | **docs/2-3-4 / 2-3-5** |
| Spring relay·Android 송신 측 배포·인프라는? | **백엔드 레포 `ChoboRunner-Backend`** |
| CI/CD 파이프라인(빌드·푸시·EC2 배포 자동화)은? | **docs/ci-cd-pipeline** |

### 핵심 결정 5가지

1. **멀티 스테이지 빌드.** Python 의존성 설치·패키지 빌드는 `builder` 스테이지에서만 수행하고(요구사항 #1), `runtime` 스테이지는 그 결과물(venv)만 복사한다. 최종 이미지에는 pip·컴파일러·빌드 캐시가 **존재하지 않는다**.
2. **배포 환경은 빌드하지 않는다.** 이미지는 CI/개발 머신에서 빌드해 **컨테이너 레지스트리**에 push하고, 배포 환경은 그 이미지를 pull해 `docker run`만 한다(요구사항 #2). 배포 환경에 소스·Python·빌드 도구가 필요 없다.
3. **healthcheck는 별도 HTTP 엔드포인트.** `GET /healthz`를 신규 추가하고(요구사항 #3), Docker `HEALTHCHECK` 지시어가 이를 호출한다. WebSocket 추론 엔드포인트(`/ws/inference`)와 완전히 분리된다.
4. **단일 책임 컨테이너.** 컨테이너 1개 = uvicorn AI 서버 1개. "WS 연결 1개 = `Pipeline` 1개"라는 컨테이너 **내부** 동시성 모델(docs/2-4-2 §7-4)은 그대로다 — 본 문서는 그 프로세스를 감싸는 컨테이너만 다룬다.
5. **이미지 경량화.** `opencv-python` → `opencv-python-headless` 전환 권장(§5), 비루트 유저 실행, 런타임 의존성 최소화.

---

## 2. 컨테이너화 대상·전제

| 항목 | 값 |
| --- | --- |
| 컨테이너가 담는 것 | uvicorn으로 구동되는 `server.main:app` (FastAPI) |
| 노출 엔드포인트 | `WS /ws/inference` (추론, docs/2-4-2) + `GET /healthz` (healthcheck, §6 — 신규) |
| 노출 포트 | `8000` (컨테이너 내부) |
| 베이스 이미지 | `python:3.11-slim` (pyproject `requires-python = ">=3.11,<3.12"` 정합) |
| 빌드 환경 | CI 또는 개발 머신 — 이미지 빌드·레지스트리 push |
| 실행(배포) 환경 | Docker만 설치 — 이미지 pull·`docker run` (소스·Python 불필요) |

> builder와 runtime 스테이지는 **반드시 동일 베이스 이미지(`python:3.11-slim`)** 를 쓴다. venv를 스테이지 간 통째로 복사하므로 Python 버전·공유 라이브러리가 동일해야 바이너리 호환된다.

---

## 3. 멀티 스테이지 빌드 설계

### 3-1. 스테이지 분리 원칙 (요구사항 #1)

```
┌─ Stage 1: builder ──────────────┐      ┌─ Stage 2: runtime ─────────────┐
│ python:3.11-slim                │      │ python:3.11-slim               │
│ • venv 생성 (/opt/venv)         │      │ • 런타임 시스템 라이브러리만   │
│ • pip install . (의존성·패키지) │ ───▶ │ • /opt/venv 복사 (builder 산출)│
│ • 빌드 캐시·pip 메타데이터 잔존 │ venv │ • server/ + 모델 복사          │
│                                 │ 복사 │ • 비루트 유저로 uvicorn 실행   │
└─ 최종 이미지에 포함 ✗ ──────────┘      └─ 최종 이미지 = 이것만 ✓ ───────┘
```

- **builder**: `pip install`로 의존성을 받아 venv를 만든다. pip 캐시·빌드 산출물은 이 스테이지에 남고 최종 이미지로 넘어가지 않는다.
- **runtime**: builder가 만든 `/opt/venv`만 복사한다. pip·빌드 도구 없이 실행에 필요한 것만 담는다 → 이미지 경량·공격 표면 축소.

### 3-2. builder 스테이지

- 격리 venv(`/opt/venv`)를 만들고 `PATH`에 추가한다.
- `pyproject.toml` + `README.md`(pyproject `readme`가 참조) + `src/`를 복사한 뒤 `pip install .`로 **정식 설치**한다(editable 아님 — 컨테이너에는 소스 마운트가 없다).
- 현재 의존성 8종은 전부 cp311 manylinux wheel을 제공하므로 **컴파일러가 필요 없다.** 향후 wheel 미제공 의존성이 추가되면 builder에 `build-essential`을 설치한다.

### 3-3. runtime 스테이지

- MediaPipe 네이티브 라이브러리가 요구하는 시스템 라이브러리만 `apt`로 설치(§5에서 최소 집합 확정).
- builder의 `/opt/venv`를 복사하고, 앱 코드(`server/`)와 MediaPipe 모델(`assets/models/`)을 복사한다.
- `choborunner_ai` 라이브러리는 builder에서 venv에 이미 설치됐으므로 `src/`를 다시 복사하지 않는다. runtime이 복사하는 소스는 패키지에 포함되지 않는 `server/`뿐이다.
- 비루트 유저로 전환 후 `uvicorn`을 `CMD`로 구동한다.

### 3-4. Dockerfile (설계 예시)

> 본 문서 산출물은 설계까지다. 아래는 설계를 반영한 **예시**이며, 실제 `Dockerfile` 생성은 후속 Phase에서 진행한다.

```dockerfile
# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: builder — Python 의존성·패키지 빌드 (요구사항 #1)
#   이 스테이지에만 pip·빌드 캐시가 존재한다. 최종 이미지엔 포함되지 않는다.
# ============================================================
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 격리 venv — runtime 스테이지로 통째 복사한다
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# pyproject·소스 복사 후 정식 설치 (editable 아님)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install .

# ============================================================
# Stage 2: runtime — 실행 전용 (요구사항 #2: 이미지 실행 배포)
#   pip·컴파일러·빌드 산출물 없음. venv + 앱 코드 + 모델만.
# ============================================================
FROM python:3.11-slim AS runtime

# MediaPipe 네이티브 라이브러리 런타임 의존 (§5 — 최소 집합은 빌드 검증)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 비루트 유저 (§7-3)
RUN useradd --create-home --uid 10001 appuser

# builder가 만든 venv 복사 — 동일 베이스라 바이너리 호환
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 앱 코드(server/) + MediaPipe 모델(§4).
# choborunner_ai 라이브러리는 venv에 이미 설치돼 src/ 재복사 불필요.
COPY --chown=appuser:appuser server/ ./server/
COPY --chown=appuser:appuser assets/models/ ./assets/models/

USER appuser
EXPOSE 8000

# Docker 레벨 healthcheck — curl 미설치, 내장 python으로 /healthz 호출 (§6-2)
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", \
         "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=2).status==200 else 1)"]

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3-5. `.dockerignore` (설계 예시)

빌드 컨텍스트를 줄이고 불필요·민감 파일이 이미지에 섞이는 것을 막는다.

```
# VCS·가상환경·캐시
.git/
.venv/
.pytest_cache/
__pycache__/
*.pyc

# 이미지에 들어가지 않는 디렉터리
tests/
scripts/
legacy/
docs/
.claude/
tmp/
data/
samples/

# 큰 미디어·산출물 — 단 MediaPipe 모델은 빌드에 필요하므로 유지 (§4)
assets/*
!assets/models/

# 문서·로그 (README.md는 pyproject readme라 유지)
CLAUDE.md
CLAUDE.local.md
*.log
```

> `assets/*`로 전체를 제외하되 `!assets/models/`로 모델 디렉터리만 되살린다 — `.dockerignore`와 `.gitignore`는 별개의 파일이며, 모델은 빌드 컨텍스트에 남아야 한다(§4).

---

## 4. MediaPipe 모델 파일 확보 전략

추론 모델 `assets/models/pose_landmarker_lite.task`(약 5.5 MB, lite)는 **git에 커밋되어 레포에 포함**된다(커밋 `46de5ed`). `docker build`는 파일시스템 빌드 컨텍스트에서 `COPY`하므로(§3-4), 개발 머신은 물론 **CI의 클린 클론에서도 모델이 존재**한다 — 별도 다운로드 단계 없이 빌드가 자기완결적으로 동작한다.

> ⚠️ `.gitignore`에 `models/*.task` 규칙이 있으나, 이 패턴은 슬래시가 중간에 있어 **레포 루트에 앵커**되어 `assets/models/...` 경로에는 매칭되지 않는다(`git check-ignore`로 확인 — 미무시). 즉 모델 파일은 의도대로 추적되고 있다. lite 모델은 5.5 MB로 git blob 부담이 없고 사전학습 가중치라 거의 변경되지 않으므로, v1에서는 레포 포함을 유지한다.

**방식.** 빌드 컨텍스트에서 `COPY`(§3-4 Dockerfile 예시) — runtime 스테이지가 `assets/models/`를 복사한다. 모델 경로·이름은 `MediaPipePoseConfig`(`config.py`)가 단일 정답이며, 컨테이너 내 경로는 `CHOBO_MEDIAPIPE_POSE__MODEL_PATH`로 override 가능하다.

---

## 5. OpenCV — headless 전환 (적용 완료)

**문제.** 기존 `pyproject.toml`은 `opencv-python`을 의존성으로 두었다. 이 패키지는 GUI 표시용 시스템 라이브러리(libGL 등)를 전제하지만, 컨테이너는 화면이 없는 **헤드리스 서버**다. 불필요한 라이브러리가 이미지 크기·공격 표면을 키운다.

**적용 (2026-05-18).** `pyproject.toml`의 의존성을 `opencv-python>=4.8.0` → `opencv-python-headless>=4.8.0,<4.12`로 교체했다. 레포 전체에 `cv2.imshow`·`waitKey` 등 GUI 호출이 없고, 유일한 로컬 시각화 스크립트 `scripts/visualize_overlay.py`는 현재 빈 스텁이라 영향이 없다. 향후 이 스크립트를 작성할 때는 화면 출력(`imshow`)이 아니라 파일 저장(`cv2.imwrite`) 방식으로 구현한다.

⚠️ **버전 상한 `<4.12` 이유**: `opencv-python(-headless)` 4.12+ 휠은 `numpy>=2`를 요구한다. 상한 없이 설치하면 numpy가 2.x로 올라가 CLAUDE.md §2의 `numpy<2.0.0`(mediapipe 호환성) 제약을 깬다. 따라서 opencv는 numpy 1.x와 공존하는 마지막 계열인 4.11에 고정한다 — Docker builder 스테이지의 `pip install .`도 이 상한을 그대로 따른다.

⚠️ **검증 항목**: `mediapipe`는 전이 의존성으로 `opencv-contrib-python`(비headless)을 끌어올 수 있고, 네이티브 라이브러리가 `libgl1`·`libglib2.0-0`를 요구한다는 보고가 있다. 따라서 headless 전환을 했더라도 runtime 스테이지의 `apt` 최소 집합(`libgl1`, `libglib2.0-0`)은 **첫 이미지 빌드 시 import·추론 스모크 테스트로 실제 필요 여부를 확정**한다(§9 #7). 불필요로 확인되면 제거한다.

---

## 6. Healthcheck 엔드포인트 설계 (요구사항 #3)

현재 `server/main.py`는 WebSocket 라우트(`/ws/inference`) **하나만** 등록한다. HTTP 라우트가 0개이므로 healthcheck 엔드포인트는 **신규 추가**가 필요하다.

### 6-1. `GET /healthz` — liveness

- **목적**: 프로세스가 살아 있고 이벤트 루프가 응답하는지 확인(liveness).
- **응답**: `200 OK`, body `{"status": "ok"}`.
- **특징**: 의존성 없는 즉답 — 모델 로드·연결 상태를 검사하지 않는다. 따라서 추론 부하와 무관하게 항상 빠르게 응답한다.
- **의료 면책**: healthcheck 응답은 분석 결과가 아니므로 `reference_feedback_only` 플래그 대상이 **아니다**(CLAUDE.md §1의 면책 의무는 분석 JSON 응답에 한정). 운영 응답과 분석 응답을 혼동하지 않는다.

```python
# server/routes/health.py  (신규 — 후속 Phase 구현)
"""Healthcheck 라우트 — Docker HEALTHCHECK / 오케스트레이터용.

docs/docker-container §6. 분석 응답이 아니므로 reference_feedback_only 대상 아님.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe — 프로세스·이벤트 루프 응답 확인."""
    return {"status": "ok"}
```

```python
# server/main.py  — create_app() 안에 라우터 등록 추가
from server.routes import health
...
app.include_router(health.router)
app.add_api_websocket_route(cfg.websocket.endpoint_path, ws_inference)
```

### 6-2. Docker `HEALTHCHECK` 지시어

Dockerfile의 `HEALTHCHECK`(§3-4)가 컨테이너 안에서 `/healthz`를 주기적으로 호출한다.

| 파라미터 | 값 | 의미 |
| --- | --- | --- |
| `--interval` | `30s` | 검사 주기 |
| `--timeout` | `3s` | 응답 제한 시간 |
| `--start-period` | `15s` | 기동 유예 — mediapipe import 등 초기화 시간 |
| `--retries` | `3` | 연속 실패 3회 시 `unhealthy` 전환 |

호출은 `curl` 대신 **이미지에 이미 있는 Python 인터프리터**(`urllib`)로 한다 — `curl`을 `apt`로 추가 설치하지 않아 이미지가 더 가볍다.

### 6-3. (선택) `GET /readyz` — readiness

쿠버네티스 등 오케스트레이터를 도입하면 liveness와 별개로 readiness probe가 유용하다. AI 서버는 모델을 **연결 시점에** 로드하므로(StreamSession이 연결당 `Pipeline` 생성), readiness는 "모델 파일이 존재·접근 가능한가"로 정의할 수 있다(`cfg.mediapipe_pose.model_path` 존재 확인). **v1 범위 밖**이며 오케스트레이터 도입 시 추가한다.

### 6-4. 구현 위치·선행 작업

- 신규 파일 `server/routes/health.py`(위 §6-1) — 기존 `routes/` 구조와 일관.
- `server/main.py`의 `create_app()`에 `app.include_router(health.router)` 추가.
- 본 문서 산출물은 설계까지이며, 위 코드는 **후속 Phase**에서 구현한다(테스트 동반 — CLAUDE.md §9).

---

## 7. 런타임 구성

### 7-1. 포트·엔드포인트

| 항목 | 값 |
| --- | --- |
| 컨테이너 내부 포트 | `8000` (`EXPOSE 8000`, uvicorn `--port 8000`) |
| 추론 (Spring → AI) | `ws://<ai-host>:<published-port>/ws/inference` |
| healthcheck | `http://<ai-host>:<published-port>/healthz` |
| 호스트 게시 포트 | `docker run -p <host>:8000` 으로 결정 (배포 환경 정책) |

`uvicorn[standard]`는 `websockets`를 포함하므로 WebSocket 구동에 추가 설정이 필요 없다. Spring↔AI 사이에 리버스 프록시를 둘 경우 WebSocket Upgrade 통과와 idle timeout(추론 세션이 수십 초 지속)을 확인한다.

### 7-2. 환경변수 — 설정 override

`AppConfig`는 `CHOBO_` prefix·중첩 구분자 `__`로 환경변수 override를 지원한다(`config.py`). 이미지를 재빌드하지 않고 배포 환경별로 동작을 바꿀 수 있다.

| 환경변수 | 대상 | 비고 |
| --- | --- | --- |
| `CHOBO_WEBSOCKET__ENDPOINT_PATH` | WS 엔드포인트 경로 | 기본 `/ws/inference` (docs/2-4-2 §9 #2) |
| `CHOBO_WEBSOCKET__NO_FRAME_TIMEOUT_SEC` | 무수신 타임아웃 | 기본 `3.0` (docs/2-4-2 §9 #4) |
| `CHOBO_WEBSOCKET__PROGRESS_INTERVAL_FRAMES` | progress 송신 간격 | 기본 `15` (⚠️ heuristic) |
| `CHOBO_MEDIAPIPE_POSE__MODEL_PATH` | 모델 파일 경로 | 기본 `assets/models/pose_landmarker_lite.task` |

비밀값은 없다 — AI↔Spring 구간 인증은 docs/2-4-2 §9 범위 밖. `.env` 파일은 `.gitignore`·`.dockerignore` 양쪽에서 제외되며 이미지에 포함하지 않는다.

### 7-3. 비루트 유저·보안

- 전용 유저 `appuser`(uid 10001)로 실행 — root 실행 금지. `COPY --chown`으로 앱 파일 소유권을 부여한다.
- runtime 이미지에는 pip·컴파일러·빌드 도구가 없다(§3-1) — 침해 시 가용 도구 최소화.

### 7-4. 리소스 — CPU 바운드 추론

- MediaPipe pose 추론은 CPU 바운드다(docs/2-4-2 §7-4). 컨테이너 CPU를 과도하게 제한하면 frame 처리량이 떨어지고 백프레셔 drop(docs/2-4-2 §7-5)이 늘어난다.
- "WS 연결 1개 = `Pipeline` 1개"이므로 동시 연결 수에 비례해 메모리·CPU가 증가한다. v1 단일 연결 기준 메모리 가이드 ≈ 1~2 GB(모델·numpy 버퍼 포함) — 첫 이미지로 실측해 확정한다.
- 권장 시작점: `--cpus` 2 이상, `--memory` 2g. 동시 연결을 받을 경우 상향.

### 7-5. `docker run` 예시 (배포 환경)

```bash
docker run -d \
  --name choborunner-ai \
  -p 8000:8000 \
  --restart unless-stopped \
  --cpus 2 --memory 2g \
  -e CHOBO_WEBSOCKET__NO_FRAME_TIMEOUT_SEC=3.0 \
  jaemin1340/capstone2-ai:latest
```

배포 환경은 위 한 줄로 끝난다 — 빌드·소스·Python이 필요 없다(요구사항 #2).

---

## 8. 배포 흐름 — 컨테이너 레지스트리

```
[빌드 환경: CI / 개발 머신]                    [배포 환경: Docker만]
  docker build -t <img>:<tag> .                  docker pull <img>:<tag>
  docker push <img>:<tag>          ── 레지스트리 ──▶  docker run ... <img>:<tag>
```

1. **build** — 빌드 환경에서 `docker build`로 멀티 스테이지 이미지를 만든다.
2. **tag** — 아래 태깅 전략으로 태그를 단다.
3. **push** — 컨테이너 레지스트리 **Docker Hub `jaemin1340/capstone2-ai`**(public)에 push. 빌드·푸시·배포 자동화 흐름은 docs/ci-cd-pipeline.
4. **pull & run** — 배포 환경이 이미지를 pull해 `docker run`(§7-5)으로 실행한다. 배포 환경은 **이미지를 빌드하지 않는다**.

### 이미지 태깅 전략

| 태그 | 용도 |
| --- | --- |
| `<git-short-sha>` | 불변(immutable) 식별자 — 배포 환경은 이 태그로 핀(pin)해 재현성 확보 |
| `v0.1.0` 등 semver | 릴리스 버전 — `pyproject.toml` `version`과 정합 |
| `latest` | 최신 빌드 편의 태그 — **배포 핀 용도로는 사용 금지**(가변) |

> v1 배포 자동화(docs/ci-cd-pipeline §3-3)는 EC2가 편의상 `latest`를 pull하되, 같은 빌드에 `<git-short-sha>` 태그를 함께 push해 "지금 EC2에 뜬 게 어느 커밋인가"의 추적성을 확보한다 — `latest` 단독 핀의 위험을 SHA 태그로 보완하는 절충이다.

### 빌드 환경과 실행 환경의 분리 (요구사항 #2 핵심)

- **빌드 환경**: 소스 트리·`docker build`·레지스트리 인증 필요.
- **실행 환경**: Docker 데몬과 레지스트리 pull 권한만 필요. 소스·Python·pip·빌드 도구 불필요.
- 이 분리로 배포 환경은 "이미지 파일을 실행"하는 역할에만 한정된다.

---

## 9. 결정·미해결 항목

| # | 항목 | 결정 / 상태 |
| --- | --- | --- |
| 1 | 베이스 이미지 | **결정** — `python:3.11-slim` (builder·runtime 공통) |
| 2 | 멀티 스테이지 분리 | **결정** — builder(Python 빌드) / runtime(실행) 분리 (요구사항 #1) |
| 3 | 배포 방식 | **결정** — 컨테이너 레지스트리 push/pull, 배포 환경은 run만 (요구사항 #2) |
| 4 | healthcheck | **결정(설계)** — `GET /healthz` liveness + Docker `HEALTHCHECK` (요구사항 #3). 코드 구현은 후속 Phase |
| 5 | OpenCV headless 전환 | **완료** — `pyproject.toml` 의존성을 `opencv-python-headless`로 교체 (2026-05-18). 레포에 GUI 호출 없음 (§5) |
| 6 | MediaPipe 모델 확보 | **결정** — 모델을 git에 커밋(레포 포함), 빌드 컨텍스트에서 `COPY`. CI 클린 클론에서도 자기완결 동작 (§4) |
| 7 | runtime `apt` 최소 라이브러리 집합 | **검증 필요** — `libgl1`·`libglib2.0-0` 잠정, 첫 빌드 스모크 테스트로 확정 (§5) |
| 8 | 컨테이너 레지스트리·조직명 | **결정** — Docker Hub `jaemin1340/capstone2-ai` (public). 배포 자동화는 docs/ci-cd-pipeline |
| 9 | readiness probe `/readyz` | **v1 범위 밖** — 오케스트레이터 도입 시 추가 (§6-3) |
| 10 | 리소스 한도(CPU/메모리) 실측값 | **미정** — 첫 이미지로 단일 연결 기준 실측 후 확정 (§7-4) |

### 컨테이너화 선행 코드 변경 (후속 Phase)

본 문서는 설계까지다. 아래는 구현 시 필요한 변경으로, 별도 Phase·테스트 동반(CLAUDE.md §9)으로 진행한다.

- `server/routes/health.py` 신규 + `server/main.py` 라우터 등록 (§6).
- ✅ `pyproject.toml` `opencv-python` → `opencv-python-headless` 교체 완료 (2026-05-18, §5).
- `Dockerfile` + `.dockerignore` 생성 (§3 예시 기반).

---

## 10. 변경 이력

- 2026-05-18 v1: 초안. AI 서버 Docker 컨테이너 멀티 스테이지 빌드(builder/runtime 분리)·컨테이너 레지스트리 배포·healthcheck 엔드포인트(`GET /healthz`) 설계. OpenCV headless 전환·MediaPipe 모델 확보 전략을 선행 조정 항목으로 명시. 백엔드(재민) 작성, docs/2-4-2와 동일한 인프라/인터페이스 문서 위치.
- 2026-05-18 v2: §5 OpenCV headless 전환 적용 완료 — `pyproject.toml` 의존성을 `opencv-python-headless`로 교체. 레포에 GUI 호출이 없어 영향 없음 확인. §9 #5 완료 처리.
- 2026-05-19 v3: §4 전제 정정 — 모델 파일은 `.gitignore`의 `models/*.task`(루트 앵커 패턴)에 매칭되지 않아 실제로는 git에 커밋·추적 중. "CI 빌드 실패" 전제가 사실과 달라 옵션 B(builder fetch)·C(git LFS) 및 `fetch_pose_model.py` 선행 구현 요구 삭제. §9 #6을 **결정**으로, §3 `.dockerignore` 주석의 `.gitignore` 서술 정정.
- 2026-05-19 v4: 레지스트리 확정 반영 — docs/ci-cd-pipeline 신설에 따라 §9 #8을 Docker Hub `jaemin1340/capstone2-ai`(public)로 **결정**. §7-5 `docker run` 예시·§8 push 단계의 `ghcr.io/<org>` 자리표시자를 실제 레포명으로 교체. §1 "정의하지 않는 것"의 CI 항목을 docs/ci-cd-pipeline 참조로 갱신. §8 태깅 전략에 v1 배포 흐름(`latest` pull + `<git-short-sha>` 추적) 주석 추가.
