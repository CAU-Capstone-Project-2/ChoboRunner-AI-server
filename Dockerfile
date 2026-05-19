# syntax=docker/dockerfile:1
# ChoboRunner AI 서버 컨테이너 — docs/docker-container.md §3 설계 구현.

# ============================================================
# Stage 1: builder — Python 의존성·패키지 빌드 (docs/docker-container §3-2)
#   pip·빌드 캐시는 이 스테이지에만 존재한다. 최종 이미지엔 포함되지 않는다.
# ============================================================
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 격리 venv — runtime 스테이지로 통째 복사한다
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# pyproject·소스 복사 후 정식 설치 (editable 아님 — 컨테이너엔 소스 마운트 없음)
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install .

# ============================================================
# Stage 2: runtime — 실행 전용 (docs/docker-container §3-3)
#   pip·컴파일러 없음. venv + 앱 코드(server/) + 모델만 담는다.
# ============================================================
FROM python:3.11-slim AS runtime

# MediaPipe 네이티브 라이브러리 런타임 의존 (docs/docker-container §5)
#   libgles2 / libegl1: 최신 MediaPipe C 바인딩(mediapipe_c_bindings)이
#   OpenGL ES 2.0 + EGL에 링크 — 없으면 libGLESv2.so.2 dlopen 실패.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgles2 \
        libegl1 \
    && rm -rf /var/lib/apt/lists/*

# 비루트 유저 (docs/docker-container §7-3)
RUN useradd --create-home --uid 10001 appuser

# builder가 만든 venv 복사 — 동일 베이스라 바이너리 호환
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 앱 코드(server/) + MediaPipe 모델. choborunner_ai 라이브러리는 builder의
# venv에 이미 설치돼 src/ 재복사는 불필요하다 (docs/docker-container §3-3).
COPY --chown=appuser:appuser server/ ./server/
COPY --chown=appuser:appuser assets/models/ ./assets/models/

USER appuser
EXPOSE 8000

# Docker 레벨 healthcheck — curl 미설치, 이미지 내장 python으로 /healthz 호출
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=2).status==200 else 1)"]

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
