"""ChoboRunner AI Server — healthcheck 라우트 (docs/docker-container §6).

``GET /healthz`` — liveness probe. Docker ``HEALTHCHECK`` 지시어 및
오케스트레이터가 호출해 프로세스·이벤트 루프가 응답하는지 확인한다. WebSocket
추론 엔드포인트(``/ws/inference``)와 완전히 분리된 운영용 엔드포인트다.

CLAUDE.md §6 / docs/2-4-2 §7-1 정책: 본 모듈은 I/O만 담당한다 — 분석 로직 0줄.

⚠️ healthcheck 응답은 분석 결과가 아니므로 의료 면책 플래그
(``reference_feedback_only``)를 붙이지 않는다 (docs/docker-container §6-1).
운영 응답과 분석 응답을 혼동하지 않는다.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    """Liveness probe — 프로세스·이벤트 루프 응답 확인.

    모델 로드·연결 상태 등 의존성을 검사하지 않는 즉답 엔드포인트다. 추론
    부하와 무관하게 항상 빠르게 응답하도록 의도적으로 단순하게 유지한다
    (docs/docker-container §6-1). 모델 파일 존재까지 확인하는 readiness probe
    (``/readyz``)는 v1 범위 밖 — 오케스트레이터 도입 시 추가한다(§6-3).

    Returns:
        ``{"status": "ok"}`` — HTTP 200.
    """
    return {"status": "ok"}
