"""ChoboRunner AI Server — FastAPI 진입점.

CLAUDE.md §3 — ``server/``는 FastAPI 얇은 레이어. 분석 로직은
``choborunner_ai`` 라이브러리에 있다.

docs/2-4-2 §7-4 — ``AppConfig``는 lifespan에서 1회 로드해 ``app.state``에
보관하고 연결마다 공유한다(연결마다 재로드 금지). 연결당 ``Pipeline`` 인스턴스는
``StreamSession``이 생성한다.

실행:
    uvicorn server.main:app --host 0.0.0.0 --port 8000

⚠️ 의료 면책: 본 서버의 모든 분석 결과는 참고용 피드백이며 의료 진단이
아니다 (CLAUDE.md §1).
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from choborunner_ai.config import AppConfig
from server.routes.stream import ws_inference

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """FastAPI app 팩토리 — lifespan + WebSocket 라우트 등록.

    WebSocket 엔드포인트 경로는 ``cfg.websocket.endpoint_path``
    (docs/2-4-2 §9 #2 결정 — ``/ws/inference``)로 등록한다.

    Returns:
        구성 완료된 FastAPI app.
    """
    cfg = AppConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """앱 생명주기 — 시작 시 AppConfig를 app.state에 보관 (docs/2-4-2 §7-4)."""
        app.state.cfg = cfg
        logger.info(
            "ChoboRunner AI Server 시작 — WS endpoint %s, 무수신 타임아웃 %.1fs",
            cfg.websocket.endpoint_path,
            cfg.websocket.no_frame_timeout_sec,
        )
        yield
        logger.info("ChoboRunner AI Server 종료")

    app = FastAPI(
        title="ChoboRunner AI Server",
        description=(
            "초보 러너 러닝 자세 분석 — 참고용 피드백 (의료 진단 아님). "
            "AI ↔ Backend WebSocket 연동: docs/2-4-2."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_api_websocket_route(cfg.websocket.endpoint_path, ws_inference)
    return app


app = create_app()
