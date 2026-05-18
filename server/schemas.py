"""server 수신 제어 메시지 Pydantic schema (docs/2-4-2 §5-1).

CLAUDE.md §3 — server/는 FastAPI 얇은 레이어. 본 모듈은 WebSocket으로 수신하는
제어 메시지(text frame)의 검증 schema만 둔다.

송신 응답 4종(frame_inference / analysis_progress / analysis_result / error)은
``choborunner_ai.result_serializer``가 단일 정답이며, 본 모듈에서 재정의하지
않는다 (docs/2-4-2 §5 — 중복 정의 금지).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class StopMessage(BaseModel):
    """docs/2-4-2 §5-1 — 분석 종료 신호 text frame.

    wire: ``{"type": "stop"}``. 수신 시 AI 서버는 현재까지 누적분으로
    ``analysis_result``를 1회 조립·송신한 뒤 세션을 정리한다.

    v1에서 확정된 수신 제어 메시지는 ``stop`` 1종뿐이다 (docs/2-4-2 §5-1).
    """

    type: Literal["stop"]
