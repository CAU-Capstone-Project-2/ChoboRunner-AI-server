"""ChoboRunner AI Server — WebSocket 추론 라우트 (docs/2-4-2 §7).

``WS /ws/inference`` — Spring relay가 binary frame을 보내면 분석 결과를 응답한다.

CLAUDE.md §6 / docs/2-4-2 §7-1 정책: 본 모듈은 **I/O만** 담당한다 — WebSocket
수신/송신, 헤더 파싱·분석 위임(StreamSession), 종료 트리거 처리. 분석 로직은
0줄이다.

동시성 (docs/2-4-2 §7-4/§7-5):
- 수신 task와 처리 task를 분리한다. 수신 task는 최신 frame 1장만 유지하고
  밀린 frame은 덮어써 drop한다(백프레셔 — 실시간 최신성 우선).
- 처리 task는 pose 추론(CPU 바운드)을 ``run_in_executor``로 이벤트 루프 밖에서
  실행한다.

종료 트리거 3종 (docs/2-4-2 §7-6):
- (a) ``{"type":"stop"}`` text frame 수신
- (b) ``WebSocketDisconnect`` — 연결 끊김
- (c) 무수신 타임아웃 — ``cfg.websocket.no_frame_timeout_sec`` (기본 3초)

종료 트리거 무관하게 ``analysis_result`` 1회 송신을 시도하고 ``StreamSession``을
정리한다 (``finally`` 보장).
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from choborunner_ai.config import AppConfig
from choborunner_ai.result_serializer import ErrorMessage
from server.schemas import StopMessage
from server.session import FrameDecodeError, ShortFrameError, StreamSession

logger = logging.getLogger(__name__)


class _RecvState:
    """수신 task ↔ 처리 task 공유 상태 — 1-slot frame holder + 종료 플래그.

    Attributes:
        latest: 처리 대기 중인 최신 binary frame. 새 frame이 오면 덮어쓴다
            (백프레셔 — docs/2-4-2 §7-5).
        terminal: stop 수신 또는 연결 끊김 여부.
        frame_event: 새 frame 도착 또는 종료 신호 시 set — 처리 task를 깨운다.
    """

    def __init__(self) -> None:
        self.latest: bytes | None = None
        self.terminal: bool = False
        self.frame_event = asyncio.Event()


def _is_stop(text: str) -> bool:
    """text frame이 docs/2-4-2 §5-1 stop 메시지인지 검증."""
    try:
        StopMessage.model_validate_json(text)
        return True
    except ValidationError:
        return False


async def _send(websocket: WebSocket, message: BaseModel) -> None:
    """응답 Pydantic 모델 → text frame(JSON) 송신. 송신 실패는 swallow.

    docs/2-4-2 §5-1 — 연결이 끊긴 뒤 송신을 시도할 수 있으므로(예: finalize)
    전송 실패는 무시한다.
    """
    try:
        await websocket.send_text(message.model_dump_json())
    except Exception:
        logger.exception("응답 송신 실패 (swallow)")


async def _receive_loop(websocket: WebSocket, state: _RecvState) -> None:
    """수신 task — binary frame 누적(최신 1장 유지) + 종료 신호 감지."""
    while True:
        try:
            msg = await websocket.receive()
        except (WebSocketDisconnect, RuntimeError):
            state.terminal = True
            state.frame_event.set()
            return
        if msg.get("type") == "websocket.disconnect":
            state.terminal = True
            state.frame_event.set()
            return

        data = msg.get("bytes")
        text = msg.get("text")
        if data is not None:
            # 백프레셔 (docs/2-4-2 §7-5): 미처리 frame은 덮어써 drop — 최신성 우선
            state.latest = data
            state.frame_event.set()
        elif text is not None and _is_stop(text):
            state.terminal = True
            state.frame_event.set()
            return
        # 그 외 text frame은 무시 — v1 수신 제어 메시지는 stop 1종 (docs/2-4-2 §5-1)


async def _process_one(
    websocket: WebSocket,
    session: StreamSession,
    loop: asyncio.AbstractEventLoop,
    raw: bytes,
) -> None:
    """binary frame 1개 — executor 추론 + frame_inference/progress(또는 error) 송신."""
    try:
        frame_msg = await loop.run_in_executor(None, session.on_binary_frame, raw)
    except ShortFrameError as e:
        await _send(
            websocket,
            ErrorMessage(error_code="malformed_binary_frame", error_detail=str(e)),
        )
        return
    except FrameDecodeError as e:
        await _send(
            websocket,
            ErrorMessage(
                error_code="failed_to_decode_binary_image", error_detail=str(e)
            ),
        )
        return
    except Exception as e:  # noqa: BLE001 — 내부 예외는 error 메시지로 표면화
        logger.exception("frame 처리 내부 예외")
        await _send(
            websocket,
            ErrorMessage(error_code="internal_error", error_detail=str(e)),
        )
        return

    await _send(websocket, frame_msg)
    progress = session.maybe_progress()
    if progress is not None:
        await _send(websocket, progress)


async def _process_loop(
    websocket: WebSocket,
    session: StreamSession,
    state: _RecvState,
    timeout_sec: float,
) -> None:
    """처리 task — 최신 frame 추론·응답. 무수신 타임아웃 또는 종료 시 반환."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            await asyncio.wait_for(state.frame_event.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            logger.info("무수신 타임아웃 (%.1fs) — 세션 종료", timeout_sec)
            return
        state.frame_event.clear()

        raw = state.latest
        state.latest = None
        if raw is not None:
            await _process_one(websocket, session, loop, raw)

        # 종료 신호 수신 + 대기 frame 없음 → 처리 완료
        if state.terminal and state.latest is None:
            return


async def ws_inference(websocket: WebSocket) -> None:
    """WS 추론 엔드포인트 핸들러 — docs/2-4-2 §3 메시지 흐름.

    첫 binary frame 수신 즉시 분석을 시작한다(별도 start 메시지 없음,
    docs/2-4-2 §3). stop 수신 / 연결 끊김 / 무수신 타임아웃 시 ``analysis_result``
    1회 송신 후 세션을 정리한다.

    경로는 ``server/main.py``가 ``cfg.websocket.endpoint_path``로 등록한다.
    """
    await websocket.accept()
    cfg: AppConfig = websocket.app.state.cfg
    session = StreamSession(cfg)
    state = _RecvState()

    try:
        recv_task = asyncio.create_task(_receive_loop(websocket, state))
        proc_task = asyncio.create_task(
            _process_loop(
                websocket, session, state, cfg.websocket.no_frame_timeout_sec
            )
        )
        done, _pending = await asyncio.wait(
            {recv_task, proc_task}, return_when=asyncio.FIRST_COMPLETED
        )
        # 처리 task가 먼저 끝났으면(타임아웃 등) 수신 task는 receive()에 묶여
        # 있을 수 있으므로 취소한다. 수신 task가 먼저 끝났으면 처리 task는
        # frame_event/terminal 신호로 곧 자연 종료하므로 그대로 대기한다.
        if proc_task in done and not recv_task.done():
            recv_task.cancel()
        await asyncio.gather(recv_task, proc_task, return_exceptions=True)
    finally:
        # 종료 트리거 무관 — finalize + analysis_result 송신 시도 후 반드시 close.
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, session.finalize)
            await _send(websocket, result)
        except Exception:
            logger.exception("finalize/analysis_result 송신 실패 (swallow)")
        session.close()
        try:
            await websocket.close()
        except Exception:
            pass
