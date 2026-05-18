# -*- coding: utf-8 -*-
"""Phase WS-B test — server 세션 레이어 순수 단위 (docs/2-4-2 §5/§8).

⚠️ 본 파일은 model(.task) 불필요한 순수 케이스만 둔다 — parse_binary_frame
(wire format 파싱), StopMessage(수신 schema), WebSocketConfig(설정 통합).

StreamSession의 on_binary_frame/maybe_progress/finalize는 MediaPipe를 띄우는
end-to-end 동작이며 test_server_ws.py의 WS e2e 테스트가 전 구간을 커버하므로
여기서 중복 검증하지 않는다.
"""
from __future__ import annotations

import struct

import pytest
from pydantic import ValidationError

from choborunner_ai.config import AppConfig, WebSocketConfig
from server.schemas import StopMessage
from server.session import ShortFrameError, parse_binary_frame


def test_parse_binary_frame_valid():
    """정상 frame — ts_ms는 초로 변환, JPEG payload 분리 (docs/2-4-2 §4/§8-1)."""
    capture_ts, payload = parse_binary_frame(struct.pack(">q", 1500) + b"JPEGDATA")
    assert capture_ts == pytest.approx(1.5)
    assert payload == b"JPEGDATA"


def test_parse_binary_frame_ts_zero_or_negative_is_fallback():
    """ts_ms <= 0 → capture_ts None (docs/2-4-2 §8-2 fallback 신호)."""
    assert parse_binary_frame(struct.pack(">q", 0) + b"X")[0] is None
    assert parse_binary_frame(struct.pack(">q", -1) + b"X")[0] is None


def test_parse_binary_frame_too_short_raises():
    """8B 헤더 미만 → ShortFrameError (docs/2-4-2 §8-1 방어적 파싱)."""
    with pytest.raises(ShortFrameError):
        parse_binary_frame(b"\x00\x00\x00")


def test_stop_message_schema():
    """{"type":"stop"}만 통과, 그 외 거부 (docs/2-4-2 §5-1)."""
    assert StopMessage.model_validate_json('{"type": "stop"}').type == "stop"
    with pytest.raises(ValidationError):
        StopMessage.model_validate_json('{"type": "start"}')


def test_websocket_config_defaults():
    """docs/2-4-2 §9 결정값 — 엔드포인트 /ws/inference, 무수신 타임아웃 3초."""
    ws = WebSocketConfig()
    assert ws.endpoint_path == "/ws/inference"
    assert ws.no_frame_timeout_sec == 3.0
    assert isinstance(AppConfig().websocket, WebSocketConfig)
