# -*- coding: utf-8 -*-
"""Phase WS-B-2 integration test — WS /ws/inference end-to-end (docs/2-4-2 §3/§7).

FastAPI TestClient로 WebSocket 연결을 맺어 docs/2-4-2 §3 메시지 흐름을 검증한다.
server/ WebSocket 레이어 전 구간(route → StreamSession → StreamPipeline)의
실질 통합 테스트 — 하위 레이어를 별도 중복 검증하지 않는다.

⚠️ StreamSession이 PoseExtractor를 생성하므로 model 파일(.task)이 필요하다.
"""
from __future__ import annotations

import struct

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from choborunner_ai.config import AppConfig

pytestmark = pytest.mark.skipif(
    not AppConfig().mediapipe_pose.model_path.is_file(),
    reason="pose model(.task) 부재",
)

_WS_PATH = "/ws/inference"


def _binary_frame(ts_ms: int, jpeg: bytes) -> bytes:
    """docs/2-4-2 §4 wire format — [8B BE int64 ts_ms][JPEG]."""
    return struct.pack(">q", ts_ms) + jpeg


def _jpeg() -> bytes:
    """단색 BGR frame의 유효 JPEG (blank → pose 미검출)."""
    ok, buf = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _make_app():
    from server.main import create_app

    return create_app()


def test_ws_frames_then_stop():
    """binary frame 연속 → frame_inference(인덱스 증가), stop → analysis_result."""
    with TestClient(_make_app()) as client:
        with client.websocket_connect(_WS_PATH) as ws:
            indices = []
            for i in range(3):
                ws.send_bytes(_binary_frame(1000 + i * 33, _jpeg()))
                fi = ws.receive_json()
                assert fi["type"] == "frame_inference"
                indices.append(fi["frame_index"])
            assert indices == [0, 1, 2]

            ws.send_text('{"type": "stop"}')
            result = ws.receive_json()
    assert result["type"] == "analysis_result"
    assert result["status"] == "failed"  # blank frame만 → pose 0


def test_ws_error_paths():
    """잘못된 binary frame → error 메시지. 세션은 유지 (docs/2-4-2 §6)."""
    with TestClient(_make_app()) as client:
        with client.websocket_connect(_WS_PATH) as ws:
            # 8B 헤더 미만
            ws.send_bytes(b"\x00\x01\x02")
            e1 = ws.receive_json()
            assert e1["type"] == "error"
            assert e1["error_code"] == "malformed_binary_frame"
            # 헤더 + 비-JPEG payload
            ws.send_bytes(_binary_frame(1000, b"not a jpeg"))
            e2 = ws.receive_json()
            assert e2["type"] == "error"
            assert e2["error_code"] == "failed_to_decode_binary_image"
            # error 후에도 stop으로 정상 종료
            ws.send_text('{"type": "stop"}')
            assert ws.receive_json()["type"] == "analysis_result"


def test_ws_no_frame_timeout(monkeypatch):
    """무수신 타임아웃 — frame 없이 대기 → analysis_result 후 종료 (docs/2-4-2 §7-6 c)."""
    monkeypatch.setenv("CHOBO_WEBSOCKET__NO_FRAME_TIMEOUT_SEC", "1.0")
    with TestClient(_make_app()) as client:
        with client.websocket_connect(_WS_PATH) as ws:
            result = ws.receive_json()
    assert result["type"] == "analysis_result"
    assert result["status"] == "failed"


def test_ws_progress_at_interval(monkeypatch):
    """progress_interval_frames 도달 시 analysis_progress 송신 (docs/2-4-2 §9 #3)."""
    monkeypatch.setenv("CHOBO_WEBSOCKET__PROGRESS_INTERVAL_FRAMES", "2")
    with TestClient(_make_app()) as client:
        with client.websocket_connect(_WS_PATH) as ws:
            ws.send_bytes(_binary_frame(1000, _jpeg()))
            assert ws.receive_json()["type"] == "frame_inference"
            ws.send_bytes(_binary_frame(1033, _jpeg()))
            assert ws.receive_json()["type"] == "frame_inference"
            progress = ws.receive_json()
    assert progress["type"] == "analysis_progress"
    assert progress["stage"] == "warming_up"
