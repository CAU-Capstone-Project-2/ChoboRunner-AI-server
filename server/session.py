"""ChoboRunner AI Server — WebSocket 스트림 세션 (docs/2-4-2 §7-2).

연결 1개 = ``StreamSession`` 1개 = ``StreamPipeline`` 1개. CLAUDE.md §3/§6
정책에 따라 분석 로직은 ``choborunner_ai`` 라이브러리(``StreamPipeline``)에
위임하고, 본 모듈은 wire format 파싱 + 전처리 primitive 조립 + 진행 메시지
빈도 관리만 담당한다 — ``routes/stream.py``(I/O 루프)와 분석 라이브러리 사이의
얇은 어댑터.

⚠️ 프레임 품질 검사(휘도·블러·흔들림, docs/2-3-2 §4)는 v1 stream 경로에
연결하지 않는다 — ``Pipeline.run_on_video_file`` 배치 경로도 동일하게
``video_preprocessor.Preprocessor``를 쓰지 않으므로 두 모드의 동작이 일치한다.
본 모듈은 wire 규약상 필요한 ``decode_jpeg_binary`` + ``resolve_timestamp``
primitive만 사용한다 (docs/2-4-2 §8-1이 지목한 진입점).
"""
from __future__ import annotations

import logging
import struct
import time
from typing import Literal, Optional

from choborunner_ai.config import AppConfig
from choborunner_ai.result_serializer import (
    AnalysisProgressMessage,
    AnalysisResultMessage,
    FrameInferenceMessage,
    FrameQualityFlag,
)
from choborunner_ai.stream_pipeline import StreamPipeline
from choborunner_ai.video_preprocessor import decode_jpeg_binary, resolve_timestamp

logger = logging.getLogger(__name__)

TIMESTAMP_HEADER_BYTES = 8
"""docs/2-4-2 §4-1 — binary frame 선두 8 byte = capture timestamp 헤더."""


class ShortFrameError(ValueError):
    """binary frame이 8 byte 헤더보다 짧음 (docs/2-4-2 §8-1).

    Spring relay가 길이 < 8B frame을 drop하므로 정상 경로에서는 발생하지
    않으나, 방어적 파싱으로 검출한다 (docs/2-4-2 §4-3).
    """


class FrameDecodeError(Exception):
    """헤더 이후 payload의 JPEG decode 실패 (docs/2-4-2 §6).

    헤더 누락한 raw JPEG·손상 JPEG 등. 호출자는 ``error`` 메시지로 응답한다.
    """


def parse_binary_frame(raw: bytes) -> tuple[Optional[float], bytes]:
    """``[8B BE int64 ts_ms][JPEG]`` → ``(capture_ts_sec | None, jpeg_bytes)``.

    docs/2-4-2 §4 wire format / §8-1 파싱.

    Args:
        raw: WebSocket binary frame 전체 (8B 헤더 포함).

    Returns:
        ``(capture_ts, jpeg_bytes)`` — ``capture_ts``는 초 단위. ``ts_ms <= 0``
        (fallback 신호, docs/2-4-2 §8-2)이면 ``capture_ts=None``.

    Raises:
        ShortFrameError: ``raw`` 길이 < 8 byte.
    """
    if len(raw) < TIMESTAMP_HEADER_BYTES:
        raise ShortFrameError(
            f"binary frame too short: {len(raw)}B < {TIMESTAMP_HEADER_BYTES}B header"
        )
    # BE signed int64 — docs/2-4-2 §4-2 (Java/네트워크 표준 byte order)
    ts_ms = struct.unpack(">q", raw[:TIMESTAMP_HEADER_BYTES])[0]
    jpeg_bytes = raw[TIMESTAMP_HEADER_BYTES:]
    capture_ts = ts_ms / 1000.0 if ts_ms > 0 else None  # 0/음수 → fallback
    return capture_ts, jpeg_bytes


class StreamSession:
    """WebSocket 연결 1개의 분석 세션 (docs/2-4-2 §7-2).

    ``routes/stream.py``가 연결 수립 시 1개 생성하고, 종료 시 ``close()``로
    정리한다. 내부에 ``StreamPipeline`` 1개를 소유 — PoseExtractor 누수 방지를
    위해 ``close()`` 호출이 필수다 (호출자가 try/finally로 보장).

    ``analysis_side``·``direction``은 WebSocket으로 전달되지 않으며 v1 기본값
    (``left`` / ``left_to_right``)을 사용한다 (docs/2-4-2 §3).
    """

    def __init__(
        self,
        cfg: AppConfig,
        analysis_side: Literal["left", "right"] = "left",
        direction: Literal["left_to_right", "right_to_left"] = "left_to_right",
    ) -> None:
        """초기화 — StreamPipeline 1개 생성.

        Args:
            cfg: AppConfig (DI). websocket / mediapipe_pose 등 사용.
            analysis_side: 'left' 또는 'right' (docs/2-4-2 §3 v1 기본 'left').
            direction: 'left_to_right' 또는 'right_to_left' (v1 기본).

        Raises:
            FileNotFoundError: PoseExtractor 모델 파일(.task) 부재.
            RuntimeError: PoseExtractor 초기화 실패.
        """
        self._cfg = cfg
        self._stream = StreamPipeline(cfg, analysis_side, direction)
        self._progress_interval = cfg.websocket.progress_interval_frames
        self._frames_since_progress = 0

    def on_binary_frame(self, raw: bytes) -> FrameInferenceMessage:
        """binary frame 1개 처리 → FrameInferenceMessage.

        docs/2-4-2 §8 — 헤더 파싱 → JPEG decode → timestamp 해소 →
        ``StreamPipeline.push_frame``. pose 추론은 CPU 바운드이므로 호출자는
        이벤트 루프 밖(executor)에서 실행한다 (docs/2-4-2 §7-4).

        Args:
            raw: WebSocket binary frame 전체 (8B 헤더 + JPEG).

        Returns:
            FrameInferenceMessage — 매 frame 직후 응답 (docs/2-3-7 §3).

        Raises:
            ShortFrameError: 8B 헤더 미만 (호출자 → ErrorMessage).
            FrameDecodeError: JPEG decode 실패 (호출자 → ErrorMessage).
        """
        capture_ts, jpeg_bytes = parse_binary_frame(raw)
        img = decode_jpeg_binary(jpeg_bytes)
        if img is None:
            raise FrameDecodeError("JPEG decode 실패 (헤더 누락 raw JPEG 또는 손상)")

        # timestamp 해소 — capture_ts None(ts_ms <= 0)이면 서버 수신 시각 fallback.
        # server_recv_ts는 단조 시계(time.monotonic) — capture_ts(Android 단조
        # 시계)와 동일 성격 (docs/2-4-2 §4-2 / §8-2).
        timestamp_sec, is_fallback = resolve_timestamp(capture_ts, time.monotonic())
        quality_flags: list[FrameQualityFlag] = (
            ["timestamp_fallback"] if is_fallback else []
        )
        return self._stream.push_frame(
            img,
            int(timestamp_sec * 1000.0),
            frame_quality_flags=quality_flags,
        )

    def maybe_progress(self) -> Optional[AnalysisProgressMessage]:
        """progress 송신 주기 도달 시 AnalysisProgressMessage, 아니면 None.

        ``on_binary_frame`` 성공마다 1회 호출되는 것을 전제로 frame을 카운트하고,
        ``cfg.websocket.progress_interval_frames``마다 1회 메시지를 산출한다.

        ⚠️ 송신 빈도는 docs/2-4-2 §9 #3 미해결 항목 — interval 기본값은 heuristic
        (config.py ``WebSocketConfig.progress_interval_frames`` 참조).

        Returns:
            주기 도달 시 AnalysisProgressMessage, 아니면 None.
        """
        self._frames_since_progress += 1
        if self._frames_since_progress < self._progress_interval:
            return None
        self._frames_since_progress = 0
        return self._stream.snapshot_progress()

    def finalize(self) -> AnalysisResultMessage:
        """세션 종료 → 누적분으로 AnalysisResultMessage 조립 (docs/2-4-2 §7-6).

        ``StreamPipeline.finalize``에 위임 — 배치 모드와 동일한 누적 평가·
        Phase 5 metrics·응답 조립.
        """
        return self._stream.finalize()

    def close(self) -> None:
        """StreamPipeline.close() — PoseExtractor 누수 방지. 예외 swallow.

        docs/2-4-2 §7-6 — 종료 트리거 무관하게 반드시 호출돼야 한다.
        """
        try:
            self._stream.close()
        except Exception:
            logger.exception("StreamSession.close 예외 (swallow)")
