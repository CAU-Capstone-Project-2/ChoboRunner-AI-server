# -*- coding: utf-8 -*-
"""Phase WS-A-2 test — StreamPipeline 증분 진입점 (docs/2-4-2 §7-3).

⚠️ MediaPipe 로드 비용을 고려해 케이스를 최소화한다:
- A. blank frame 1건 — push_frame/snapshot_progress/finalize 핵심 동작 묶음
- B. jaemin.mp4 배치-스트림 등가성 — WS-A-1/A-2 추출의 등가성 회귀 보호
  (영상 fixture 부재 시 skip)

StreamPipeline의 server 레이어 통합은 test_server_ws.py의 WS e2e가 커버한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from choborunner_ai.config import AppConfig
from choborunner_ai.pipeline import Pipeline
from choborunner_ai.result_serializer import (
    AnalysisProgressMessage,
    AnalysisResultMessage,
    FeedbackMessage,
    FrameInferenceMessage,
)
from choborunner_ai.stream_pipeline import StreamPipeline
from choborunner_ai.video_preprocessor import get_video_meta, iter_frames

VIDEO_PATH = Path("legacy/demo_02/jaemin.mp4")
MAX_FRAMES = 100

_MODEL_PATH = AppConfig().mediapipe_pose.model_path
_model_missing = pytest.mark.skipif(
    not _MODEL_PATH.is_file(), reason=f"pose model 부재: {_MODEL_PATH}"
)


def _blank_frame() -> np.ndarray:
    """단색 BGR frame — MediaPipe는 pose를 검출하지 못한다."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@_model_missing
def test_push_snapshot_finalize_mechanics():
    """blank frame 흐름 — push_frame/snapshot_progress/finalize 핵심 동작 묶음.

    - frame_index 0부터 증가, timestamp_sec는 원본 ts_ms/1000
    - 동일 ts 연속 입력도 MediaPipe 단조성 가드(+1ms bump)로 무탈 (docs/2-4-2 §8-3)
    - quality_flags passthrough
    - IC 미검출(blank) → snapshot stage warming_up
    - finalize → AnalysisResultMessage (pose 0 → failed)
    """
    with StreamPipeline(AppConfig()) as sp:
        m0 = sp.push_frame(_blank_frame(), timestamp_ms=0)
        m1 = sp.push_frame(_blank_frame(), timestamp_ms=33, frame_quality_flags=["low_brightness"])
        m2 = sp.push_frame(_blank_frame(), timestamp_ms=33)  # 동일 ts — 단조성 가드
        prog = sp.snapshot_progress()
        ar = sp.finalize()

    assert isinstance(m0, FrameInferenceMessage)
    assert [m0.frame_index, m1.frame_index, m2.frame_index] == [0, 1, 2]
    assert m1.timestamp_sec == pytest.approx(0.033)
    assert m0.result.pose_detected is False
    assert m1.result.frame_quality_flags == ["low_brightness"]

    assert isinstance(prog, AnalysisProgressMessage)
    assert prog.stage == "warming_up"
    assert prog.valid_stride_count == 0
    # docs/2-3-7 §4 — message는 항상 채움, feedback_messages는 analyzing 단계만
    assert prog.message == "분석 준비 중입니다."
    assert prog.feedback_messages is None

    assert isinstance(ar, AnalysisResultMessage)
    assert ar.status == "failed"


# ============================================================
# 실시간 피드백 빈도 dedup 단위 테스트 (docs/2-3-6 §3-2)
# ============================================================


def _make_msg(display_text: str, *, category: str = "posture_warning") -> FeedbackMessage:
    """테스트용 FeedbackMessage factory."""
    return FeedbackMessage(
        category=category,  # type: ignore[arg-type]
        metric="trunk_lean",
        tts_text=None,
        display_text=display_text,
        priority=2,
        tts_enabled=False,
        confidence_prefix=False,
    )


@_model_missing
def test_frequency_dedup_rules():
    """3 빈도 정책을 단일 인스턴스로 검증 (docs/2-3-6 §3-2). MediaPipe 로드 1회.

    - same-message: 동일 display_text 5초 내 재송신 skip
    - cool-down: 직전 cycle 후 2초 미만 시 모든 candidates skip
    - positive (GOOD_PACE): 30초 임계로 same-message 임계 override
    """
    from choborunner_ai.feedback_engine import GOOD_PACE_MESSAGE

    with StreamPipeline(AppConfig()) as sp:
        sp._first_ts_ms = 0
        msg_trunk = _make_msg("trunk warning")
        msg_other = _make_msg("other warning")
        msg_positive = _make_msg(GOOD_PACE_MESSAGE[1], category="posture_info")

        # @0s — 첫 송신 (cool-down·same 모두 통과)
        sp._last_ts_ms = 0
        assert len(sp._filter_by_frequency([msg_trunk])) == 1

        # @1s — cool-down 2초 미만 → 어떤 candidate든 skip
        sp._last_ts_ms = 1000
        assert sp._filter_by_frequency([msg_other]) == []

        # @3s — cool-down 통과, but same-message는 5초 임계 → trunk skip / other emit
        sp._last_ts_ms = 3000
        kept = sp._filter_by_frequency([msg_trunk, msg_other])
        assert [m.display_text for m in kept] == ["other warning"]

        # @9s — trunk 9초 경과(>5s) → emit
        sp._last_ts_ms = 9000
        assert len(sp._filter_by_frequency([msg_trunk])) == 1

        # @15s — positive 첫 송신 (cool-down·same 모두 통과)
        sp._last_ts_ms = 15_000
        assert len(sp._filter_by_frequency([msg_positive])) == 1

        # @30s — positive 15초 경과(30초 임계 미만) → skip
        sp._last_ts_ms = 30_000
        assert sp._filter_by_frequency([msg_positive]) == []

        # @50s — positive 35초 경과(30초 임계 초과) → emit
        sp._last_ts_ms = 50_000
        assert len(sp._filter_by_frequency([msg_positive])) == 1


def test_progress_message_set_for_all_stages():
    """stage가 무엇이든 ``message`` 필드는 채워져야 한다 (docs/2-3-7 §4).

    pure — MediaPipe 불필요 (모듈-level 매핑만 검증).
    """
    # blank frame 경로는 warming_up — test_push_snapshot_finalize_mechanics에서 검증됨.
    from choborunner_ai.stream_pipeline import _PROGRESS_MESSAGE_BY_STAGE

    assert set(_PROGRESS_MESSAGE_BY_STAGE.keys()) == {
        "warming_up",
        "collecting_strides",
        "analyzing",
    }
    for text in _PROGRESS_MESSAGE_BY_STAGE.values():
        assert isinstance(text, str) and text.strip()


def test_analyzing_stage_threshold_default_matches_docs():
    """analyzing 단계 진입 = 유효 stride 3개 (docs/2-3-7 §4-5, docs/2-3-6 §3-1).

    pure — config default 값만 검증. 값이 잘못 바뀌면 본 회귀 테스트가 잡아낸다.
    """
    assert AppConfig().feedback_frequency.min_valid_strides_for_analyzing == 3


@pytest.mark.skipif(not VIDEO_PATH.exists(), reason="jaemin.mp4 not present")
def test_stream_matches_batch_jaemin():
    """동일 영상 frame을 배치/스트림에 흘려 동일 결과를 산출하는지 검증.

    StreamPipeline은 run_on_video_file의 누적 로직(_accumulate_one_frame +
    _accumulate)을 공유하므로, 같은 frame·timestamp 입력에 동일 결과여야 한다
    (Phase WS-A-1/WS-A-2 추출의 등가성 회귀 보호).
    """
    cfg = AppConfig()
    meta = get_video_meta(VIDEO_PATH)
    fps_safe = meta.fps if meta.fps > 1e-6 else 30.0

    with Pipeline(cfg) as batch:
        batch_ar = batch.run_on_video_file(
            VIDEO_PATH, "left", max_frames=MAX_FRAMES, direction="left_to_right"
        ).analysis_result
    assert batch_ar is not None

    with StreamPipeline(cfg, "left", "left_to_right") as sp:
        for idx, frame in enumerate(iter_frames(VIDEO_PATH)):
            if idx >= MAX_FRAMES:
                break
            sp.push_frame(frame, timestamp_ms=int(idx * 1000.0 / fps_safe))
        stream_ar = sp.finalize()

    assert stream_ar.status == batch_ar.status
    assert stream_ar.reason_codes == batch_ar.reason_codes
    if batch_ar.metrics is not None and stream_ar.metrics is not None:
        assert (
            stream_ar.metrics.foot_strike_pattern
            == batch_ar.metrics.foot_strike_pattern
        )
        assert stream_ar.metrics.initial_knee_flexion_deg == pytest.approx(
            batch_ar.metrics.initial_knee_flexion_deg, abs=0.5
        )
