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

    assert isinstance(ar, AnalysisResultMessage)
    assert ar.status == "failed"


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
