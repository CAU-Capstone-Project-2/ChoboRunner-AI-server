# -*- coding: utf-8 -*-
"""Phase WS-A-2 integration test — StreamPipeline 증분 진입점.

docs/2-4-2 §7-3 — WebSocket 스트림 모드의 frame 증분 누적 진입점.

⚠️ scope:
- A. blank frame 시나리오 — push_frame/snapshot_progress/finalize 기본 동작
  (model 파일 존재 전제, MediaPipe 실호출. blank frame → pose 미검출 경로)
- B. jaemin.mp4 배치-스트림 등가성 — run_on_video_file과 동일 결과 회귀 보호
  (영상 fixture 부재 시 skip)

StreamPipeline은 PoseExtractor를 생성하므로 cfg.mediapipe_pose.model_path
(.task) 파일이 필요하다. AppConfig 기본값 assets/models/pose_landmarker_lite.task.
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
MAX_FRAMES = 100  # 빠른 회귀 (test_pipeline_e2e와 동일 단위)

_MODEL_PATH = AppConfig().mediapipe_pose.model_path
_model_missing = pytest.mark.skipif(
    not _MODEL_PATH.is_file(),
    reason=f"pose model 부재: {_MODEL_PATH}",
)


def _blank_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """단색 BGR frame — MediaPipe는 pose를 검출하지 못한다 (미검출 경로 검증용)."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# ============================================================
# A. blank frame 시나리오 (pose 미검출 경로)
# ============================================================


@_model_missing
def test_push_frame_returns_frame_inference_message():
    """push_frame → FrameInferenceMessage. blank frame이라 pose_detected=False."""
    with StreamPipeline(AppConfig()) as sp:
        msg = sp.push_frame(_blank_frame(), timestamp_ms=0)

    assert isinstance(msg, FrameInferenceMessage)
    assert msg.type == "frame_inference"
    assert msg.frame_index == 0
    assert msg.timestamp_sec == 0.0
    assert msg.result.pose_detected is False


@_model_missing
def test_push_frame_index_and_timestamp_increment():
    """frame_index는 0부터 증가, timestamp_sec는 입력 ts_ms/1000."""
    with StreamPipeline(AppConfig()) as sp:
        m0 = sp.push_frame(_blank_frame(), timestamp_ms=0)
        m1 = sp.push_frame(_blank_frame(), timestamp_ms=33)
        m2 = sp.push_frame(_blank_frame(), timestamp_ms=66)

    assert [m0.frame_index, m1.frame_index, m2.frame_index] == [0, 1, 2]
    assert m1.timestamp_sec == pytest.approx(0.033)
    assert m2.timestamp_sec == pytest.approx(0.066)


@_model_missing
def test_push_frame_duplicate_timestamp_no_crash():
    """비감소 ts(동일 ts 연속) — MediaPipe 단조성 가드 +1ms bump (docs/2-4-2 §8-3).

    응답 timestamp_sec는 bump 영향 없이 원본 ts_ms 유지.
    """
    with StreamPipeline(AppConfig()) as sp:
        m0 = sp.push_frame(_blank_frame(), timestamp_ms=100)
        m1 = sp.push_frame(_blank_frame(), timestamp_ms=100)
        m2 = sp.push_frame(_blank_frame(), timestamp_ms=100)

    assert [m0.frame_index, m1.frame_index, m2.frame_index] == [0, 1, 2]
    assert m0.timestamp_sec == m1.timestamp_sec == m2.timestamp_sec
    assert m2.timestamp_sec == pytest.approx(0.1)


@_model_missing
def test_push_frame_quality_flags_passthrough():
    """server 전처리 frame_quality_flags가 응답 result로 그대로 전달."""
    with StreamPipeline(AppConfig()) as sp:
        msg = sp.push_frame(
            _blank_frame(),
            timestamp_ms=0,
            frame_quality_flags=["low_brightness", "timestamp_fallback"],
        )

    assert msg.result.frame_quality_flags == ["low_brightness", "timestamp_fallback"]


@_model_missing
def test_snapshot_progress_warming_up():
    """IC 미검출(blank frame) → stage warming_up, valid_stride_count 0."""
    with StreamPipeline(AppConfig()) as sp:
        for i in range(5):
            sp.push_frame(_blank_frame(), timestamp_ms=i * 33)
        prog = sp.snapshot_progress()

    assert isinstance(prog, AnalysisProgressMessage)
    assert prog.type == "analysis_progress"
    assert prog.stage == "warming_up"
    assert prog.valid_stride_count == 0
    assert prog.elapsed_sec == pytest.approx(4 * 33 / 1000.0)


@_model_missing
def test_finalize_returns_analysis_result_message():
    """blank frame만 push → pose 0 → finalize는 status=failed AnalysisResultMessage."""
    with StreamPipeline(AppConfig()) as sp:
        for i in range(10):
            sp.push_frame(_blank_frame(), timestamp_ms=i * 33)
        ar = sp.finalize()

    assert isinstance(ar, AnalysisResultMessage)
    assert ar.type == "analysis_result"
    assert ar.status == "failed"
    # JSON 직렬화 round-trip
    back = AnalysisResultMessage.model_validate_json(ar.model_dump_json())
    assert back.status == ar.status


@_model_missing
def test_finalize_zero_frames():
    """frame 0장 finalize — 합성 VideoMeta·empty 누적 — 예외 없이 failed."""
    with StreamPipeline(AppConfig()) as sp:
        ar = sp.finalize()

    assert isinstance(ar, AnalysisResultMessage)
    assert ar.status == "failed"


# ============================================================
# B. jaemin.mp4 배치-스트림 등가성 (영상 fixture 부재 시 skip)
# ============================================================


@pytest.mark.skipif(not VIDEO_PATH.exists(), reason="jaemin.mp4 not present")
def test_stream_matches_batch_jaemin():
    """동일 영상 frame을 배치/스트림에 각각 흘려 동일 결과를 산출하는지 검증.

    StreamPipeline은 run_on_video_file의 누적 로직(_accumulate_one_frame +
    _accumulate)을 그대로 공유하므로, 같은 frame·timestamp 입력에는 동일한
    분석 결과가 나와야 한다 (Phase WS-A-1/WS-A-2 추출의 등가성 회귀 보호).
    """
    cfg = AppConfig()
    meta = get_video_meta(VIDEO_PATH)
    fps_safe = meta.fps if meta.fps > 1e-6 else 30.0

    # 배치 모드
    with Pipeline(cfg) as batch:
        batch_result = batch.run_on_video_file(
            video_path=VIDEO_PATH,
            analysis_side="left",
            max_frames=MAX_FRAMES,
            direction="left_to_right",
        )
    batch_ar = batch_result.analysis_result
    assert batch_ar is not None

    # 스트림 모드 — 배치와 동일 timestamp 부여
    with StreamPipeline(cfg, analysis_side="left", direction="left_to_right") as sp:
        for idx, frame in enumerate(iter_frames(VIDEO_PATH)):
            if idx >= MAX_FRAMES:
                break
            sp.push_frame(frame, timestamp_ms=int(idx * 1000.0 / fps_safe))
        stream_ar = sp.finalize()

    # 안정 필드 등가성 (test_pipeline_e2e 회귀 기준과 동일 축)
    assert stream_ar.status == batch_ar.status
    assert (stream_ar.metrics is None) == (batch_ar.metrics is None)
    if batch_ar.metrics is not None and stream_ar.metrics is not None:
        assert (
            stream_ar.metrics.foot_strike_pattern
            == batch_ar.metrics.foot_strike_pattern
        )
        assert stream_ar.metrics.initial_knee_flexion_deg == pytest.approx(
            batch_ar.metrics.initial_knee_flexion_deg, abs=0.5
        )
    assert stream_ar.reason_codes == batch_ar.reason_codes
