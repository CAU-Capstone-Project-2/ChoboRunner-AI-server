"""input_validator.py 통합 테스트 (Phase 5).

설계문서 docs/2-3-1 + Phase 1~4 자가 검토 학습 검증.

카테고리:
A. ValidationStatus / ValidationResult 토대
B. validate_first_frame 분기 + 경계값
C. validate_frame_decodable
D. validate_duration 분기 + 경계값
E. validate_effective_fps 분기 + ZeroDivision 방어
F. validate_frame_count 분기 + 경계값
G. aggregate_results 우선순위 규칙
H. validate_session jaemin.mp4 실측 메타 (hardcode)

영상 파일 의존 X — input_validator는 영상 메타만 다루므로 메타값 hardcode로 충분.
실측 영상 의존성 정책: 멘토 코멘트 정합 — 합성은 수식 검증 한정, 행동 검증은
실측 메타 활용 (hardcode).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from choborunner_ai.config import AppConfig, InputMetadataConfig
from choborunner_ai.input_validator import (
    ValidationResult,
    ValidationStatus,
    aggregate_results,
    validate_duration,
    validate_effective_fps,
    validate_first_frame,
    validate_frame_count,
    validate_frame_decodable,
    validate_session,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def cfg() -> InputMetadataConfig:
    """default 임계값 InputMetadataConfig."""
    return InputMetadataConfig()


@pytest.fixture
def app_cfg() -> AppConfig:
    """default AppConfig (validate_session 용)."""
    return AppConfig()


# ============================================================
# A. ValidationStatus / ValidationResult 토대
# ============================================================


def test_status_ordering():
    assert ValidationStatus.FAILED > ValidationStatus.LOW_CONFIDENCE
    assert ValidationStatus.LOW_CONFIDENCE > ValidationStatus.OK
    assert ValidationStatus.FAILED.value == 2
    assert ValidationStatus.LOW_CONFIDENCE.value == 1
    assert ValidationStatus.OK.value == 0


def test_result_is_frozen():
    r = ValidationResult(status=ValidationStatus.OK)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.status = ValidationStatus.FAILED  # type: ignore[misc]


# ============================================================
# B. validate_first_frame
# ============================================================


def test_first_frame_high_res(cfg):
    r = validate_first_frame(1080, 1920, cfg)
    assert r.status == ValidationStatus.OK


def test_first_frame_low_res_square(cfg):
    r = validate_first_frame(480, 480, cfg)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "low_resolution"
    assert r.details["long_edge_px"] == 480
    assert r.details["threshold"] == 720


def test_first_frame_720_landscape(cfg):
    r = validate_first_frame(720, 1280, cfg)
    assert r.status == ValidationStatus.OK


def test_first_frame_exactly_720(cfg):
    # 긴 변 = 720, 임계 == 720 → ≥ 충족 (< 만 FAILED)
    r = validate_first_frame(480, 720, cfg)
    assert r.status == ValidationStatus.OK


def test_first_frame_just_below_720(cfg):
    r = validate_first_frame(480, 719, cfg)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "low_resolution"


def test_first_frame_symmetric(cfg):
    r1 = validate_first_frame(1920, 1080, cfg)
    r2 = validate_first_frame(1080, 1920, cfg)
    assert r1.status == ValidationStatus.OK
    assert r2.status == ValidationStatus.OK


# ============================================================
# C. validate_frame_decodable
# ============================================================


def test_decodable_none():
    r = validate_frame_decodable(None)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "decode_failed"
    assert r.details["cause"] == "frame_is_none"


def test_decodable_valid_3d():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    r = validate_frame_decodable(frame)
    assert r.status == ValidationStatus.OK


def test_decodable_2d():
    frame = np.zeros((720, 1280), dtype=np.uint8)
    r = validate_frame_decodable(frame)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "decode_failed"
    assert r.details["cause"] == "invalid_shape"


def test_decodable_4_channels():
    frame = np.zeros((720, 1280, 4), dtype=np.uint8)
    r = validate_frame_decodable(frame)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "decode_failed"


# ============================================================
# D. validate_duration
# ============================================================


def test_duration_3s(cfg):
    r = validate_duration(3.0, cfg)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "too_short"


def test_duration_5s_boundary(cfg):
    # 경계: < 5.0이 FAILED, >= 5.0이 LOW_CONF or OK
    r = validate_duration(5.0, cfg)
    assert r.status == ValidationStatus.LOW_CONFIDENCE
    assert r.reason == "short_duration"


def test_duration_just_below_5s(cfg):
    r = validate_duration(4.99, cfg)
    assert r.status == ValidationStatus.FAILED


def test_duration_7s(cfg):
    r = validate_duration(7.0, cfg)
    assert r.status == ValidationStatus.LOW_CONFIDENCE


def test_duration_10s_boundary(cfg):
    r = validate_duration(10.0, cfg)
    assert r.status == ValidationStatus.OK


def test_duration_just_below_10s(cfg):
    r = validate_duration(9.99, cfg)
    assert r.status == ValidationStatus.LOW_CONFIDENCE


def test_duration_15s(cfg):
    r = validate_duration(15.0, cfg)
    assert r.status == ValidationStatus.OK


# ============================================================
# E. validate_effective_fps
# ============================================================


def test_fps_zero_duration(cfg):
    r = validate_effective_fps(100, 0.0, cfg)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "invalid_duration"


def test_fps_negative_duration(cfg):
    r = validate_effective_fps(100, -1.5, cfg)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "invalid_duration"


def test_fps_15(cfg):
    r = validate_effective_fps(150, 10.0, cfg)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "low_fps"


def test_fps_24_boundary(cfg):
    r = validate_effective_fps(240, 10.0, cfg)
    assert r.status == ValidationStatus.LOW_CONFIDENCE
    assert r.reason == "borderline_fps"


def test_fps_just_below_24(cfg):
    r = validate_effective_fps(239, 10.0, cfg)
    assert r.status == ValidationStatus.FAILED


def test_fps_29(cfg):
    r = validate_effective_fps(290, 10.0, cfg)
    assert r.status == ValidationStatus.LOW_CONFIDENCE


def test_fps_30_boundary(cfg):
    r = validate_effective_fps(300, 10.0, cfg)
    assert r.status == ValidationStatus.OK


def test_fps_31(cfg):
    r = validate_effective_fps(310, 10.0, cfg)
    assert r.status == ValidationStatus.OK


# ============================================================
# F. validate_frame_count
# ============================================================


def test_frame_count_80(cfg):
    r = validate_frame_count(80, cfg)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "insufficient_frames"


def test_frame_count_120_boundary(cfg):
    r = validate_frame_count(120, cfg)
    assert r.status == ValidationStatus.LOW_CONFIDENCE
    assert r.reason == "borderline_frames"


def test_frame_count_just_below_120(cfg):
    r = validate_frame_count(119, cfg)
    assert r.status == ValidationStatus.FAILED


def test_frame_count_240_boundary(cfg):
    r = validate_frame_count(240, cfg)
    assert r.status == ValidationStatus.OK


def test_frame_count_just_below_240(cfg):
    r = validate_frame_count(239, cfg)
    assert r.status == ValidationStatus.LOW_CONFIDENCE


def test_frame_count_300(cfg):
    r = validate_frame_count(300, cfg)
    assert r.status == ValidationStatus.OK


# ============================================================
# G. aggregate_results
# ============================================================


def test_aggregate_all_ok():
    results = [
        ("a", ValidationResult(status=ValidationStatus.OK)),
        ("b", ValidationResult(status=ValidationStatus.OK)),
        ("c", ValidationResult(status=ValidationStatus.OK)),
        ("d", ValidationResult(status=ValidationStatus.OK)),
    ]
    r = aggregate_results(results)
    assert r.status == ValidationStatus.OK
    assert r.reason is None
    assert r.details["summary"] == {
        "total": 4,
        "ok": 4,
        "low_confidence": 0,
        "failed": 0,
    }


def test_aggregate_one_low_conf():
    results = [
        ("a", ValidationResult(status=ValidationStatus.OK)),
        ("b", ValidationResult(status=ValidationStatus.LOW_CONFIDENCE, reason="x")),
        ("c", ValidationResult(status=ValidationStatus.OK)),
    ]
    r = aggregate_results(results)
    assert r.status == ValidationStatus.LOW_CONFIDENCE
    assert r.reason == "x"


def test_aggregate_one_failed():
    results = [
        ("a", ValidationResult(status=ValidationStatus.OK)),
        ("b", ValidationResult(status=ValidationStatus.FAILED, reason="y")),
        ("c", ValidationResult(status=ValidationStatus.LOW_CONFIDENCE, reason="z")),
    ]
    r = aggregate_results(results)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "y"


def test_aggregate_multiple_failed_first_reason():
    # 첫 FAILED reason 채택 (named_results 순서로 우선순위 표현)
    results = [
        ("first", ValidationResult(status=ValidationStatus.FAILED, reason="first_reason")),
        ("second", ValidationResult(status=ValidationStatus.FAILED, reason="second_reason")),
    ]
    r = aggregate_results(results)
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "first_reason"


def test_aggregate_empty_raises():
    with pytest.raises(ValueError, match="empty results"):
        aggregate_results([])


# ============================================================
# H. validate_session (jaemin.mp4 실측 메타 hardcode)
# ============================================================
# Phase D 실측 보고 — Vertical Slice 4/4 PASS 확인된 메타:
# - width:       1080
# - height:      1920
# - frame_count: 982
# - fps:         30.01 → duration ≈ 32.72s


def test_session_jaemin_normal(app_cfg):
    r = validate_session(
        first_frame_width=1080,
        first_frame_height=1920,
        received_frames=982,
        duration_sec=32.7,
        cfg=app_cfg,
    )
    assert r.status == ValidationStatus.OK
    assert r.reason is None
    assert r.details["summary"]["ok"] == 4


def test_session_short_video(app_cfg):
    r = validate_session(
        first_frame_width=1080,
        first_frame_height=1920,
        received_frames=90,
        duration_sec=3.0,
        cfg=app_cfg,
    )
    assert r.status == ValidationStatus.FAILED
    assert r.reason == "too_short"


def test_session_multi_failure(app_cfg):
    # 480x640 (low_resolution) + 2.5s (too_short) + 60 frames (insufficient_frames)
    r = validate_session(
        first_frame_width=480,
        first_frame_height=640,
        received_frames=60,
        duration_sec=2.5,
        cfg=app_cfg,
    )
    assert r.status == ValidationStatus.FAILED
    # 첫 FAILED는 first_frame → low_resolution
    assert r.reason == "low_resolution"
    # sub_results에 4개 검증 모두 기록
    assert len(r.details["sub_results"]) == 4
    failed_checks = [
        s["check_name"] for s in r.details["sub_results"] if s["status"] == "FAILED"
    ]
    assert "first_frame" in failed_checks
    assert "duration" in failed_checks
    assert "frame_count" in failed_checks
