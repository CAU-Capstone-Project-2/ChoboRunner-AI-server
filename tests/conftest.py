"""tests/ 공통 fixture.

Phase 5-A-2 신설 — jaemin.mp4 경로 + default AppConfig fixture.
기존 test_input_validator.py / test_video_preprocessor.py는 본 conftest 의존 X
(자기충족 hardcode 메타 + 합성 JPEG). 본 conftest는 metrics integration test
에서 처음 사용.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from choborunner_ai.config import AppConfig


JAEMIN_VIDEO_PATH = Path("legacy/demo_02/jaemin.mp4")


@pytest.fixture
def jaemin_video_path() -> Path:
    """jaemin.mp4 경로 fixture. 영상 미존재 시 pytest.skip.

    위치: legacy/demo_02/jaemin.mp4 (작업자 PC 로컬). CI 등 영상 없는 환경은
    자동 skip — pytest 실패 X.
    """
    if not JAEMIN_VIDEO_PATH.is_file():
        pytest.skip(
            f"jaemin.mp4 미존재 (검증 환경 외): {JAEMIN_VIDEO_PATH.resolve()}"
        )
    return JAEMIN_VIDEO_PATH


@pytest.fixture
def app_cfg() -> AppConfig:
    """default AppConfig (모든 sub-config default)."""
    return AppConfig()
