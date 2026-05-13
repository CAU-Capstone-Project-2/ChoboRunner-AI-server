"""Input Validator — 영상 입력 규격 및 메타데이터 검증.

설계문서: docs/2-3-1.md
대응 카테고리:
- 즉시 검증 (첫 frame 시점): 해상도, decode 가능 여부
- 누적 검증 (분석 종료 시점): duration, effective fps, frame count

임계값 출처: InputMetadataConfig (config.py).
본 모듈은 임계값을 직접 박지 않고 DI(Dependency Injection)으로
InputMetadataConfig를 받음.

⚠️ Day 4~ 보정 마일스톤:
- effective_fps_failed_threshold (24fps): 실측 환경에서 18~22fps 빈발 가능,
  LTE/5G/Wi-Fi 환경 측정으로 보정 예정 (docs/2-3-1 §6)
- effective_fps_low_confidence_threshold (30fps): 위와 동일
- analysis_end_timeout_sec (2.0초): 백엔드 heartbeat interval과 동기화 필요

References:
- docs/2-3-1.md §3-1: 4개 임계값 표
- docs/2-3-1.md §3-2: failed > low_confidence > success 우선순위
- docs/2-3-5.md §6-1: low_confidence 처리 (결과 + 재촬영 권장)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from choborunner_ai.config import InputMetadataConfig

if TYPE_CHECKING:
    import numpy as np


class ValidationStatus(IntEnum):
    """검증 결과 상태. 큰 값일수록 심각."""

    OK = 0
    LOW_CONFIDENCE = 1
    FAILED = 2


@dataclass(frozen=True)
class ValidationResult:
    """검증 함수의 표준 반환 타입.

    Attributes:
        status: 검증 결과 상태 (OK / LOW_CONFIDENCE / FAILED).
        reason: reason code 문자열. status가 OK면 None.
            예: ``"too_short"``, ``"low_fps"``, ``"low_resolution"``,
            ``"decode_failed"``. 전체 사전은 docs/2-3-5.md §8 참조.
        details: 추가 진단 정보 (실제 측정값·임계값 등).
            예: ``{"duration_sec": 4.2, "threshold": 5.0}``.
    """

    status: ValidationStatus
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def validate_first_frame(
    width: int,
    height: int,
    cfg: InputMetadataConfig,
) -> ValidationResult:
    """첫 frame 시점 해상도 검증 (docs/2-3-1 §3-1, §3-4 즉시 검증).

    긴 변 기준으로 비교한다 — 입력이 가로(landscape) / 세로(portrait) 어느
    방향이든 회전과 무관하게 동일 임계 적용. low_confidence 단계 없음
    (docs/2-3-1 §3-1 표의 비대칭 설계: 해상도 행 "별도 임계 없음" 명시).

    Args:
        width: frame 픽셀 너비.
        height: frame 픽셀 높이.
        cfg: 임계 출처 — `cfg.min_resolution_long_edge_px` 사용.

    Returns:
        - FAILED + reason="low_resolution" + details {"long_edge_px", "threshold"}:
          긴 변이 임계 미만.
        - 그 외 OK.
    """
    long_edge = max(width, height)
    if long_edge < cfg.min_resolution_long_edge_px:
        return ValidationResult(
            status=ValidationStatus.FAILED,
            reason="low_resolution",
            details={
                "long_edge_px": long_edge,
                "threshold": cfg.min_resolution_long_edge_px,
            },
        )
    return ValidationResult(status=ValidationStatus.OK)


def validate_frame_decodable(frame: np.ndarray | None) -> ValidationResult:
    """디코딩 결과 frame의 구조 검증 (docs/2-3-1 §3-4 즉시 검증).

    `cv2.VideoCapture.read()`가 실패하면 None을 반환할 수 있으므로 None을
    안전하게 처리한다. 임계값 의존이 없는 구조 검증이므로 cfg 인자가 없음.

    Args:
        frame: BGR np.ndarray (H, W, 3) 또는 None.

    Returns:
        - FAILED + reason="decode_failed" + details {"cause": "frame_is_none"}:
          frame이 None.
        - FAILED + reason="decode_failed" + details {"cause": "invalid_shape", ...}:
          (H, W, 3) 형태가 아님.
        - 그 외 OK.
    """
    if frame is None:
        return ValidationResult(
            status=ValidationStatus.FAILED,
            reason="decode_failed",
            details={"cause": "frame_is_none"},
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        return ValidationResult(
            status=ValidationStatus.FAILED,
            reason="decode_failed",
            details={
                "cause": "invalid_shape",
                "ndim": int(frame.ndim),
                "shape": tuple(int(s) for s in frame.shape),
            },
        )
    return ValidationResult(status=ValidationStatus.OK)
