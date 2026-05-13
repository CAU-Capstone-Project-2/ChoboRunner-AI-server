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
from typing import Any


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
