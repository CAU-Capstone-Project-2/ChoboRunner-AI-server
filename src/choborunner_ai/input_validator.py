"""Input Validator — 영상 입력 규격 및 메타데이터 검증.

설계문서: docs/2-3-1.md
대응 카테고리:
- 즉시 검증 (첫 frame 시점): 해상도, decode 가능 여부
- 누적 검증 (분석 종료 시점): duration, effective fps, frame count
- 통합 (세션 종료): aggregate_results, validate_session

임계값 출처: InputMetadataConfig (config.py).
본 모듈은 임계값을 직접 박지 않고 DI(Dependency Injection)으로
InputMetadataConfig 또는 AppConfig를 받음.

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

from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from choborunner_ai.config import AppConfig, InputMetadataConfig

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


def validate_duration(
    duration_sec: float,
    cfg: InputMetadataConfig,
) -> ValidationResult:
    """누적 분석 시간 검증 (docs/2-3-1 §3-1 누적 검증).

    분기 (cfg default 기준):
    - duration_sec < 5.0초          → FAILED, reason="too_short"
    - 5.0초 ≤ duration_sec < 10.0초 → LOW_CONFIDENCE, reason="short_duration"
    - 그 외                         → OK

    low_confidence 의미: docs/2-3-5 §6-1 — 결과 제공 + 재촬영 권장 메시지
    동시 표시.

    Args:
        duration_sec: 누적 분석 시간 (초).
        cfg: 임계 출처 — `duration_failed_sec`, `duration_low_confidence_sec`.

    Returns:
        FAILED / LOW_CONFIDENCE / OK 중 하나, 측정값·임계 details 포함.
    """
    if duration_sec < cfg.duration_failed_sec:
        return ValidationResult(
            status=ValidationStatus.FAILED,
            reason="too_short",
            details={
                "duration_sec": duration_sec,
                "threshold": cfg.duration_failed_sec,
            },
        )
    if duration_sec < cfg.duration_low_confidence_sec:
        return ValidationResult(
            status=ValidationStatus.LOW_CONFIDENCE,
            reason="short_duration",
            details={
                "duration_sec": duration_sec,
                "threshold": cfg.duration_low_confidence_sec,
            },
        )
    return ValidationResult(status=ValidationStatus.OK)


def validate_effective_fps(
    received_frames: int,
    duration_sec: float,
    cfg: InputMetadataConfig,
) -> ValidationResult:
    """Effective FPS 검증 (docs/2-3-1 §3-1 누적 검증).

    Effective FPS = `received_frames / duration_sec`. nominal fps (Android
    캡처 설정값) 아닌, AI 서버가 실제 수신·디코딩에 성공한 frame 수 기준
    (§3-1 주석).

    분기 (cfg default 기준):
    - duration_sec ≤ 0                  → FAILED, reason="invalid_duration"
      (ZeroDivisionError 방어, fps 계산 불가)
    - effective_fps < 24.0              → FAILED, reason="low_fps"
    - 24.0 ≤ effective_fps < 30.0       → LOW_CONFIDENCE, reason="borderline_fps"
    - 그 외                             → OK

    ⚠️ 24fps 임계 보정 마일스톤 (모듈 docstring 참조):
    실측 환경에서 nominal 30fps 캡처해도 effective는 18~22fps까지 떨어질 수
    있음. LTE/5G/Wi-Fi 환경 측정 후 임계 보정 예정 (docs/2-3-1 §6).

    Args:
        received_frames: 분석 종료까지 수신·디코딩 성공한 frame 수.
        duration_sec: 누적 분석 시간 (초).
        cfg: 임계 출처.

    Returns:
        FAILED / LOW_CONFIDENCE / OK 중 하나, 측정값·임계 details 포함.
    """
    if duration_sec <= 0.0:
        return ValidationResult(
            status=ValidationStatus.FAILED,
            reason="invalid_duration",
            details={"duration_sec": duration_sec, "note": "fps 계산 불가"},
        )
    effective_fps = received_frames / duration_sec
    if effective_fps < cfg.effective_fps_failed_threshold:
        return ValidationResult(
            status=ValidationStatus.FAILED,
            reason="low_fps",
            details={
                "effective_fps": round(effective_fps, 2),
                "received_frames": received_frames,
                "duration_sec": duration_sec,
                "threshold": cfg.effective_fps_failed_threshold,
            },
        )
    if effective_fps < cfg.effective_fps_low_confidence_threshold:
        return ValidationResult(
            status=ValidationStatus.LOW_CONFIDENCE,
            reason="borderline_fps",
            details={
                "effective_fps": round(effective_fps, 2),
                "received_frames": received_frames,
                "duration_sec": duration_sec,
                "threshold": cfg.effective_fps_low_confidence_threshold,
            },
        )
    return ValidationResult(status=ValidationStatus.OK)


def validate_frame_count(
    frame_count: int,
    cfg: InputMetadataConfig,
) -> ValidationResult:
    """누적 frame 수 검증 (docs/2-3-1 §3-1 누적 검증).

    분기 (cfg default 기준):
    - frame_count < 120         → FAILED, reason="insufficient_frames"
    - 120 ≤ frame_count < 240   → LOW_CONFIDENCE, reason="borderline_frames"
    - 그 외                     → OK

    duration과 함께 `too_short` reason의 트리거 (§3-1 표 "duration 또는
    frame_count 중 하나라도 위반 시 too_short"). 본 함수는 내부 진단 정확성
    위해 `insufficient_frames`/`borderline_frames`로 분리한다. 외부 응답에서
    `too_short`으로 묶을지는 상위 통합 함수(Phase 4) 책임.

    Args:
        frame_count: 누적 frame 수.
        cfg: 임계 출처 — `frame_count_failed`, `frame_count_low_confidence`.

    Returns:
        FAILED / LOW_CONFIDENCE / OK 중 하나, 측정값·임계 details 포함.
    """
    if frame_count < cfg.frame_count_failed:
        return ValidationResult(
            status=ValidationStatus.FAILED,
            reason="insufficient_frames",
            details={
                "frame_count": frame_count,
                "threshold": cfg.frame_count_failed,
            },
        )
    if frame_count < cfg.frame_count_low_confidence:
        return ValidationResult(
            status=ValidationStatus.LOW_CONFIDENCE,
            reason="borderline_frames",
            details={
                "frame_count": frame_count,
                "threshold": cfg.frame_count_low_confidence,
            },
        )
    return ValidationResult(status=ValidationStatus.OK)


def aggregate_results(
    named_results: list[tuple[str, ValidationResult]],
) -> ValidationResult:
    """여러 검증 결과를 우선순위 규칙으로 통합 (docs/2-3-1 §3-2).

    우선순위: FAILED > LOW_CONFIDENCE > OK. 어느 한 항목이라도 FAILED면 최종
    FAILED, LOW_CONFIDENCE 있으면 LOW_CONFIDENCE, 모두 OK면 OK.

    reason 채택 규칙: 최악 status에 해당하는 **첫 매칭 항목의 reason** 채택.
    호출자가 named_results 순서로 우선순위 표현 가능 (예: validate_session은
    first_frame을 첫 항목으로 두어 해상도 실패가 다른 실패보다 먼저 보고됨).

    Args:
        named_results: (check_name, ValidationResult) 페어 리스트. check_name은
            details.sub_results 구성과 외부 진단에 사용.

    Returns:
        ValidationResult — status=worst_status, reason=첫 매칭 reason,
        details 구조:

            {
                "sub_results": [
                    {"check_name", "status", "reason", "details"},
                    ...
                ],
                "summary": {"total", "ok", "low_confidence", "failed"},
            }

    Raises:
        ValueError: named_results가 빈 리스트.
    """
    if not named_results:
        raise ValueError("empty results")
    worst_status = max(r.status for _, r in named_results)
    chosen_reason: str | None = None
    for _, r in named_results:
        if r.status == worst_status:
            chosen_reason = r.reason
            break
    sub_results = [
        {
            "check_name": name,
            "status": r.status.name,
            "reason": r.reason,
            "details": r.details,
        }
        for name, r in named_results
    ]
    counts = Counter(r.status for _, r in named_results)
    summary = {
        "total": len(named_results),
        "ok": counts.get(ValidationStatus.OK, 0),
        "low_confidence": counts.get(ValidationStatus.LOW_CONFIDENCE, 0),
        "failed": counts.get(ValidationStatus.FAILED, 0),
    }
    return ValidationResult(
        status=worst_status,
        reason=chosen_reason,
        details={"sub_results": sub_results, "summary": summary},
    )


def validate_session(
    *,
    first_frame_width: int,
    first_frame_height: int,
    received_frames: int,
    duration_sec: float,
    cfg: AppConfig,
) -> ValidationResult:
    """세션 종료 시점 누적 메타데이터 통합 검증 (docs/2-3-1 §3).

    4개 검증 함수를 호출하고 `aggregate_results`로 우선순위 통합:

    1. `validate_first_frame` — 해상도 (긴 변)
    2. `validate_duration` — 누적 분석 시간
    3. `validate_effective_fps` — 실수신 fps
    4. `validate_frame_count` — 누적 frame 수

    ⚠️ `validate_frame_decodable`은 본 함수에서 호출하지 않는다. decode 검증은
    프레임 수신 중 즉시 검증(per-frame)이라 세션 종료 시점 호출 위치가 다름.
    WebSocket handler에서 frame 수신 직후 별도 호출 예정.

    keyword-only 시그니처 — 4개 숫자 인자가 의미 유사(특히 `received_frames`
    와 `duration_sec`의 단위 혼동)해서 positional 호출 금지로 호출 명확성 확보.

    Args:
        first_frame_width: 첫 frame 픽셀 너비.
        first_frame_height: 첫 frame 픽셀 높이.
        received_frames: 수신·디코딩 성공한 frame 수 (frame_count와 동일 값).
        duration_sec: 누적 분석 시간 (초).
        cfg: AppConfig — 내부에서 `cfg.input_metadata` 사용.

    Returns:
        4개 검증의 통합 `ValidationResult`. result_serializer 입력으로 그대로
        사용 가능.
    """
    meta = cfg.input_metadata
    named_results: list[tuple[str, ValidationResult]] = [
        ("first_frame", validate_first_frame(first_frame_width, first_frame_height, meta)),
        ("duration", validate_duration(duration_sec, meta)),
        ("effective_fps", validate_effective_fps(received_frames, duration_sec, meta)),
        ("frame_count", validate_frame_count(received_frames, meta)),
    ]
    return aggregate_results(named_results)
