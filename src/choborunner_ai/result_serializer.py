"""docs/2-3-7 응답 메시지 정식판 (Phase 7-A).

본 모듈은 docs/2-3-7 (자세 분석 결과 구조화) 단일 정답 구현.
Pydantic v2 BaseModel + 4 메시지 종류 + 보조 모델 + 통계 helper.

Phase 7-A 결정 사항 (Day 5/6 잠금, 압축 모드):
- (i)   docs/2-3-7 단일 정답 정합 (어제 5/15 회고 미해결 1번 해소)
- (ii)  Pydantic 모델 docs §5 1:1 매핑
- (iii) schema만, reason_codes/status 산출은 Phase 8 (docs/2-3-5 §8-7 우선순위
        + docs/2-3-6 트리거 룰)
- (iv)  통계 helper 본 모듈 (compute_angle_stats / compute_fsp_dominant)
- (v)   4 메시지 schema 모두 (재민 미팅 인터페이스 정합)
- (vi)  reference_feedback_only 필드 제외 (docs/2-3-7 정합)
- (vii) classification UPPER 변환 ('rfs' → 'RFS')

⚠️ CLAUDE.md §1 vs docs/2-3-7 충돌 catch (Day 6 학습 자산):
- CLAUDE.md §1: "모든 JSON 응답에 reference_feedback_only: true 필수"
- docs/2-3-7: 본 필드 명시 없음
- decision (vi): docs/2-3-7 정합 (필드 없음) — 별도 회고에서 docs 보강 또는
  CLAUDE.md §1 수정 검토. Day 6 6번째 docs 정합 catch.

4 메시지 (docs §1):
- FrameInferenceMessage (docs §3): 매 frame 직후 응답, 디버그 성격
- AnalysisProgressMessage (docs §4): 진행 상태 + 실시간 피드백
- AnalysisResultMessage (docs §5): 최종 누적 응답 (3 status 필드 분기)
- ErrorMessage (docs §6): 시스템 에러

reason_codes는 str 단일 타입 (Phase 8에서 Literal로 좁힘).
docs/2-3-5 §8 reason code 사전 20+개 — 본 Phase 7은 schema만, 산출 Phase 8.

분류 라벨 변환 (decision vii):
- Phase 5 dataclass: 'rfs'/'mfs'/'ffs' (lower, Day 5 lock)
- docs §5-5 응답: 'RFS'/'MFS'/'FFS' (UPPER)
- convert_fsp_label() helper로 변환

Phase 6 VideoMeta → docs/2-3-7 video_meta 변환:
- 필드 매핑: frame_count→total_frames / fps→fps_actual /
  width+height→resolution{width,height}
- duration_sec = frame_count / fps_safe (derived)
- rotation_degrees는 docs/2-3-7에 없음 → 제외
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from choborunner_ai.metrics.foot_strike import FootStrikeClassification
from choborunner_ai.video_preprocessor import VideoMeta as Phase6VideoMeta


# ============================================================
# Literal 타입 (docs §1, §4-5, §5-5, §5-6)
# ============================================================


MessageType = Literal[
    "frame_inference", "analysis_progress", "analysis_result", "error"
]
"""docs/2-3-7 §1 4 메시지 종류."""


AnalysisStatus = Literal["success", "low_confidence", "failed"]
"""docs/2-3-5 §6 / docs/2-3-7 §5 분석 상태값."""


AnalysisStage = Literal["warming_up", "collecting_strides", "analyzing"]
"""docs/2-3-7 §4-5 분석 단계."""


AnalysisSide = Literal["left", "right"]
"""docs/2-3-5 §3 분석측."""


FootStrikePatternLabel = Literal["RFS", "MFS", "FFS"]
"""docs/2-3-7 §5-5 foot_strike_pattern UPPER 라벨 (Phase 5 lower 변환 후).

Phase 5 FootStrikeClassification ('rfs'/'mfs'/'ffs', Day 5 lock) → UPPER 변환.
"""


TrackingStability = Literal["stable", "borderline", "unstable"]
"""docs/2-3-7 §5-5 quality_summary.target_tracking_stability."""


FeedbackCategory = Literal["system_info", "posture_warning", "posture_info"]
"""docs/2-3-6 §1 + docs/2-3-7 §5-6 카테고리."""


FrameQualityFlag = Literal[
    "low_brightness", "motion_blur", "frame_unstable", "timestamp_fallback"
]
"""docs/2-3-2 §4 frame quality flags."""


# ============================================================
# 보조 모델 (docs §5-5)
# ============================================================


class Resolution(BaseModel):
    """docs/2-3-7 §5-5 video_meta.resolution."""

    width: int
    height: int


class VideoMeta(BaseModel):
    """docs/2-3-7 §5-5 video_meta (정규화 후 해상도)."""

    duration_sec: float
    fps_actual: float
    resolution: Resolution
    total_frames: int


class AngleStats(BaseModel):
    """docs/2-3-7 §5-5 metric_details per metric.

    Attributes:
        median: stride 중앙값 (= metrics 단일값과 동일).
        iqr: [Q1, Q3] 사분위 범위.
        n_strides: 통계 모집단 (NaN 제외).
    """

    median: float
    iqr: list[float] = Field(min_length=2, max_length=2)
    n_strides: int


class Metrics(BaseModel):
    """docs/2-3-7 §5-5 metrics (핵심 지표 3개 단일값)."""

    foot_strike_pattern: FootStrikePatternLabel
    foot_strike_angle_deg: float
    initial_knee_flexion_deg: float
    trunk_lean_deg: float


class MetricDetails(BaseModel):
    """docs/2-3-7 §5-5 metric_details (지표별 stride 통계)."""

    foot_strike_angle_deg: AngleStats
    initial_knee_flexion_deg: AngleStats
    trunk_lean_deg: AngleStats


class QualitySummary(BaseModel):
    """docs/2-3-7 §5-5 quality_summary."""

    valid_frame_ratio: float = Field(ge=0.0, le=1.0)
    ic_candidate_count: int
    valid_stride_count: int
    landmark_visibility_avg: float = Field(ge=0.0, le=1.0)
    target_tracking_stability: TrackingStability


class FeedbackMessage(BaseModel):
    """docs/2-3-7 §5-6 + docs/2-3-6 §1~§3 피드백 메시지.

    Attributes:
        category: 'system_info' / 'posture_warning' / 'posture_info'.
        metric: 어떤 지표 관련. 시스템 안내는 None.
        tts_text: TTS 음성 합성용 짧은 문구 (~1초). 출력 안 할 메시지는 None.
        display_text: 화면 표시용 자세한 문구.
        priority: 1(최우선) / 2 / 3 (docs/2-3-6 §3-3).
        tts_enabled: Android TTS 합성 여부.
        confidence_prefix: low_confidence 시 True (신뢰도 안내 prefix).
    """

    category: FeedbackCategory
    metric: Optional[str] = None
    tts_text: Optional[str] = None
    display_text: str
    priority: Literal[1, 2, 3]
    tts_enabled: bool
    confidence_prefix: bool


class FrameInferenceResult(BaseModel):
    """docs/2-3-7 §3-2 frame_inference.result."""

    pose_detected: bool
    frame_quality_flags: list[FrameQualityFlag] = Field(default_factory=list)


# ============================================================
# 4 메시지 종류 (docs §3, §4, §5, §6)
# ============================================================


class FrameInferenceMessage(BaseModel):
    """docs/2-3-7 §3 frame_inference — 매 frame 직후 응답 (디버그 성격).

    docs §3-4 정책: 항상 status: ok (별도 status 필드 X). Frame 단위 비정상은
    result 내부 (pose_detected=False, frame_quality_flags)로 표현.
    """

    type: Literal["frame_inference"] = "frame_inference"
    frame_index: int
    timestamp_sec: float
    result: FrameInferenceResult


class AnalysisProgressMessage(BaseModel):
    """docs/2-3-7 §4 analysis_progress — 진행 상태 + 실시간 피드백.

    docs §4-1 발생 시점: 단계 전환 또는 stride 갱신 (analyzing 단계).
    """

    type: Literal["analysis_progress"] = "analysis_progress"
    stage: AnalysisStage
    valid_stride_count: int
    elapsed_sec: float
    message: Optional[str] = None
    feedback_messages: Optional[list[FeedbackMessage]] = None


class AnalysisResultMessage(BaseModel):
    """docs/2-3-7 §5 analysis_result — 최종 누적 응답 (3 status 분기).

    필드 매트릭스 (docs §5-5):
    - success: 모든 필드 (primary_reason_code=null, reason_codes=[])
    - low_confidence: success 동일 + primary_reason_code/message 필수
    - failed: type, status, video_meta, primary_reason_code, reason_codes, message
              (metrics/metric_details/analysis_side/quality_summary/
              feedback_messages 제외)

    Pydantic Optional 필드 + None default로 처리. 호출자가 status별 분기 책임
    (Phase 8 책임).
    """

    type: Literal["analysis_result"] = "analysis_result"
    status: AnalysisStatus
    video_meta: VideoMeta
    analysis_side: Optional[AnalysisSide] = None
    metrics: Optional[Metrics] = None
    metric_details: Optional[MetricDetails] = None
    quality_summary: Optional[QualitySummary] = None
    primary_reason_code: Optional[str] = None
    reason_codes: list[str] = Field(default_factory=list)
    message: Optional[str] = None
    feedback_messages: Optional[list[FeedbackMessage]] = None


class ErrorMessage(BaseModel):
    """docs/2-3-7 §6 error — 시스템 처리 실패.

    docs §6-1: failed 상태와 다른 차원 (failed=비즈니스 결과, error=시스템 에러).
    """

    type: Literal["error"] = "error"
    frame_index: Optional[int] = None
    error_code: str
    error_detail: str


# ============================================================
# 통계 helper (decision iv)
# ============================================================


def compute_angle_stats(values: list[float]) -> AngleStats:
    """list[float] → AngleStats. NaN 자동 제외.

    Args:
        values: 각도 list (도). NaN 포함 가능.

    Returns:
        AngleStats(median, iqr=[Q1, Q3], n_strides). 빈 입력 시
        median/iqr=NaN, n_strides=0.
    """
    finite = [v for v in values if math.isfinite(v)]
    n = len(finite)
    if n == 0:
        return AngleStats(
            median=float("nan"),
            iqr=[float("nan"), float("nan")],
            n_strides=0,
        )
    arr = np.array(finite)
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return AngleStats(median=median, iqr=[q1, q3], n_strides=n)


def compute_fsp_dominant(
    classifications: list[Optional[FootStrikeClassification]],
) -> tuple[Optional[FootStrikePatternLabel], float, dict[str, int]]:
    """list classifications → (최빈 UPPER 라벨, 비율, 분포 dict).

    None 자동 제외 (Uncertain). 빈 입력 또는 모두 None → (None, 0.0, {}).

    Args:
        classifications: list[FootStrikeClassification | None] (Phase 5 dataclass).

    Returns:
        (dominant_upper, ratio, distribution):
        - dominant_upper: 'RFS'/'MFS'/'FFS' 또는 None (모두 Uncertain).
        - ratio: dominant count / total (None 제외 모집단).
        - distribution: {UPPER 라벨: count}.
    """
    valid = [c for c in classifications if c is not None]
    if not valid:
        return None, 0.0, {}
    counter = Counter(valid)
    dominant_lower, count = counter.most_common(1)[0]
    dominant_upper = convert_fsp_label(dominant_lower)
    ratio = count / len(valid)
    distribution = {convert_fsp_label(k): v for k, v in counter.items()}
    return dominant_upper, ratio, distribution


def convert_fsp_label(label_lower: FootStrikeClassification) -> FootStrikePatternLabel:
    """Phase 5 lower 라벨 → docs/2-3-7 UPPER 라벨 변환 (decision vii).

    Args:
        label_lower: 'rfs' / 'mfs' / 'ffs'.

    Returns:
        'RFS' / 'MFS' / 'FFS'.
    """
    return label_lower.upper()  # type: ignore[return-value]


def convert_phase6_video_meta(meta: Phase6VideoMeta) -> VideoMeta:
    """Phase 6 VideoMeta → docs/2-3-7 video_meta 변환.

    필드 매핑:
    - frame_count → total_frames
    - fps → fps_actual
    - width + height → resolution: {width, height}
    - duration_sec = frame_count / fps_safe (derived)
    - rotation_degrees는 docs/2-3-7에 없음 → 제외

    fps_safe: fps <= 1e-6 시 30.0 default (Phase 6 패턴 일관).
    """
    fps_safe = meta.fps if meta.fps > 1e-6 else 30.0
    return VideoMeta(
        duration_sec=meta.frame_count / fps_safe,
        fps_actual=meta.fps,
        resolution=Resolution(width=meta.width, height=meta.height),
        total_frames=meta.frame_count,
    )
