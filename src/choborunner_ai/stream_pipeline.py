"""ChoboRunner AI Server — 증분(stream) 진입점 (Phase WS-A-2).

docs/2-4-2 §7-3 — ``Pipeline``의 file path 배치 모드(``run_on_video_file``)와
달리, WebSocket 연결로 들어오는 frame을 한 장씩 ``push_frame``으로 밀어넣고
누적 상태를 보관한다. 종료 시 ``finalize``가 ``Pipeline._accumulate``
(Phase WS-A-1 추출)를 호출해 배치 모드와 **동일한** 누적 평가·Phase 5 metrics·
응답 조립 로직을 공유한다.

CLAUDE.md §6 정책: 본 모듈도 조립만 담당. 계산 로직은 ``Pipeline``/단위 모듈.
``StreamPipeline``은 ``Pipeline`` 서브클래스 — PoseExtractor 생성·close·
``_extract_and_evaluate_one_frame``·``_accumulate_one_frame``·``_accumulate``를
전부 상속하고, 스트림 누적 상태 + 3개 진입점(push_frame/snapshot_progress/
finalize)만 추가한다.

Phase 관계:
- Phase WS-A-1: ``Pipeline._accumulate`` 추출 (§2~§9 누적 평가 + 응답 조립)
- Phase WS-A-2 (본 모듈): ``Pipeline._accumulate_one_frame`` 추출 (§1 frame당
  처리) + ``StreamPipeline`` 신규
- Phase WS-B: server WebSocket 레이어 (``StreamSession``이 본 클래스 소유)
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

from choborunner_ai.config import AppConfig
from choborunner_ai.metrics import foot_strike, knee_flexion, trunk_lean
from choborunner_ai.metrics.ic_detector import compute_ic_indices
from choborunner_ai.pipeline import Pipeline, PipelineResult
from choborunner_ai.pose_extractor import PoseLandmarks
from choborunner_ai.quality_gate import (
    FrameGeometryResult,
    FrameSideViewResult,
    FrameVisibilityResult,
)
from choborunner_ai.result_serializer import (
    AnalysisProgressMessage,
    AnalysisResultMessage,
    AnalysisStage,
    FeedbackMessage,
    FrameInferenceMessage,
    FrameInferenceResult,
    FrameQualityFlag,
    compute_angle_stats,
    compute_classification_dominant,
    compute_fsp_dominant,
    compute_uncertain_stride_count,
)
from choborunner_ai.video_preprocessor import VideoMeta

logger = logging.getLogger(__name__)


# docs/2-3-7 §4-2/§4-3 예시 — analysis_progress.message stage별 텍스트
_PROGRESS_MESSAGE_BY_STAGE: dict[AnalysisStage, str] = {
    "warming_up": "분석 준비 중입니다.",
    "collecting_strides": "분석 중입니다.",
    "analyzing": "분석 중입니다.",
}


class StreamPipeline(Pipeline):
    """증분(stream) 진입점 — WebSocket 연결 1개 = 인스턴스 1개.

    docs/2-4-2 §7-2/§7-3 — ``StreamSession``이 연결당 본 인스턴스 1개를 소유한다.
    frame을 ``push_frame``으로 증분 누적하고, 종료 시 ``finalize``로 최종 응답을
    조립한다. 연결 종료 시 반드시 ``close()`` 호출 (PoseExtractor 누수 방지) —
    with-as 컨텍스트 매니저 사용 권장.

    상태:
    - frame-level 누적 5종 (``_accumulate_one_frame``이 채움) + pose 미검출 카운터
    - frame 인덱스 + MediaPipe stream timestamp 단조성 가드 (docs/2-4-2 §8-3)
    - 마지막 frame 크기·timestamp 범위 (finalize 시 합성 VideoMeta 도출용)

    analysis_side·direction은 WebSocket으로 전달되지 않으며 v1 기본값
    (``left`` / ``left_to_right``)을 사용한다 (docs/2-4-2 §3, pipeline.py 현행
    default 일관). 영상 기반 자동 결정은 docs §3-1 Phase 9 anchor.
    """

    def __init__(
        self,
        cfg: AppConfig,
        analysis_side: Literal["left", "right"] = "left",
        direction: Literal["left_to_right", "right_to_left"] = "left_to_right",
    ) -> None:
        """초기화 — Pipeline(PoseExtractor 1개 생성) + 스트림 누적 상태.

        Args:
            cfg: AppConfig (DI).
            analysis_side: 'left' 또는 'right' (docs/2-4-2 §3 v1 기본 'left').
            direction: 'left_to_right' 또는 'right_to_left' (v1 기본).

        Raises:
            FileNotFoundError: PoseExtractor 모델 파일(.task) 부재.
            RuntimeError: PoseExtractor 초기화 실패.
        """
        super().__init__(cfg)
        self._analysis_side = analysis_side
        self._direction = direction

        # frame-level 누적 컨테이너 (run_on_video_file 배치 루프와 동일 5종)
        self._frame_results: list[FrameVisibilityResult] = []
        self._landmarks_series: list[Optional[PoseLandmarks]] = []
        self._body_inclusion_results: list[FrameGeometryResult] = []
        self._foot_cutoff_results: list[FrameGeometryResult] = []
        self._side_view_results: list[FrameSideViewResult] = []
        self._pose_not_detected_count = 0

        # frame 인덱스 + MediaPipe stream timestamp 단조성 가드 (docs/2-4-2 §8-3)
        self._frame_count = 0
        self._last_mp_ts = -1

        # 합성 VideoMeta 도출용 — 마지막 frame 크기 + timestamp 범위
        self._last_frame_hw: Optional[tuple[int, int]] = None
        self._first_ts_ms: Optional[int] = None
        self._last_ts_ms: Optional[int] = None

        # 실시간 피드백 빈도 dedup 상태 (docs/2-3-6 §3-2)
        # key = FeedbackMessage.display_text, value = 마지막 송신 시각(_elapsed_sec)
        self._last_emit_time_by_display: dict[str, float] = {}
        self._last_any_emit_time: Optional[float] = None

    def __enter__(self) -> "StreamPipeline":
        """with-as 컨텍스트 진입 — self 반환 (Pipeline.__exit__가 close 자동 호출)."""
        return self

    def push_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        frame_quality_flags: Optional[list[FrameQualityFlag]] = None,
    ) -> FrameInferenceMessage:
        """단일 frame 증분 처리 → FrameInferenceMessage (docs/2-3-7 §3).

        docs/2-4-2 §7-3 — ``Pipeline._accumulate_one_frame``로 frame-level 누적
        5종을 증분 갱신하고, 매 frame 직후 응답인 ``FrameInferenceMessage``를
        조립한다.

        ⚠️ MediaPipe stream 모드 단조성 가드 (docs/2-4-2 §8-3): wire 규약상
        ``ts_ms``는 비감소(non-decreasing)만 보장하므로, MediaPipe에 넘기는
        timestamp는 직전 값보다 같거나 작으면 +1 ms bump해 엄격 증가를 보장한다.
        bump는 MediaPipe 호출용 내부 값일 뿐 — 응답 ``timestamp_sec``는 원본 유지.

        Args:
            frame: BGR np.ndarray. server 측에서 JPEG decode·전처리 완료된 frame.
            timestamp_ms: frame capture timestamp (ms). docs/2-4-2 §4 wire 규약
                헤더 값 또는 fallback 해소 후 값.
            frame_quality_flags: server 전처리(video_preprocessor)가 산출한 frame
                품질 플래그. None이면 빈 list.

        Returns:
            FrameInferenceMessage — frame_index·timestamp_sec·pose_detected.
        """
        frame_index = self._frame_count
        self._frame_count += 1

        # 합성 VideoMeta 도출용 메타 갱신
        self._last_frame_hw = (int(frame.shape[0]), int(frame.shape[1]))
        if self._first_ts_ms is None:
            self._first_ts_ms = timestamp_ms
        self._last_ts_ms = timestamp_ms

        # MediaPipe stream 단조성 가드 — +1 ms bump (docs/2-4-2 §8-3)
        mp_ts = max(timestamp_ms, self._last_mp_ts + 1)
        self._last_mp_ts = mp_ts

        pose_detected = self._accumulate_one_frame(
            frame,
            mp_ts,
            self._analysis_side,
            frame_results=self._frame_results,
            landmarks_series=self._landmarks_series,
            body_inclusion_results=self._body_inclusion_results,
            foot_cutoff_results=self._foot_cutoff_results,
            side_view_results=self._side_view_results,
        )
        if not pose_detected:
            self._pose_not_detected_count += 1

        return FrameInferenceMessage(
            frame_index=frame_index,
            timestamp_sec=timestamp_ms / 1000.0,
            result=FrameInferenceResult(
                pose_detected=pose_detected,
                frame_quality_flags=frame_quality_flags or [],
            ),
        )

    def snapshot_progress(self) -> AnalysisProgressMessage:
        """현재까지 누적분으로 진행 상태 산출 → AnalysisProgressMessage (docs/2-3-7 §4).

        docs/2-4-2 §7-3 — IC 검출·Phase 5 metrics를 현재 ``landmarks_series``로
        재계산한다(비용 있음 — 호출 빈도는 server 레이어 결정, docs/2-4-2 §9 #3).

        stage 판정은 임계값 없는 presence 기반:
        - ``warming_up``: IC 미검출 (분석할 stride 없음)
        - ``collecting_strides``: IC 검출됐으나 3 metric 공통 valid stride 0
        - ``analyzing``: 3 metric 모두 valid stride ≥ 1

        ``valid_stride_count``는 ``result_serializer._build_quality_summary``와
        동일 정의 — ``min(foot_valid, knee_valid, trunk_valid)``.

        실시간 피드백 (docs/2-3-7 §4-3, docs/2-3-6):
        - ``analyzing`` 단계에서만 ``feedback_messages`` 생성 (docs/2-3-7 §4-5).
        - mid-stream status는 ``success`` 고정 — reason_code 기반 low_confidence
          /failed 판단은 누적 평가가 필요하므로 ``finalize``의 최종 응답에서만.
        - 빈도 제한은 ``_filter_by_frequency``로 dedup (docs/2-3-6 §3-2).

        Returns:
            AnalysisProgressMessage — stage·valid_stride_count·elapsed_sec +
            stage별 ``message`` + analyzing 단계 한정 ``feedback_messages``.
        """
        ic_results = compute_ic_indices(
            self._landmarks_series, self._analysis_side, self._cfg.ic
        )
        ic_indices = [r.frame_index for r in ic_results]

        trunk_results = trunk_lean.compute_at_ic(
            self._landmarks_series, ic_indices, self._cfg.trunk_lean
        )
        knee_results = knee_flexion.compute_at_ic(
            self._landmarks_series,
            ic_indices,
            self._analysis_side,
            self._cfg.knee_flexion,
        )
        foot_results = foot_strike.compute_at_ic(
            self._landmarks_series,
            ic_indices,
            self._analysis_side,
            self._direction,
            self._cfg.foot_strike,
        )

        foot_valid = sum(1 for r in foot_results if r.is_valid)
        knee_valid = sum(1 for r in knee_results if r.is_valid)
        trunk_valid = sum(1 for r in trunk_results if r.is_valid)
        valid_stride_count = min(foot_valid, knee_valid, trunk_valid)

        if not ic_indices:
            stage: AnalysisStage = "warming_up"
        elif valid_stride_count >= 1:
            stage = "analyzing"
        else:
            stage = "collecting_strides"

        feedback_messages: Optional[list[FeedbackMessage]] = None
        if stage == "analyzing":
            candidates = self._compute_feedback_candidates(
                foot_results, knee_results, trunk_results
            )
            kept = self._filter_by_frequency(candidates)
            feedback_messages = kept if kept else None

        return AnalysisProgressMessage(
            stage=stage,
            valid_stride_count=valid_stride_count,
            elapsed_sec=self._elapsed_sec(),
            message=_PROGRESS_MESSAGE_BY_STAGE[stage],
            feedback_messages=feedback_messages,
        )

    def _compute_feedback_candidates(
        self,
        foot_results: list,
        knee_results: list,
        trunk_results: list,
    ) -> list[FeedbackMessage]:
        """누적 결과 → FeedbackContext → compute_feedback_messages 호출 (docs/2-3-6).

        mid-stream status는 ``success`` 고정 (snapshot_progress docstring 참조).
        ``build_analysis_result``의 §6 패턴과 동일한 dominant·IQR 산출 사용.
        """
        # ⚠️ Lazy import (circular 회피 — build_analysis_result와 동일)
        from choborunner_ai.feedback_engine import (
            FeedbackContext,
            compute_feedback_messages,
        )

        foot_stats = compute_angle_stats([r.deg for r in foot_results])
        knee_stats = compute_angle_stats([r.deg for r in knee_results])
        trunk_stats = compute_angle_stats([r.deg for r in trunk_results])

        foot_dominant_upper, _ratio, _dist = compute_fsp_dominant(
            [r.classification for r in foot_results]
        )
        trunk_dominant = compute_classification_dominant(
            [r.classification for r in trunk_results]
        )
        knee_dominant = compute_classification_dominant(
            [r.classification for r in knee_results]
        )
        uncertain_count = compute_uncertain_stride_count(foot_results)

        ctx = FeedbackContext(
            status="success",
            primary_reason_code=None,
            trunk_classification=trunk_dominant,
            knee_classification=knee_dominant,
            foot_dominant=foot_dominant_upper,
            foot_iqr=(foot_stats.iqr[0], foot_stats.iqr[1])
            if foot_stats.n_strides > 0
            else None,
            knee_iqr=(knee_stats.iqr[0], knee_stats.iqr[1])
            if knee_stats.n_strides > 0
            else None,
            trunk_iqr=(trunk_stats.iqr[0], trunk_stats.iqr[1])
            if trunk_stats.n_strides > 0
            else None,
            uncertain_stride_count=uncertain_count,
        )
        return compute_feedback_messages(ctx)

    def _filter_by_frequency(
        self, candidates: list[FeedbackMessage]
    ) -> list[FeedbackMessage]:
        """docs/2-3-6 §3-2 빈도 정책 dedup — display_text 기준.

        - cool-down: 직전 송신 시각으로부터 ``different_message_min_interval_sec``
          (기본 2초) 미달 시 전체 skip (cycle 자체를 건너뜀).
        - same-message: 동일 display_text가 ``same_message_min_interval_sec``
          (기본 5초) 내 재송신되면 skip.
        - positive (GOOD_PACE): ``positive_message_min_interval_sec`` (기본 30초)
          기준으로 same-message 임계를 override.

        emit된 경우 ``_last_emit_time_by_display`` + ``_last_any_emit_time``
        둘 다 갱신한다.
        """
        # GOOD_PACE display_text 식별 (lazy import — feedback_engine에서 가져옴)
        from choborunner_ai.feedback_engine import GOOD_PACE_MESSAGE

        positive_display_text = GOOD_PACE_MESSAGE[1]

        cfg = self._cfg.feedback_frequency
        now = self._elapsed_sec()

        # cool-down — 직전 cycle 송신 후 different_message_min_interval_sec 미달이면 skip
        if self._last_any_emit_time is not None:
            if now - self._last_any_emit_time < cfg.different_message_min_interval_sec:
                return []

        kept: list[FeedbackMessage] = []
        for msg in candidates:
            last_same = self._last_emit_time_by_display.get(msg.display_text)
            is_positive = msg.display_text == positive_display_text
            threshold = (
                cfg.positive_message_min_interval_sec
                if is_positive
                else cfg.same_message_min_interval_sec
            )
            if last_same is not None and (now - last_same) < threshold:
                continue
            kept.append(msg)

        if kept:
            self._last_any_emit_time = now
            for msg in kept:
                self._last_emit_time_by_display[msg.display_text] = now

        return kept

    def finalize(self) -> AnalysisResultMessage:
        """누적 종료 → AnalysisResultMessage (docs/2-3-7 §5).

        docs/2-4-2 §7-3 — ``Pipeline._accumulate``(Phase WS-A-1 추출)에 누적
        5종 + pose 미검출 카운터를 넘겨 배치 모드와 동일한 누적 평가·Phase 5
        metrics·응답 조립을 수행한다. 호출 후에도 인스턴스 상태는 유지되나
        재호출은 권장하지 않는다(누적은 동일 — 동작은 동일).

        스트림 모드는 영상 파일이 없으므로 합성 ``VideoMeta``를 도출한다
        (``_build_video_meta``).

        Returns:
            AnalysisResultMessage — status별 필드 분기(docs/2-3-7 §5-5).
        """
        result: PipelineResult = self._accumulate(
            video_meta=self._build_video_meta(),
            analysis_side=self._analysis_side,
            direction=self._direction,
            frame_results=self._frame_results,
            landmarks_series=self._landmarks_series,
            body_inclusion_results=self._body_inclusion_results,
            foot_cutoff_results=self._foot_cutoff_results,
            side_view_results=self._side_view_results,
            pose_not_detected_count=self._pose_not_detected_count,
        )
        # _accumulate는 build_analysis_result로 analysis_result를 항상 채움.
        assert result.analysis_result is not None, (
            "_accumulate는 analysis_result를 항상 조립해야 함"
        )
        return result.analysis_result

    def _elapsed_sec(self) -> float:
        """첫 frame ~ 마지막 frame timestamp 경과 시간 (초). frame < 1장이면 0.0."""
        if self._first_ts_ms is None or self._last_ts_ms is None:
            return 0.0
        return max(0.0, (self._last_ts_ms - self._first_ts_ms) / 1000.0)

    def _build_video_meta(self) -> VideoMeta:
        """스트림 누적 상태 → 합성 VideoMeta (finalize용).

        스트림 모드는 영상 파일 메타가 없으므로 push_frame 누적분으로 도출한다:
        - width/height: 마지막 push_frame frame 크기 (frame 0장이면 0)
        - fps: ``(frame_count - 1) / 경과 시간`` 추정. frame < 2장 또는 경과
          시간 ≤ 0이면 30.0 fallback (run_on_video_file fps_safe 패턴 일관).
        - frame_count: push_frame 총 호출 수
        - rotation_degrees: 0 — server 전처리가 회전 보정 완료한 frame 전제.
        """
        height, width = (
            self._last_frame_hw if self._last_frame_hw is not None else (0, 0)
        )
        elapsed = self._elapsed_sec()
        if self._frame_count >= 2 and elapsed > 1e-6:
            fps = (self._frame_count - 1) / elapsed
        else:
            fps = 30.0
        return VideoMeta(
            width=width,
            height=height,
            fps=fps,
            frame_count=self._frame_count,
            rotation_degrees=0,
        )
