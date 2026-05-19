"""ChoboRunner AI Server — 모듈 오케스트레이션 (pipeline).

CLAUDE.md §6 정책: 본 모듈은 **조립만** 담당. 계산 로직은 각 단위 모듈
(pose_extractor, video_preprocessor, quality_gate, ...) 책임.

본 Phase 6 scope (Day 5 결정 잠금 5건):
1. file path mode 전용 — WebSocket binary stream mode는 Phase 8 통합·연동 범위
2. 입력: 영상 파일 경로 + 분석측 (CLI 임시 default 'left')
3. 출력: PipelineResult dataclass (frame_results, accumulation_reasons,
   video_meta, pose_detected counter)
4. 분석측 결정 본격 구현(docs §3-1, 1.5초 AND 30 frame 정책)은 별도 Phase
   예정. 본 모듈은 호출자가 입력한 analysis_side 사용.
5. metrics(2-3-4), feedback(2-3-6), result_serializer(2-3-7) 미통합 — 본 Phase
   범위 외. docs/2-3-6 + 2-3-7 빈 placeholder 상태, Phase 7 진입 전 작성 선행
   필요 (Day 6 catch).

Phase 6 작업 단위:
- Phase 6-A: PipelineResult + Pipeline.__init__ + _extract_and_evaluate_one_frame
- Phase 6-B (본 단계): __enter__/__exit__ + run_on_video_file 본체
- Phase 6-C: 통합 sanity (scripts/sanity/)

docs 정합:
- docs/2-3-1 ~ 2-3-5: 본 모듈이 조립하는 단위 모듈 책임 정의 (각 모듈 SoT)
- docs/2-3-6 (feedback) + 2-3-7 (result_serializer): 빈 placeholder
- pipeline.py 직접 대응 docs 없음 — CLAUDE.md §3·§6 정책 + 모듈 인터페이스
  정합으로 진행.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from choborunner_ai.config import AppConfig
from choborunner_ai.metrics import foot_strike, knee_flexion, trunk_lean
from choborunner_ai.metrics.foot_strike import FootStrikeResult
from choborunner_ai.metrics.ic_detector import ICConfidence, ICResult, compute_ic_indices
from choborunner_ai.metrics.knee_flexion import KneeFlexionResult
from choborunner_ai.metrics.trunk_lean import TrunkLeanResult
from choborunner_ai.pose_extractor import PoseExtractor, PoseLandmarks
from choborunner_ai.quality_gate import (
    FrameGeometryResult,
    FrameSideViewResult,
    FrameVisibilityResult,
    ReasonCodeEntry,
    evaluate_body_inclusion_accumulation,
    evaluate_camera_stability,
    evaluate_foot_cutoff_accumulation,
    evaluate_frame_body_inclusion,
    evaluate_frame_foot_cutoff,
    evaluate_frame_side_view,
    evaluate_frame_visibility,
    evaluate_ic_validation,
    evaluate_metric_variability,
    evaluate_side_view_accumulation,
    evaluate_tracking_stability,
    evaluate_visibility_accumulation,
)
from choborunner_ai.result_serializer import (
    AnalysisResultMessage,
    build_analysis_result,
)
from choborunner_ai.video_preprocessor import (
    VideoMeta,
    get_video_meta,
    iter_frames,
)

logger = logging.getLogger(__name__)


# ============================================================
# PipelineResult — file path mode 전용 결과 객체
# ============================================================


@dataclass(eq=False)
class PipelineResult:
    """file path mode 결과 객체 (Phase 8-H 확장).

    Phase 8-H 6 신규 필드 추가 — Phase 5 metrics + Phase 8-A~8-E reason_code 누적
    보존. Phase 8-I (응답 조립) + Phase 8-J (pytest integration) 입력.

    ⚠️ Phase 8-B-1 δ 잔재 해소 (lock 8-H-3 α):
    - 기존 `accumulation_reasons: list[ReasonCode]` 필드 제거
    - `reason_code_entries: list[ReasonCodeEntry]`로 통합 (Phase 8-A~8-E 모두 누적)
    - Phase 8-B-1 commit 시 pipeline.py backport 누락 catch 해소

    `eq=False`: ndarray + Optional[PoseLandmarks] 멤버 — dataclass 자동 __eq__
    bool ambiguous error 회피.

    Attributes:
        video_meta: 영상 메타데이터 (해상도, fps, duration 등).
        frame_results: list[FrameVisibilityResult] (Phase 4-A §5-1 누적).
            pose_not_detected frame은 본 list에 포함 X — 모듈 경계.
        landmarks_series: list[PoseLandmarks | None] (Phase 8-H 신규).
            전체 영상 frame 시퀀스 — pose 미검출은 None. Phase 5 metrics + Phase
            8-C/8-D/8-E 입력. Phase 8-C anchor 해소.
        ic_results: list[ICResult] (Phase 5-B-2 산출). compute_ic_indices 결과.
            Phase 8-D anchor 해소 (`evaluate_ic_validation` 입력 ic_confidences 도출).
        trunk_lean_results: list[TrunkLeanResult] (Phase 5-A 산출).
            Phase 8-D anchor 해소 (trunk_window_valid_ratios 도출) + Phase 8-I
            FeedbackContext 입력.
        knee_flexion_results: list[KneeFlexionResult] (Phase 5-C 산출).
        foot_strike_results: list[FootStrikeResult] (Phase 5-D 산출).
        reason_code_entries: list[ReasonCodeEntry] (Phase 8-A~8-E 누적 통합).
            §5-1 visibility + §5-2 body + §5-3 foot_cutoff + §5-4 side_view +
            §5-5 camera_unstable + §5-6 metric_variability + §5-7 IC validation +
            §4 tracking_stability. Phase 8-I `compute_response_status` 입력.
        pose_landmarks_count: pose_detected=True frame 수 (= len(frame_results)).
        pose_not_detected_count: pose 미검출 frame 수.
    """

    video_meta: VideoMeta
    frame_results: list[FrameVisibilityResult] = field(default_factory=list)
    landmarks_series: list[Optional[PoseLandmarks]] = field(default_factory=list)
    ic_results: list[ICResult] = field(default_factory=list)
    trunk_lean_results: list[TrunkLeanResult] = field(default_factory=list)
    knee_flexion_results: list[KneeFlexionResult] = field(default_factory=list)
    foot_strike_results: list[FootStrikeResult] = field(default_factory=list)
    reason_code_entries: list[ReasonCodeEntry] = field(default_factory=list)
    analysis_result: Optional[AnalysisResultMessage] = None
    """Phase 8-I lock 8-I-3 γ — AnalysisResultMessage 응답 조립 결과.

    Pipeline.run_on_video_file 마지막 단계에서 `build_analysis_result` 호출 →
    본 필드 채움. 디버깅 자산 (frame_results / landmarks_series 등) + 응답
    (analysis_result) 둘 다 보존 (lock 8-I-3 γ).
    """
    pose_landmarks_count: int = 0
    pose_not_detected_count: int = 0


# ============================================================
# Pipeline class — 모듈 오케스트레이션
# ============================================================


class Pipeline:
    """모듈 오케스트레이션 — pose_extractor + quality_gate 조립.

    인스턴스 1개 원칙 (PoseExtractor 세션당 1개, Day 4 학습). `__init__`에서
    PoseExtractor 1개 생성, run 호출마다 재사용. 세션 종료 시 `close()` 호출
    또는 **with-as 컨텍스트 매니저 사용 권장** (Phase 6-B decision 6):

        with Pipeline(cfg) as p:
            result = p.run_on_video_file(path, analysis_side="left")

    Phase 단계:
    - Phase 6-A: __init__ + _extract_and_evaluate_one_frame 헬퍼
    - Phase 6-B (본 단계): __enter__/__exit__ + run_on_video_file 본체
    - Phase 6-C: 통합 sanity end-to-end (scripts/sanity/)

    cfg 정책: AppConfig 통째 주입 (DI). 본 Phase에서 사용 범위:
    - cfg.mediapipe_pose -> PoseExtractor init
    - cfg.visibility_check -> evaluate_frame_visibility (헬퍼 내부)
    metrics/feedback 추가 Phase 진입 시 본 docstring 갱신.
    """

    def __init__(self, cfg: AppConfig) -> None:
        """초기화 — PoseExtractor 1개 생성.

        Args:
            cfg: AppConfig (DI). mediapipe_pose / visibility_check 사용.

        Raises:
            FileNotFoundError: PoseExtractor 모델 파일(.task) 부재.
            RuntimeError: PoseExtractor 초기화 실패.
        """
        self._cfg = cfg
        self._pose_extractor = PoseExtractor(cfg.mediapipe_pose)

    def close(self) -> None:
        """세션 종료 — PoseExtractor 내부 landmarker close.

        예외 swallow + logger.exception — 메인 분석 흐름 보호. with-as 패턴
        사용 시 `__exit__`가 본 메서드를 자동 호출 (Phase 6-B decision 6).
        """
        try:
            self._pose_extractor._landmarker.close()
        except Exception:
            logger.exception("Pipeline.close 예외 (swallow)")

    def __enter__(self) -> "Pipeline":
        """with-as 컨텍스트 진입 — self 반환."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """with-as 컨텍스트 종료 — `close()` 자동 호출 (예외 swallow 정책 그대로)."""
        self.close()

    def _extract_and_evaluate_one_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        analysis_side: Literal["left", "right"],
    ) -> tuple[Optional[FrameVisibilityResult], Optional[PoseLandmarks]]:
        """단일 frame 처리 — PoseExtractor + evaluate_frame_visibility (Phase 8-H 확장).

        ⚠️ Phase 8-H lock 8-H-4 α — tuple 반환 (FrameVisibilityResult + PoseLandmarks):
        호출자(run_on_video_file)가 landmarks_series 누적 — Phase 5 metrics + Phase
        8-C/8-D/8-E 입력.

        pose 미검출 시: `(None, None)` 반환. pose 검출 + visibility 평가 시:
        `(FrameVisibilityResult, PoseLandmarks)`.

        Args:
            frame: BGR np.ndarray.
            timestamp_ms: 호출자 부여 timestamp.
            analysis_side: 'left' 또는 'right'.

        Returns:
            tuple[Optional[FrameVisibilityResult], Optional[PoseLandmarks]]:
            - pose 미검출: (None, None)
            - pose 검출: (FrameVisibilityResult, PoseLandmarks)
        """
        pl = self._pose_extractor.process_frame(frame, timestamp_ms)
        if pl is None:
            return None, None
        fvr = evaluate_frame_visibility(
            pl, analysis_side=analysis_side, cfg=self._cfg.visibility_check
        )
        return fvr, pl

    def _accumulate_one_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        analysis_side: Literal["left", "right"],
        *,
        frame_results: list[FrameVisibilityResult],
        landmarks_series: list[Optional[PoseLandmarks]],
        body_inclusion_results: list[FrameGeometryResult],
        foot_cutoff_results: list[FrameGeometryResult],
        side_view_results: list[FrameSideViewResult],
    ) -> bool:
        """단일 frame 추출·평가 → 5개 frame-level 누적 컨테이너에 in-place append.

        ⚠️ Phase WS-A-2 lock — 기존 ``run_on_video_file`` frame loop(§1)의 frame당
        처리 블록을 본 메서드로 추출(behavior-preserving). 배치 모드와 스트림 모드
        (``StreamPipeline.push_frame``, docs/2-4-2 §7-3)가 동일 frame 누적 로직을
        공유하기 위함. ``_accumulate``(§2~§9, Phase WS-A-1 추출)와 짝.

        pose 미검출(또는 frame 처리 예외) 시 ``landmarks_series``에 ``None``을
        append하고 ``False``를 반환한다. 호출자는 ``False`` 수신 시
        ``pose_not_detected_count``를 1 증가시킨다.

        Args:
            frame: BGR np.ndarray.
            timestamp_ms: MediaPipe stream 모드 timestamp (호출자가 엄격 증가 보장).
            analysis_side: 'left' 또는 'right'.
            frame_results: §5-1 visibility 누적 컨테이너 (in-place append).
            landmarks_series: 전체 frame landmark 시퀀스 (pose 미검출은 None).
            body_inclusion_results: §5-2 body 포함 누적 컨테이너.
            foot_cutoff_results: §5-3 발 잘림 누적 컨테이너.
            side_view_results: §5-4 측면 구도 누적 컨테이너.

        Returns:
            pose 검출 + visibility 평가 성공 시 True, 그 외(미검출·예외) False.
        """
        try:
            fvr, pl = self._extract_and_evaluate_one_frame(
                frame, timestamp_ms, analysis_side
            )
        except Exception:
            logger.exception("_accumulate_one_frame: frame 처리 예외 (swallow, skip)")
            landmarks_series.append(None)
            return False

        # landmarks_series 누적 (Phase 8-C anchor 해소)
        landmarks_series.append(pl)
        if fvr is None:
            return False

        frame_results.append(fvr)

        # §5-2 body 포함 + §5-3 발 잘림 (lock 8-H-8)
        body_inclusion_results.append(
            evaluate_frame_body_inclusion(pl, self._cfg.visibility_check)
        )
        foot_cutoff_results.append(
            evaluate_frame_foot_cutoff(pl, analysis_side, self._cfg.visibility_check)
        )
        # §5-4 측면 구도 (lock 8-H-9)
        side_view_results.append(
            evaluate_frame_side_view(pl, self._cfg.side_view)
        )
        return True

    def _compute_visibility_per_frame(
        self, landmarks_series: list[Optional[PoseLandmarks]]
    ) -> list[float]:
        """각 frame의 주요 12 LandmarkPair visibility 평균 (Phase 8-E §4 입력).

        ⚠️ §5-1 overall_avg 패턴 일관 (12점 평균, lock 8-E-5 δ + 8-H-11).
        None frame은 visibility=0.0 처리 (target_lost 시그널 자연).
        """
        out: list[float] = []
        for pl in landmarks_series:
            if pl is None:
                out.append(0.0)
                continue
            vis_sum = (
                pl.shoulder.left.visibility + pl.shoulder.right.visibility
                + pl.hip.left.visibility + pl.hip.right.visibility
                + pl.knee.left.visibility + pl.knee.right.visibility
                + pl.ankle.left.visibility + pl.ankle.right.visibility
                + pl.heel.left.visibility + pl.heel.right.visibility
                + pl.foot_index.left.visibility + pl.foot_index.right.visibility
            )
            out.append(vis_sum / 12.0)
        return out

    @staticmethod
    def _extract_ic_confidences(
        ic_results: list[ICResult],
    ) -> list[ICConfidence]:
        """ICResult list → list[ICConfidence] (Phase 8-D evaluate_ic_validation 입력)."""
        return [r.confidence for r in ic_results]

    @staticmethod
    def _extract_trunk_window_ratios(
        trunk_lean_results: list[TrunkLeanResult],
    ) -> list[float]:
        """TrunkLeanResult.window_valid_count / window_total_count 산출 (Phase 8-D 입력).

        ⚠️ window_total_count=0 (영상 경계) 가드: ratio=0.0 fallback (Phase 8-D
        catch 7-4 정합).
        """
        out: list[float] = []
        for r in trunk_lean_results:
            if r.window_total_count <= 0:
                out.append(0.0)  # ZeroDivisionError 가드
            else:
                out.append(r.window_valid_count / r.window_total_count)
        return out

    def _accumulate(
        self,
        video_meta: VideoMeta,
        analysis_side: Literal["left", "right"],
        direction: Literal["left_to_right", "right_to_left"],
        frame_results: list[FrameVisibilityResult],
        landmarks_series: list[Optional[PoseLandmarks]],
        body_inclusion_results: list[FrameGeometryResult],
        foot_cutoff_results: list[FrameGeometryResult],
        side_view_results: list[FrameSideViewResult],
        pose_not_detected_count: int,
    ) -> PipelineResult:
        """frame loop 산출물 → 누적 평가 + Phase 5 metrics + 응답 조립.

        ⚠️ Phase WS-A-1 lock — 기존 ``run_on_video_file``의 §2~§9 블록을 본
        메서드로 추출(behavior-preserving). 배치 모드(``run_on_video_file``)와
        스트림 모드(``StreamPipeline.finalize``, Phase WS-A-2 예정)가 동일 누적
        로직을 공유하기 위함. 동작 불변 — 기존 회귀 테스트 그대로 통과해야 함.

        호출자는 §1 frame loop로 아래 5개 frame-level 산출물을 누적해 전달한다
        (배치: ``iter_frames`` 루프 / 스트림: ``push_frame`` 증분 누적).

        Args:
            video_meta: 영상 메타데이터. fps_safe 재계산 근거.
            analysis_side: 'left' 또는 'right'.
            direction: 'left_to_right' 또는 'right_to_left'.
            frame_results: pose 검출 frame의 §5-1 visibility 결과.
            landmarks_series: 전체 frame landmark 시퀀스 (pose 미검출은 None).
            body_inclusion_results: §5-2 frame-level body 포함 결과.
            foot_cutoff_results: §5-3 frame-level 발 잘림 결과.
            side_view_results: §5-4 frame-level 측면 구도 결과.
            pose_not_detected_count: pose 미검출 frame 수.

        Returns:
            PipelineResult — analysis_result 포함 전 필드 채움.
        """
        # fps_safe: 호출자 중복 계산 회피 — video_meta에서 재도출 (run_on_video_file
        # §0과 동일 식). 영상 경계/손상 시 30fps fallback.
        fps_safe = video_meta.fps if video_meta.fps > 1e-6 else 30.0

        # ── 2. §5-1/§5-2/§5-3/§5-4 누적 평가 → reason_code_entries ──
        reason_code_entries: list[ReasonCodeEntry] = []
        reason_code_entries.extend(
            evaluate_visibility_accumulation(
                frame_results, self._cfg.visibility_check
            )
        )
        reason_code_entries.extend(
            evaluate_body_inclusion_accumulation(
                body_inclusion_results, self._cfg.visibility_check
            )
        )
        reason_code_entries.extend(
            evaluate_foot_cutoff_accumulation(
                foot_cutoff_results, self._cfg.visibility_check
            )
        )
        reason_code_entries.extend(
            evaluate_side_view_accumulation(
                side_view_results, self._cfg.side_view
            )
        )

        # ── 3. Phase 5 metrics (lock 8-H-10) ──
        ic_results: list[ICResult] = compute_ic_indices(
            landmarks_series, analysis_side, self._cfg.ic
        )
        ic_indices = [r.frame_index for r in ic_results]

        trunk_lean_results: list[TrunkLeanResult] = trunk_lean.compute_at_ic(
            landmarks_series, ic_indices, self._cfg.trunk_lean
        )
        knee_flexion_results: list[KneeFlexionResult] = knee_flexion.compute_at_ic(
            landmarks_series, ic_indices, analysis_side, self._cfg.knee_flexion
        )
        foot_strike_results: list[FootStrikeResult] = foot_strike.compute_at_ic(
            landmarks_series,
            ic_indices,
            analysis_side,
            direction,
            self._cfg.foot_strike,
        )

        # ── 4. Phase 8-C §5-5 camera_unstable (lock 8-H-10) ──
        reason_code_entries.extend(
            evaluate_camera_stability(
                landmarks_series, ic_indices, self._cfg.stride_exclusion
            )
        )

        # ── 5. Phase 8-C §5-6 metric_variability ──
        foot_degs = [r.deg for r in foot_strike_results]
        knee_degs = [r.deg for r in knee_flexion_results]
        trunk_degs = [r.deg for r in trunk_lean_results]
        reason_code_entries.extend(
            evaluate_metric_variability(
                foot_degs, knee_degs, trunk_degs, self._cfg.variability
            )
        )

        # ── 6. Phase 8-D §5-7 IC 검증 ──
        ic_confidences = self._extract_ic_confidences(ic_results)
        trunk_window_valid_ratios = self._extract_trunk_window_ratios(
            trunk_lean_results
        )
        reason_code_entries.extend(
            evaluate_ic_validation(
                ic_confidences, trunk_window_valid_ratios, self._cfg.ic_validation
            )
        )

        # ── 7. Phase 8-E §4 추적 안정성 (lock 8-H-11) ──
        visibility_per_frame = self._compute_visibility_per_frame(landmarks_series)
        reason_code_entries.extend(
            evaluate_tracking_stability(
                visibility_per_frame, fps_safe, self._cfg.tracking
            )
        )

        # ── 8. PipelineResult 조립 (확장 schema) ──
        result = PipelineResult(
            video_meta=video_meta,
            frame_results=frame_results,
            landmarks_series=landmarks_series,
            ic_results=ic_results,
            trunk_lean_results=trunk_lean_results,
            knee_flexion_results=knee_flexion_results,
            foot_strike_results=foot_strike_results,
            reason_code_entries=reason_code_entries,
            pose_landmarks_count=len(frame_results),
            pose_not_detected_count=pose_not_detected_count,
        )

        # ── 9. Phase 8-I 응답 조립 (lock 8-I-3 γ — PipelineResult.analysis_result 필드) ──
        result.analysis_result = build_analysis_result(result, analysis_side)
        return result

    def run_on_video_file(
        self,
        video_path: Path,
        analysis_side: Literal["left", "right"],
        max_frames: Optional[int] = None,
        direction: Literal["left_to_right", "right_to_left"] = "left_to_right",
    ) -> PipelineResult:
        """file path mode end-to-end (Phase 8-H 확장 — Phase 5 metrics + Phase 8-A~8-E 통합).

        ⚠️ Phase 8-H 통합 흐름:
        1. video_meta + frame loop (Phase 5/8 통합 산출물 누적)
        2. §5-1/§5-2/§5-3/§5-4 frame-level + 누적
        3. Phase 5 metrics: compute_ic_indices + trunk_lean/knee_flexion/foot_strike compute_at_ic
        4. Phase 8-C §5-5 evaluate_camera_stability
        5. Phase 8-C §5-6 evaluate_metric_variability
        6. Phase 8-D §5-7 evaluate_ic_validation
        7. Phase 8-E §4 evaluate_tracking_stability (visibility_per_frame helper)
        8. reason_code_entries 누적 → PipelineResult 조립

        ⚠️ Phase 8-I (응답 조립) scope:
        compute_response_status + compute_feedback_messages + AnalysisResultMessage 조립.
        본 8-H는 reason_code_entries 누적까지만.

        ⚠️ direction (lock catch 7-12): 'left_to_right' default. jaemin 영상 기준.
        Phase 9 자동 결정 anchor (docs §3-1 1.5초 AND 30 frame 정책).

        ⚠️ analysis_side (lock 8-H-6): 호출자 입력 그대로 (현재 CLI default 'left',
        Phase 9 자동 결정 anchor).

        Args:
            video_path: 영상 파일 경로.
            analysis_side: 'left' 또는 'right'.
            max_frames: 처리할 최대 frame 수 (None이면 full run).
            direction: 'left_to_right' 또는 'right_to_left' (foot_strike compute_at_ic 입력).

        Returns:
            PipelineResult — 확장 schema 전부 채움.

        Raises:
            FileNotFoundError: video_path 미존재.
            ValueError: max_frames <= 0.

        견고성 가드:
        - max_frames 음수/0 → ValueError
        - fps ≤ 1e-6 → 30fps fallback
        - frame 단위 예외 → logger.exception + skip
        - Phase 5/8 함수 호출 예외 → 각 함수 내부 failed-safe (logger.exception 적용)
        """
        if max_frames is not None and max_frames <= 0:
            raise ValueError(
                f"max_frames must be > 0 or None, got {max_frames}"
            )

        video_meta = get_video_meta(video_path)
        fps_safe = video_meta.fps if video_meta.fps > 1e-6 else 30.0

        frame_results: list[FrameVisibilityResult] = []
        landmarks_series: list[Optional[PoseLandmarks]] = []
        body_inclusion_results: list[FrameGeometryResult] = []
        foot_cutoff_results: list[FrameGeometryResult] = []
        side_view_results: list[FrameSideViewResult] = []
        pose_not_detected_count = 0

        # ── 1. frame loop (Phase WS-A-2: _accumulate_one_frame 위임) ──
        # ⚠️ frame당 처리 블록을 _accumulate_one_frame으로 추출 — 배치/스트림 공용.
        for idx, frame in enumerate(iter_frames(video_path)):
            if max_frames is not None and idx >= max_frames:
                break
            timestamp_ms = int(idx * 1000.0 / fps_safe)
            pose_detected = self._accumulate_one_frame(
                frame,
                timestamp_ms,
                analysis_side,
                frame_results=frame_results,
                landmarks_series=landmarks_series,
                body_inclusion_results=body_inclusion_results,
                foot_cutoff_results=foot_cutoff_results,
                side_view_results=side_view_results,
            )
            if not pose_detected:
                pose_not_detected_count += 1

        # ── 2~9. 누적 평가 + Phase 5 metrics + 응답 조립 (Phase WS-A-1: _accumulate 위임) ──
        # ⚠️ 기존 §2~§9 인라인 블록을 _accumulate로 추출 — 배치/스트림 공용 로직.
        return self._accumulate(
            video_meta=video_meta,
            analysis_side=analysis_side,
            direction=direction,
            frame_results=frame_results,
            landmarks_series=landmarks_series,
            body_inclusion_results=body_inclusion_results,
            foot_cutoff_results=foot_cutoff_results,
            side_view_results=side_view_results,
            pose_not_detected_count=pose_not_detected_count,
        )
