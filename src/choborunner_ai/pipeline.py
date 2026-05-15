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
- Phase 6-A (본 단계): PipelineResult + Pipeline.__init__ + _extract_and_evaluate_one_frame
- Phase 6-B: Pipeline.run_on_video_file 본체
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
from typing import Literal, Optional

import numpy as np

from choborunner_ai.config import AppConfig
from choborunner_ai.pose_extractor import PoseExtractor
from choborunner_ai.quality_gate import (
    FrameVisibilityResult,
    ReasonCode,
    evaluate_frame_visibility,
)
from choborunner_ai.video_preprocessor import VideoMeta

logger = logging.getLogger(__name__)


# ============================================================
# PipelineResult — file path mode 전용 결과 객체
# ============================================================


@dataclass(eq=False)
class PipelineResult:
    """file path mode 결과 객체.

    WebSocket binary stream mode는 별도 결과 객체 — Phase 8 진입 시 결정.
    `eq=False`: ndarray 멤버는 현재 없지만 dataclass 일관성 + 향후 frame snapshot
    추가 대비.

    Attributes:
        video_meta: 영상 메타데이터 (해상도, fps, duration 등 — video_preprocessor SoT).
        frame_results: list[FrameVisibilityResult] (Phase 4-A 결과 누적).
            pose_not_detected frame은 본 list에 포함 X — 모듈 경계, docs §4-2
            "후속 단계가 분석 제외" 정책 정합.
        accumulation_reasons: list[ReasonCode] (Phase 4-B 결과). 빈 list = 통과.
        pose_landmarks_count: pose_detected=True frame 수 (디버깅 자산).
            = len(frame_results) (pose 미검출 frame 제외).
        pose_not_detected_count: pose 미검출 frame 수 (디버깅 자산, docs §4-2
            누적이 별도 reason code trigger 가능성 — 별도 Phase에서 활용).
    """

    video_meta: VideoMeta
    frame_results: list[FrameVisibilityResult] = field(default_factory=list)
    accumulation_reasons: list[ReasonCode] = field(default_factory=list)
    pose_landmarks_count: int = 0
    pose_not_detected_count: int = 0


# ============================================================
# Pipeline class — 모듈 오케스트레이션
# ============================================================


class Pipeline:
    """모듈 오케스트레이션 — pose_extractor + quality_gate 조립.

    인스턴스 1개 원칙 (PoseExtractor 세션당 1개, Day 4 학습). `__init__`에서
    PoseExtractor 1개 생성, run 호출마다 재사용. 세션 종료 시 `close()` 명시 호출.

    Phase 단계:
    - Phase 6-A (본 단계): __init__ + _extract_and_evaluate_one_frame 헬퍼
    - Phase 6-B: run_on_video_file 본체 (iter_frames + accumulation)
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
        도입은 Phase 6-B 이후 검토.
        """
        try:
            self._pose_extractor._landmarker.close()
        except Exception:
            logger.exception("Pipeline.close 예외 (swallow)")

    def _extract_and_evaluate_one_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
        analysis_side: Literal["left", "right"],
    ) -> Optional[FrameVisibilityResult]:
        """단일 frame 처리 — PoseExtractor.process_frame + evaluate_frame_visibility.

        pose 미검출 시 (PoseLandmarks=None) **None 반환** — 호출자가 skip 결정
        (docs §4-2 "후속 단계가 분석 제외" 정합). 본 frame은 누적 frame_results
        에 포함 X. pose_not_detected 누적 카운트는 호출자(run_on_video_file)
        가 별도 관리.

        Args:
            frame: BGR np.ndarray (video_preprocessor 출력 형식).
            timestamp_ms: 호출자 부여 timestamp (option C, monotonic + unique 보장).
            analysis_side: 'left' 또는 'right'.

        Returns:
            FrameVisibilityResult (4 카테고리 평가 결과) 또는 None (pose 미검출).
        """
        pl = self._pose_extractor.process_frame(frame, timestamp_ms)
        if pl is None:
            return None
        return evaluate_frame_visibility(
            pl, analysis_side=analysis_side, cfg=self._cfg.visibility_check
        )
