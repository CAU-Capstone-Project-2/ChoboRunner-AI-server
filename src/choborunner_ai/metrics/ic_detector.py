"""docs/2-3-4 §4 Initial Contact 검출 — 정식판 (Phase 5-B).

본 모듈은 docs/2-3-4 §4 (Initial Contact 탐지 로직) 2단계 hybrid 구조 단일
정답 구현.

Phase 5-B 작업 단위:
- Phase 5-B-1: Stage 1 단독 (Zeni 2008, heel_rel_x local max)
- Phase 5-B-2: Stage 2 추가 (Fellin 2010, foot v_y zero-crossing) + confidence
- Phase 5-B-3 (본 단계): batch compute_ic_indices wrapper + pytest 회귀
  → Day 5 decision i γ 완성 (stream API + batch wrapper 둘 다)

학술 알고리즘:
- Zeni, J. A. et al. (2008). Gait & Posture, 27(4), 710-714.
- Fellin, R. E. et al. (2010). J Sci Med Sport, 13(6), 646-650.
- Bonci, T. et al. (2022). Front Bioeng Biotechnol, 10, 868928.

Stage 1 (Zeni 2008) — heel_rel_x local max (Phase 5-B-1 본 단계):
    pelvis_x(t) = (left_hip.x(t) + right_hip.x(t)) / 2
    heel_rel_x(t) = heel.x(t) - pelvis_x(t)
    t_candidate = argmax heel_rel_x  (local maximum)

- 의도: 발이 가장 앞으로 뻗은 순간 = "곧 닿을 시점"
- walking 검증, running 30~100ms 오차 (Fellin 2010, Milner 2015)
- 부록 D 합성 sanity: MAE 33.3ms, Median 33.3ms, 검출률 100%
  (heel forward peak가 실제 IC보다 ~1 frame lead)

Stage 2 (Fellin 2010) — foot v_y 양→음 zero-crossing (Phase 5-B-2 본 단계):
    v_y_heel(t) = (heel.y(t) - heel.y(t-1)) * fps
    v_y_foot(t) = (foot_index.y(t) - foot_index.y(t-1)) * fps
    IC = v_y가 양→음 전환 시점 (하강 멈추고 반발/상승 진입)

- **heel + foot_index 둘 다 추적, 더 일찍 부호 전환 시점 채택** (docs §4-2 본문 정합)
  → RFS / MFS / FFS 모두 강건 (Milner 2015, Handsaker 2016)
- 이산 채택 규칙: v_y[t] > 0 AND v_y[t+1] ≤ 0 만족 가장 이른 t → IC = t+1
- 부록 D hybrid 결과: MAE 9.6ms (-71.2% 개선 vs Stage 1 단독 33.3ms)
  Fellin 2010 RMSE ≤ 8ms 자릿수 일치

⚠️ 부록 B 의사코드 catch (Day 6 학습 자산):
- catch ①: 부록 B 의사코드는 foot_y만 사용 (단순화). docs §4-2 본문은 heel + foot
  둘 다 추적 명시. 정식 구현은 docs 본문 정합 (decision i β, CLAUDE.md §4 단일 정답).
- catch ②: 부록 B _find_pos_to_neg는 int 반환 (`return i`)이나 사용처는
  `min(crossings)` list 가정 → 로직 오류. docs §4-2 "가장 이른 t" 정합 위해
  list 반환으로 정정 (decision ii α).
- decision vi (Phase 5-B-1) 일관성: "부록 B 의사코드 직접 구현"은 buffer 구조
  1:1 의도. 알고리즘은 docs 본문 단일 정답 — decision vi 위반 X.

좌표계 (docs §3-2): x 좌→우, y 화면 아래 + (MediaPipe convention).
- y 증가 = 발 하강 (v_y > 0)
- y 감소 = 발 상승 (v_y < 0)

부록 B 의사코드 핵심 (Phase 5-B-1 직접 구현):
    deque buffer (maxlen=15) — 최근 frame heel_rel_x / heel_y / foot_y / visibility
    on_new_frame(frame_idx, landmarks):
      1. buffer 적재 (pelvis_x, heel_rel_x, heel_y, foot_y, visibility)
      2. warmup: len(buffer) < LOOKAHEAD*2+1 (=7) → return None
      3. Stage 1: check_pos = len(buffer) - LOOKAHEAD - 1
         _is_local_max(heel_rel_arr, check_pos, LOOKAHEAD) 검증
         visibility >= visibility_min
         frame_idx - last_ic_frame >= MIN_IC_INTERVAL
      4. Phase 5-B-1: Stage 1 candidate 그대로 IC, confidence='low'
         5-B-2: Stage 2 정밀화 (foot v_y zero-crossing) + high/medium/low 분류

Phase 5-B-1 결정 사항:
- (i)   stream + batch 둘 다 (batch wrapper는 5-B-3 추가)
- (ii)  dataclass ICResult
- (iii) analysis_side: Literal["left","right"] 입력
- (vi)  부록 B 의사코드 직접 구현 (demo2 재사용 X, 학습 모드 한정)
        5-C / 5-D 압축 모드는 재검토 여지

visibility_min 정책:
- 부록 B 의사코드는 VISIBILITY_TH=0.6 인자 형태 → 본 구현도 함수 인자.
- ICDetectorConfig 신규 필드 추가 검토는 5-B-2 또는 별도 결정.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

from choborunner_ai.config import ICDetectorConfig
from choborunner_ai.pose_extractor import PoseLandmarks

logger = logging.getLogger(__name__)


# ============================================================
# IC 신뢰도 Literal (docs §4-3)
# ============================================================


ICConfidence = Literal["high", "medium", "low"]
"""docs/2-3-4 §4-3 IC 신뢰도 3 분류.

- 'high'  : Stage 2 zero-crossing 성공 + Stage 1 offset 절댓값 ≤ 2 frame
- 'medium': Stage 2 zero-crossing 성공 + Stage 1 offset 절댓값 > 2 frame
- 'low'   : Stage 2 zero-crossing 실패 (Stage 1 그대로 사용)
            → docs §4-3: stride 누적 통계 제외 (가중치 0)

Phase 5-B-2 (본 단계): Stage 2 추가로 high/medium/low 정상 분류 작동.
임계 inline 2 frame (decision iii, 부록 B 의사코드 1:1, 향후 cfg 이동 검토).
"""


# ============================================================
# IC 결과 dataclass (Day 5 decision ii)
# ============================================================


@dataclass
class ICResult:
    """IC 검출 1건 결과 (docs §4-3 신뢰도 포함).

    Attributes:
        frame_index: IC 시점 frame 절대 인덱스 (영상 시작 0).
        confidence: 'high' / 'medium' / 'low' (§4-3 단일 정답).
        stage1_offset: Stage 1 후보 대비 Stage 2 정밀화 offset (frame 수, signed).
            Stage 2 미수행(low) 또는 Phase 5-B-1 단독은 0.
    """

    frame_index: int
    confidence: ICConfidence
    stage1_offset: int


# ============================================================
# ICDetector class — stream API (Phase 5-B-1: Stage 1 단독)
# ============================================================


class ICDetector:
    """2단계 hybrid IC detector — stream API (docs/2-3-4 §4 + 부록 B 의사코드).

    Phase 5-B-2 (본 단계): Stage 1+2 hybrid 완성.
    - Stage 1 (Zeni 2008): heel_rel_x local max 후보 검출
    - Stage 2 (Fellin 2010): heel + foot v_y zero-crossing 정밀화 + confidence 분류

    stream API: on_new_frame(frame_idx, pl, analysis_side) → Optional[ICResult].
    호출자가 매 frame 호출, IC 검출 시에만 ICResult 반환.

    인스턴스 1개 원칙 — buffer / last_ic_frame 상태 보유.
    Batch wrapper (Phase 5-B-3): compute_ic_indices(landmarks_series, ...).

    부록 B 의사코드 흐름 (docs §4-2 본문 정합으로 정정):
        1. buffer 적재
        2. warmup
        3. Stage 1 (heel_rel_x local max + visibility + interval 가드)
        4. Stage 2 정밀화 (heel + foot 둘 다 v_y zero-crossing, 더 일찍 채택)
           + confidence 분류 (high/medium/low, docs §4-3)

    Args:
        cfg: ICDetectorConfig — fps / buffer_size / lookahead / refine_window /
            min_ic_interval (§4 + 부록 C 정합).
        visibility_min: Stage 1 visibility 가드 임계 (default 0.6, 부록 B
            VISIBILITY_TH 정합). ⚠️ ICDetectorConfig 신규 필드 검토 5-B-2 또는
            별도.
    """

    def __init__(
        self, cfg: ICDetectorConfig, visibility_min: float = 0.6
    ) -> None:
        self._cfg = cfg
        self._visibility_min = visibility_min
        # buffer: deque of dict (frame_idx, heel_rel_x, heel_y, foot_y, visibility)
        self._buffer: deque[dict] = deque(maxlen=cfg.buffer_size)
        # 최초 IC 검출 시 MIN_IC_INTERVAL 통과 보장 (-10^9 sentinel)
        self._last_ic_frame: int = -(10**9)

    def on_new_frame(
        self,
        frame_idx: int,
        pl: PoseLandmarks,
        analysis_side: Literal["left", "right"],
    ) -> Optional[ICResult]:
        """단일 frame 입력 → IC 검출 (Phase 5-B-2: Stage 1+2 hybrid).

        부록 B 의사코드 흐름 (docs §4-2 본문 정합 정정):
        1. buffer 적재 (pelvis_x, heel_rel_x, heel_y, foot_y, visibility)
        2. warmup: len(buffer) < LOOKAHEAD*2+1 → None
        3. Stage 1: check_pos = len(buffer) - LOOKAHEAD - 1
           - _is_local_max(heel_rel_arr, check_pos, LOOKAHEAD)
           - visibility >= visibility_min
           - frame_idx - last_ic_frame >= MIN_IC_INTERVAL
        4. Stage 2: search window ±REFINE_WINDOW
           - heel v_y / foot v_y 둘 다 양→음 zero-crossing 검색
           - 둘 중 더 일찍 부호 전환 시점 채택 (docs §4-2 본문 정합)
           - 둘 다 빈 → low confidence + Stage 1 그대로
        5. confidence 분류: |offset| ≤ 2 → high, > 2 → medium, Stage 2 실패 → low

        Args:
            frame_idx: 절대 frame 인덱스 (영상 시작 0).
            pl: PoseLandmarks (Phase 3, 6 LandmarkPair).
            analysis_side: 'left' 또는 'right' (Day 5 decision iii).

        Returns:
            ICResult 또는 None. IC 검출 frame에만 반환.

        Raises:
            ValueError: analysis_side가 'left'/'right' 아닌 경우 (런타임 가드).
        """
        if analysis_side not in ("left", "right"):
            raise ValueError(
                f"analysis_side는 'left' 또는 'right'만 허용, got {analysis_side!r}"
            )

        # ── 1. buffer 적재 ─────────────────────────────────
        pelvis_x = (pl.hip.left.x + pl.hip.right.x) / 2.0
        side_heel = pl.heel.left if analysis_side == "left" else pl.heel.right
        side_foot = (
            pl.foot_index.left if analysis_side == "left" else pl.foot_index.right
        )

        # visibility: heel + foot_index 최소값 (부록 B 의사코드 정합)
        vis = min(side_heel.visibility, side_foot.visibility)

        self._buffer.append(
            {
                "frame_idx": frame_idx,
                "heel_rel_x": side_heel.x - pelvis_x,
                "heel_y": side_heel.y,
                "foot_y": side_foot.y,
                "visibility": vis,
            }
        )

        # ── 2. warmup ─────────────────────────────────────
        # buffer가 LOOKAHEAD*2+1 이상 채워질 때까지 후보 검사 X (부록 B)
        if len(self._buffer) < self._cfg.lookahead * 2 + 1:
            return None

        # ── 3. Stage 1: heel_rel_x local max + 가드 ───────
        check_pos = len(self._buffer) - self._cfg.lookahead - 1
        candidate = self._buffer[check_pos]
        heel_rel_arr = [b["heel_rel_x"] for b in self._buffer]

        if not self._is_local_max(heel_rel_arr, check_pos, self._cfg.lookahead):
            return None
        if candidate["visibility"] < self._visibility_min:
            return None
        if (
            candidate["frame_idx"] - self._last_ic_frame
            < self._cfg.min_ic_interval
        ):
            return None

        # ── 4. Stage 2: Fellin 2010 — heel + foot v_y zero-crossing 정밀화 ──
        # search window: buffer[check_pos - REFINE_WINDOW : check_pos + REFINE_WINDOW + 1]
        start = max(0, check_pos - self._cfg.refine_window)
        end = check_pos + self._cfg.refine_window + 1
        search = [self._buffer[i] for i in range(start, end)]

        # heel + foot 두 v_y (docs §4-2 본문 정합, decision i β)
        # v_y[j] = (search[j+1].y - search[j].y) * fps, j ∈ [0, len(search)-2]
        v_y_heel = [
            (search[i]["heel_y"] - search[i - 1]["heel_y"]) * self._cfg.fps
            for i in range(1, len(search))
        ]
        v_y_foot = [
            (search[i]["foot_y"] - search[i - 1]["foot_y"]) * self._cfg.fps
            for i in range(1, len(search))
        ]

        # 양→음 zero-crossings (list 반환, decision ii α — 부록 B catch 정정)
        heel_crossings = self._find_pos_to_neg(v_y_heel)
        foot_crossings = self._find_pos_to_neg(v_y_foot)

        # 둘 중 더 일찍 부호 전환 시점 채택 (decision vi)
        heel_idx = min(heel_crossings) if heel_crossings else None
        foot_idx = min(foot_crossings) if foot_crossings else None

        # ── 5. IC 확정 + confidence 분류 ────────────────────
        if heel_idx is None and foot_idx is None:
            # Stage 2 실패 → low confidence + Stage 1 그대로 (decision iv, 부록 B 정합)
            ic_frame = candidate["frame_idx"]
            confidence: ICConfidence = "low"
            offset = 0
        else:
            # 둘 다 있으면 min, 한쪽만 있으면 그쪽 (decision vi: 더 일찍)
            if heel_idx is None:
                earliest_idx = foot_idx
            elif foot_idx is None:
                earliest_idx = heel_idx
            else:
                earliest_idx = min(heel_idx, foot_idx)
            # IC = 첫 비양수 프레임 (v_y[j+1]) → search 인덱스 j+1
            ic_idx_in_search = earliest_idx + 1
            ic_frame = search[ic_idx_in_search]["frame_idx"]
            offset = ic_frame - candidate["frame_idx"]
            # confidence (decision iii — 인라인 2, 부록 B 1:1)
            confidence = "high" if abs(offset) <= 2 else "medium"

        self._last_ic_frame = ic_frame
        return ICResult(
            frame_index=ic_frame,
            confidence=confidence,
            stage1_offset=offset,
        )

    @staticmethod
    def _is_local_max(arr: list[float], idx: int, window: int) -> bool:
        """arr[idx]가 ±window 범위에서 strict local maximum인지 (부록 B 정합).

        본 함수는 부록 B `_is_local_max` 의사코드 직접 구현 (Day 5 decision vi,
        학습 모드 한정). demo2 `find_local_maxima_indices` / `nms_peaks` 재사용
        X — Phase 5-C/5-D 압축 모드는 재검토 여지.

        Args:
            arr: heel_rel_x list (또는 임의 float series).
            idx: 검사 위치.
            window: ±window 범위 (cfg.lookahead).

        Returns:
            True: arr[idx]가 ±window 범위 안의 모든 값보다 strict greater.
            False: 경계 reach 또는 인접에 같거나 큰 값 존재.
        """
        if idx - window < 0 or idx + window >= len(arr):
            return False
        v = arr[idx]
        for off in range(1, window + 1):
            if arr[idx - off] >= v or arr[idx + off] >= v:
                return False
        return True

    @staticmethod
    def _find_pos_to_neg(velocities: list[float]) -> list[int]:
        """v_y[i] > 0 AND v_y[i+1] ≤ 0 만족 인덱스 list 반환 (docs §4-2 정합).

        부록 B 의사코드 로직 오류 정정 (decision ii, Day 6 학습 자산):
        - 부록 B 의사코드: `return i` int 반환 + 사용처 `min(crossings)` list
          가정 → 불일치 (i=0 매치 시 `if not crossings` falsy 오작동 + min 적용 불가)
        - 정정: 모든 crossings list 반환 + 호출자가 `min()` 적용
        - docs §4-2 본문 "가장 이른 t" 정합 = list 중 min

        IC 해석 (호출자):
        - earliest_idx = min(crossings) (가장 이른 양→음 전환)
        - ic_idx_in_search = earliest_idx + 1 (첫 비양수 프레임)

        Args:
            velocities: v_y list (heel 또는 foot vertical velocity).

        Returns:
            crossings list. 빈 list = zero-crossing 미발견 (low confidence trigger).
        """
        crossings: list[int] = []
        for i in range(len(velocities) - 1):
            if velocities[i] > 0 and velocities[i + 1] <= 0:
                crossings.append(i)
        return crossings


# ============================================================
# Batch wrapper (Phase 5-B-3, Day 5 decision i γ 완성)
# ============================================================


def compute_ic_indices(
    landmarks_series: list[Optional[PoseLandmarks]],
    analysis_side: Literal["left", "right"],
    cfg: ICDetectorConfig,
    visibility_min: float = 0.6,
) -> list[ICResult]:
    """Batch wrapper — `list[Optional[PoseLandmarks]]` → `list[ICResult]`.

    Stream API `on_new_frame`을 매 frame 호출, 누적 결과 반환. ICDetector
    인스턴스 1개 + 순회 패턴. Day 5 decision i γ 완성 (stream + batch 둘 다).

    Args:
        landmarks_series: `list[PoseLandmarks | None]` (decision i β).
            Pipeline.run_on_video_file 흐름 정합 (pose 미검출 frame None).
        analysis_side: 'left' 또는 'right' (decision iii).
        cfg: ICDetectorConfig.
        visibility_min: Stage 1 visibility 가드 임계 (default 0.6).

    Returns:
        `list[ICResult]` (decision ii α — rich: frame_index + confidence + offset).
        호출자가 frame_index만 추출 시 `[r.frame_index for r in results]`,
        `trunk_lean.compute_at_ic(landmarks_series, ic_indices=...)` 호환.

    None frame 처리 (decision iii α):
    - `if pl is None: continue` (buffer 미적재 skip)
    - frame_idx는 enumerate 절대 인덱스 유지 (시간 일관성)

    통합 sanity (trunk_lean.compute_at_ic 연결 등)는 본 Phase 5-B-3 scope
    밖 (decision v β) — 별도 Phase 5 통합 마일스톤 (5-E 또는 Pipeline 통합).

    Raises:
        ValueError: analysis_side가 'left'/'right' 아닐 시 (on_new_frame 가드 전파).
    """
    detector = ICDetector(cfg, visibility_min=visibility_min)
    results: list[ICResult] = []
    for idx, pl in enumerate(landmarks_series):
        if pl is None:
            continue
        r = detector.on_new_frame(idx, pl, analysis_side)
        if r is not None:
            results.append(r)
    return results
