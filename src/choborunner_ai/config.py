"""ChoboRunner AI Server — Pydantic Settings 기반 통합 설정.

설계문서 2-3-1~7의 통합 설정. 모든 임계값과 알고리즘 파라미터는
이 모듈 하나에서 관리한다 (CLAUDE.md §6, §8).

본 모듈에서 다루는 영역:
- 2-3-1 (입력 영상 메타데이터): InputMetadata
- 2-3-4 (자세 지표 산출 + IC hybrid 검출): Smoothing / FrameQuality /
  ICDetector / StrideAggregation / FootStrike / KneeFlexion / TrunkLean
- 2-3-5 (Pose 후 품질 검사 + 분석 상태값): VisibilityCheck / SideView /
  AnalysisSide / TrackingStability / StrideExclusion / MetricVariability /
  ICValidation

2-3-2/3/6/7 영역은 해당 docs 확정 후 추가한다.

참고문헌:
- Zeni, J. A., Richards, J. G., & Higginson, J. S. (2008).
  Two simple methods for determining gait events during treadmill and
  overground walking using kinematic data. Gait & Posture, 27(4), 710–714.
- Fellin, R. E., Rose, W. C., Royer, T. D., & Davis, I. S. (2010).
  Comparison of methods for kinematic identification of footstrike and toe-off
  during overground and treadmill running. J Sci Med Sport, 13(6), 646–650.
- Knorz, S., et al. (2017). Three-dimensional biomechanical analysis of
  rearfoot and forefoot running. J Vis Exp, (122), e54818.
- Teng, H. L., & Powers, C. M. (2014). The influence of trunk posture on
  lower extremity biomechanics during running. Med Sci Sports Exerc, 46(9),
  1739–1747.
- Bonci, T., et al. (2022). An algorithm for accurate marker-based gait event
  detection in healthy and pathological populations during complex motor
  tasks. Front Bioeng Biotechnol, 10, 868928.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ============================================================
# 2-3-1. 입력 영상 메타데이터 (input_validator.py)
# ============================================================


class InputMetadataConfig(BaseModel):
    """입력 영상 메타데이터 검증 임계 (docs/2-3-1 §3, §3-5).

    누적 검증 (분석 종료 시점): 누적 분석 시간 / Effective FPS / 누적 frame 수.
    즉시 검증 (첫 frame 시점): 해상도.
    분석 종료 trigger: timeout (reason code 아님).

    상태값 결정 우선순위는 `failed > low_confidence > success` (§3-2).
    duration_sec 또는 frame_count 중 하나라도 위반 시 `too_short` 트리거.

    ⚠️ 파일럿 보정 필요. 본 모듈에서 6번째 ⚠️ 클래스.

    ⚠️ 두 임계 특별 주의:
    - `effective_fps_*`: 네트워크 환경(LTE/5G/Wi-Fi)에 따라 frame drop으로
      effective fps 변동. nominal 30fps 캡처해도 effective는 18~22fps까지
      떨어질 수 있음. docs/2-3-1 §6 실험 계획 따라 보정 필요.
    - `analysis_end_timeout_sec`: 백엔드 WebSocket heartbeat interval보다
      짧으면 연결 살아있는데 분석 종료 오판정. 파일럿 후 1.5~3.0초 범위
      보정. docs/2-3-1 §3-5 참조.
    """

    duration_failed_sec: float = Field(
        default=5.0,
        ge=0.0,
        description="누적 분석 시간 failed 임계 (초). 미만 시 reason_code='too_short' (failed). duration 또는 frame_count 중 하나라도 위반 시 too_short. 출처: docs/2-3-1 §3-1.",
    )
    duration_low_confidence_sec: float = Field(
        default=10.0,
        ge=0.0,
        description="누적 분석 시간 low_confidence 상한 = 권장 기준 하한 (초). 미만 시 reason_code='too_short' (low_confidence). 출처: docs/2-3-1 §3-1.",
    )
    effective_fps_failed_threshold: float = Field(
        default=24.0,
        ge=0.0,
        description="Effective FPS failed 임계. 미만 시 reason_code='low_fps' (failed). ⚠️ 네트워크 환경 보정 필요 (docs/2-3-1 §6 실험 계획). 출처: docs/2-3-1 §3-1.",
    )
    effective_fps_low_confidence_threshold: float = Field(
        default=30.0,
        ge=0.0,
        description="Effective FPS low_confidence 상한 = 권장 기준 하한. 미만 시 reason_code='low_fps' (low_confidence). ⚠️ 네트워크 환경 보정 필요. 출처: docs/2-3-1 §3-1.",
    )
    min_resolution_long_edge_px: int = Field(
        default=720,
        ge=1,
        description="해상도 (긴 변 기준) failed 임계 (픽셀). 미만 시 reason_code='low_resolution' (failed, 즉시 검증 §3-4). low_confidence 임계 없음 (§3-1 표 '별도 임계 없음'). 출처: docs/2-3-1 §3-1.",
    )
    frame_count_failed: int = Field(
        default=120,
        ge=1,
        description="누적 frame 수 failed 임계. 미만 시 reason_code='too_short' (failed). duration 또는 frame_count 중 하나라도 위반 시 too_short. 출처: docs/2-3-1 §3-1.",
    )
    frame_count_low_confidence: int = Field(
        default=240,
        ge=1,
        description="누적 frame 수 low_confidence 상한 = 권장 기준 하한. 미만 시 reason_code='too_short' (low_confidence). 출처: docs/2-3-1 §3-1.",
    )
    analysis_end_timeout_sec: float = Field(
        default=2.0,
        ge=0.0,
        description="자동 분석 종료 timeout (초). 마지막 frame 수신 후 본 시간 동안 추가 frame 미도착 시 분석 종료 trigger. trigger이며 reason code 아님. ⚠️ 백엔드 heartbeat 동기화 필요, 1.5~3.0초 범위 보정 예정. 출처: docs/2-3-1 §3-5.",
    )


# ============================================================
# 2-3-4. 자세 지표 산출 (metrics/)
# ============================================================


class SmoothingConfig(BaseModel):
    """좌표 안정화 파라미터 (docs/2-3-4 §3-5, 부록 C)."""

    ema_alpha: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="EMA 계수 α. 초기 default, 구현 단계에서 MA(window=3) 및 EMA(0.3/0.4/0.5)와 비교 평가 후 확정.",
    )
    ma_window: int = Field(
        default=3,
        ge=1,
        description="MA 윈도우 크기 (EMA 비교 후보, docs/2-3-4 부록 C).",
    )


class FrameQualityConfig(BaseModel):
    """프레임 단위 품질 검증 (docs/2-3-4 §3-6).

    ⚠️ 본 모든 임계값은 초기 default. 파일럿 데이터 보정 필요.
    """

    visibility_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Landmark visibility 임계값.",
    )
    consecutive_miss_limit: int = Field(
        default=5,
        ge=1,
        description="연속 visibility 미달 허용 프레임 수 — 초과 시 stride 제외.",
    )
    outlier_variation_threshold: float = Field(
        default=0.20,
        ge=0.0,
        description="이전 평균 대비 변동률 임계 (±20%). 초과 시 outlier로 EMA 제외.",
    )


class ICDetectorConfig(BaseModel):
    """2단계 hybrid IC 검출기 파라미터 (docs/2-3-4 §4, 부록 B/C).

    Stage 1: Zeni 2008 — heel_rel_x local maximum (후보 검출).
    Stage 2: Fellin 2010 — vertical foot velocity 양→음 zero-crossing (정밀화).
    """

    fps: int = Field(default=30, ge=1, description="입력 영상 frame rate (부록 B 의사코드 기준).")
    buffer_size: int = Field(default=15, ge=1, description="순환 버퍼 크기 — 지연 예산에 따라 고정.")
    lookahead: int = Field(default=3, ge=1, description="Stage 1 local max 검증 윈도우 (±3 frame, 고정).")
    refine_window: int = Field(default=3, ge=1, description="Stage 2 zero-crossing 검색 윈도우 (±3 frame, 고정).")
    min_ic_interval: int = Field(default=15, ge=1, description="인접 IC 최소 간격 (frame, 중복 검출 방지).")


class StrideAggregationConfig(BaseModel):
    """Stride 누적 통계 파라미터 (docs/2-3-4 §9-3)."""

    accumulation_n: int = Field(
        default=5,
        ge=1,
        description="누적 stride 개수 N — 평균 stride 0.7s 기준 ≈3~4s. UX 검증 후 확정.",
    )
    min_valid_stride: int = Field(
        default=3,
        ge=1,
        description="대표값 출력을 위한 최소 유효 stride 수. UX 검증.",
    )


class FootStrikeConfig(BaseModel):
    """Foot Strike Pattern 산출 파라미터 (docs/2-3-4 §5).

    분류: RFS (Rearfoot Strike) / MFS (Midfoot Strike) / FFS (Forefoot Strike).

    ⚠️ 임계값 출처: Knorz et al. (2017)은 3D mocap 기반. 본 시스템은 2D MediaPipe
    좌표계로 foot angle 산출 — 절대값 mismatch 가능성 있음. 파일럿 데이터 보정 필수
    (docs/2-3-4 §5-5).
    """

    ic_window_offset: int = Field(
        default=1,
        ge=0,
        description="IC ± offset 프레임 평균 (단일 프레임 검출 오차 보정, ±1~2).",
    )
    rfs_above_deg: float = Field(
        default=5.0,
        description="θ_foot ≥ +5° → RFS (뒤꿈치 먼저 닿음). ⚠️ 파일럿 보정 예정.",
    )
    ffs_below_deg: float = Field(
        default=-5.0,
        description="θ_foot ≤ −5° → FFS (앞발부 먼저 닿음). ⚠️ 파일럿 보정 예정.",
    )
    hysteresis_deg: float = Field(
        default=3.0,
        ge=0.0,
        description="분류 전환 히스테리시스 폭 ±3° (잦은 분류 변동 방지, docs/2-3-4 §5-6).",
    )


class KneeFlexionConfig(BaseModel):
    """Initial Knee Flexion 산출 파라미터 (docs/2-3-4 §6).

    분류 v2.1 — Below Typical / Typical / Above Typical (3분류, 임계 2개).
    CLAUDE.md §5-2의 demo2 4분류 schema와 다름 — docs/2-3-4 v2.1이 단일 정답
    (CLAUDE.md §4: 설계문서가 단일 정답).

    ⚠️ 임계값 출처: Teng & Powers (2014) 등 3D mocap 기반. 본 측정은 2D pose이므로
    절대값 mismatch 위험 (docs/2-3-4 §6-5). 파일럿 데이터 보정 필수.
    윈도 평균은 demo2의 단일 프레임 측정 폐기, IC ± 2~3 프레임 평균 사용
    (CLAUDE.md §5-5).
    """

    ic_window_offset: int = Field(
        default=2,
        ge=0,
        description="IC ± offset 프레임 평균 (단일 프레임 검출 오차 보정, ±2~3).",
    )
    below_typical_deg: float = Field(
        default=15.0,
        description="knee_flexion < 15° → Below Typical Range (충격 흡수 부족 가능). ⚠️ 파일럿 보정.",
    )
    above_typical_deg: float = Field(
        default=25.0,
        description="knee_flexion ≥ 25° → Above Typical Range (충격 흡수↑/효율↓ trade-off). ⚠️ 파일럿 보정.",
    )


class TrunkLeanConfig(BaseModel):
    """Trunk Lean 산출 파라미터 (docs/2-3-4 §7).

    부호 보존: 전방 기울기=+, 후방=−. demo2의 `clip(c, 0, 1)` 절대값화 폐기
    (CLAUDE.md §5-6: 디버깅·재교정 시 정보 손실 방지).

    ⚠️ 임계값 출처: Teng & Powers (2014) 등 3D mocap 기반. 본 측정은 2D pose이므로
    절대값 mismatch 위험 (docs/2-3-4 §7-5). 파일럿 데이터 보정 필수.
    """

    ic_window_offset: int = Field(
        default=2,
        ge=0,
        description="IC ± offset 프레임 구간 평균 (팔 스윙 등 순간 흔들림 완화, ±2~3).",
    )
    near_vertical_below_deg: float = Field(
        default=5.0,
        description="θ_trunk < 5° → Near Vertical (거의 수직). ⚠️ 파일럿 보정.",
    )
    forward_above_deg: float = Field(
        default=10.0,
        description="θ_trunk > 10° → Above Typical Range (일반 범위 초과, Teng & Powers 2014 참고). ⚠️ 파일럿 보정.",
    )


# ============================================================
# 2-3-5. Pose 후 품질 검사 + 분석 상태값 (quality_gate.py)
# ============================================================


class VisibilityCheckConfig(BaseModel):
    """신체 가시성·전신 포함·발 잘림 검사 (docs/2-3-5 §5-1, §5-2, §5-3).

    한 클래스에 통합한 이유: 세 절 모두 "신체가 화면에 충분히 들어와 있나"
    한 도메인. §5-3 (foot cutoff y < 0.95)는 visibility가 아닌 좌표 범위
    기반이지만, "전신 일부 잘림" 카테고리로 묶임.

    ⚠️ 본 모든 임계값은 초기 default. 파일럿 데이터 보정 필요.
    """

    # §5-1 Landmark visibility
    visibility_threshold_lower_body: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="분석측 hip/knee/ankle 평균 visibility 임계. 미달 시 `lower_body_not_visible`.",
    )
    visibility_threshold_foot: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="분석측 heel/foot_index 평균 visibility 임계. 미달 시 `foot_not_visible`.",
    )
    visibility_threshold_upper_body: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="양측 shoulder 평균 visibility 임계. 미달 시 `upper_body_not_visible`.",
    )
    visibility_threshold_overall_avg: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="전체 주요 landmark visibility 평균 임계. 미달 시 `low_landmark_visibility`.",
    )
    valid_frame_ratio_min: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="visibility 임계 통과 frame 비율 최소. 미달 시 `low_landmark_visibility`.",
    )
    # §5-2 전신 포함
    body_inclusion_frame_ratio_min: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="nose/ankle visibility ≥ 0.6 AND 전신 좌표 [0,1] 범위 내 frame 비율 최소. 미달 시 `body_not_fully_visible`.",
    )
    # §5-3 발 잘림
    foot_cutoff_y_max: float = Field(
        default=0.95,
        ge=0.0, le=1.0,
        description="분석측 ankle/heel/foot_index y 좌표 최대 (> 0.95면 사실상 화면 하단 닿음, 발 잘림).",
    )
    foot_cutoff_frame_ratio_min: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="발 잘림 조건 만족 frame 비율 최소. 미달 시 `foot_out_of_frame`.",
    )


class SideViewConfig(BaseModel):
    """측면 촬영 구도 검사 (docs/2-3-5 §5-4).

    1차 조건 (hip 거리) + 보조 조건 (shoulder 거리 또는 yaw proxy) 둘 중 하나 이상.
    단일 ratio 불안정성·shoulder 흔들림을 보완하기 위한 robust 판정.

    ⚠️ 파일럿 보정 필요.
    """

    hip_x_distance_max: float = Field(
        default=0.05,
        ge=0.0, le=1.0,
        description="좌우 hip의 x 거리 최대 (화면 폭의 5%). 1차 조건 (필수).",
    )
    shoulder_x_distance_max: float = Field(
        default=0.07,
        ge=0.0, le=1.0,
        description="좌우 shoulder의 x 거리 최대 (화면 폭의 7%). 보조 조건 (a).",
    )
    torso_yaw_proxy_max: float = Field(
        default=0.15,
        ge=0.0,
        description="torso yaw proxy 임계 (좌우 hip 거리 / hip-shoulder 수직 거리). 0.15 미만 시 측면. 보조 조건 (b).",
    )
    primary_condition_frame_ratio_min: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="1차 조건 충족 frame 비율 최소. 60% 미만 시 `invalid_view` failed 강도.",
    )


class AnalysisSideConfig(BaseModel):
    """분석측 결정 파라미터 (docs/2-3-5 §3-1).

    분석측은 두 조건 모두 충족된 시점에 결정 — 시간 기준만 사용 시 effective FPS
    낮은 환경에서 표본 부족, frame 수 기준만 사용 시 effective FPS 변동에 흔들림.
    """

    min_decision_seconds: float = Field(
        default=1.5,
        ge=0.0,
        description="분석측 결정 시작 최소 경과 시간 (초). AND 조건의 한 쪽.",
    )
    min_valid_frames: int = Field(
        default=30,
        ge=1,
        description="분석측 결정 시작 최소 유효 frame 수 (visibility 임계 통과). AND 조건의 다른 쪽.",
    )


class TrackingStabilityConfig(BaseModel):
    """분석 대상자 추적 안정성 파라미터 (docs/2-3-5 §4-2 + §4-3 + §4-4).

    모든 신호는 window 기반 잔차로 평가 — 단일 frame 변화량이 아닌 짧은 window
    평균 변화량 (§4-2-1 평활화 원칙). 정상 범위 임계와 reason code 트리거
    임계를 명확히 구분한다.

    ⚠️ 파일럿 보정 필요.
    """

    # §4-2 정상 범위 임계
    pelvis_residual_max: float = Field(
        default=0.10,
        ge=0.0,
        description="Pelvis 중심 좌표 잔차 정상 임계 (화면 폭의 10% 미만, 0.5초 window).",
    )
    scale_variation_max: float = Field(
        default=0.15,
        ge=0.0,
        description="신체 크기 (landmark scale) 변동률 정상 임계 (±15%, 1초 window).",
    )
    visibility_avg_min: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="주요 landmark visibility 평균 정상 임계 (1초 window).",
    )
    heel_midstance_residual_max: float = Field(
        default=0.08,
        ge=0.0,
        description="heel/foot_index 좌표 잔차 정상 임계 (mid-stance 구간, 화면 폭의 8% 미만).",
    )
    stride_interval_deviation_max: float = Field(
        default=0.30,
        ge=0.0,
        description="Stride 간격 중앙값 대비 편차 정상 임계 (30% 미만, 누적 5개 stride 기준).",
    )

    # §4-3 reason code 트리거 임계
    pelvis_residual_spike: float = Field(
        default=0.15,
        ge=0.0,
        description="Pelvis 잔차 급증 임계 (15% 초과) — `target_switch_detected` 트리거 조건 중 하나.",
    )
    scale_spike: float = Field(
        default=0.20,
        ge=0.0,
        description="Scale 급변 임계 (±20% 이상) — `target_switch_detected` 트리거 조건 중 하나.",
    )
    target_lost_visibility_threshold: float = Field(
        default=0.4,
        ge=0.0, le=1.0,
        description="대상자 자체가 보이지 않는 visibility 임계 — `target_switch_detected`/`target_lost` 트리거.",
    )
    target_lost_seconds: float = Field(
        default=5.0,
        ge=0.0,
        description="`target_lost` 트리거를 위한 연속 미달 시간 (초).",
    )
    heel_midstance_violation_ratio_max: float = Field(
        default=0.30,
        ge=0.0, le=1.0,
        description="`unstable_landmark_sequence` 트리거 — heel/foot 잔차 임계 위반이 분석 시간의 30% 이상.",
    )
    visibility_borderline_low: float = Field(
        default=0.4,
        ge=0.0, le=1.0,
        description="`background_person_interference` 트리거 borderline visibility 하한.",
    )
    visibility_borderline_high: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="`background_person_interference` 트리거 borderline visibility 상한.",
    )
    visibility_borderline_violation_ratio_max: float = Field(
        default=0.30,
        ge=0.0, le=1.0,
        description="borderline visibility가 30% 이상 frame에서 발생 시 `background_person_interference`.",
    )

    # §4-4 평활화 윈도우
    pelvis_window_seconds: float = Field(
        default=0.5,
        ge=0.0,
        description="Pelvis 잔차 평가 sliding window (초).",
    )
    visibility_window_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Visibility 평균 평가 sliding window (초).",
    )
    scale_window_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description="Scale 변동 평가 sliding window (초).",
    )


class StrideExclusionConfig(BaseModel):
    """Stride 제외 cross-cutting 조건 (docs/2-3-4 §10 + docs/2-3-5 §5-5).

    §10 표의 각 항목은 "해당 stride 제외" 처리가 공통. 이 클래스는 visibility
    기반 (VisibilityCheckConfig)과 IC 기반 (ICValidationConfig) 외의 stride
    제외 트리거 조건을 모은다.
    """

    camera_pelvis_x_stride_variation_max: float = Field(
        default=0.30,
        ge=0.0,
        description="Pelvis_x의 stride 평균 대비 변동 임계 (±30%). 초과 시 `camera_unstable` (docs/2-3-5 §5-5).",
    )
    stride_time_max_seconds: float = Field(
        default=1.5,
        ge=0.0,
        description="Stride time 최대 (초). 초과 시 정지/보행 구간으로 판단, `insufficient_stride` (docs/2-3-4 §10).",
    )


class MetricVariabilityConfig(BaseModel):
    """자세 지표 stride 간 변동성 검사 (docs/2-3-5 §5-6).

    측정 변동성이 분류 임계 구간 폭에 가까우면 분류 자체가 무의미. 따라서 변동성
    기준은 임계 구간 폭보다 작게 설정 (예: foot 분류 폭 ±5° → 변동성 < 5°).

    ⚠️ 파일럿 보정 필요.
    """

    foot_stddev_max_deg: float = Field(
        default=5.0,
        ge=0.0,
        description="Foot angle stride 간 표준편차 임계. 초과 시 `unstable_foot_angle`.",
    )
    knee_stddev_max_deg: float = Field(
        default=7.0,
        ge=0.0,
        description="Knee flexion stride 간 표준편차 임계. 초과 시 `unstable_knee_angle`.",
    )
    trunk_stddev_max_deg: float = Field(
        default=4.0,
        ge=0.0,
        description="Trunk lean stride 간 표준편차 임계. 초과 시 `unstable_trunk_angle`.",
    )


class ICValidationConfig(BaseModel):
    """IC 후보 사후 검증 (docs/2-3-5 §5-7).

    ICDetectorConfig는 검출 알고리즘 파라미터 (per-stride 동작), 본 클래스는
    누적된 IC 결과에 대한 검증 임계 (분석 종료 시점 평가). 모듈 경계 분리:
    `quality_gate.py`가 본 클래스를 참조한다.
    """

    min_total_ic: int = Field(
        default=3,
        ge=1,
        description="전체 IC 후보 수 최소. 미달 시 `insufficient_stride` (docs/2-3-4 §8 인용).",
    )
    min_high_medium_confidence_ic: int = Field(
        default=2,
        ge=1,
        description="신뢰도 high/medium IC 수 최소. 미달 시 `low_ic_confidence`.",
    )
    trunk_lean_window_min_valid_ratio: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Trunk lean 계산 IC ±2~3 frame window 내 visibility 통과 비율 최소. 미달 시 `insufficient_window` (docs/2-3-4 §7-2).",
    )


# ============================================================
# AppConfig — 메인 통합 클래스
# ============================================================


class AppConfig(BaseSettings):
    """ChoboRunner AI Server 통합 설정.

    환경변수 prefix `CHOBO_`. 중첩 구분자 `__`.
    예: `CHOBO_IC__LOOKAHEAD=5` → `app_config.ic.lookahead = 5`로 override.

    사용 예:
        from choborunner_ai.config import app_config
        alpha = app_config.smoothing.ema_alpha
        rfs_threshold = app_config.foot_strike.rfs_above_deg
        stride_max = app_config.stride_exclusion.stride_time_max_seconds
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CHOBO_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ── 2-3-1 입력 영상 메타데이터 ──────────────────────────
    input_metadata: InputMetadataConfig = Field(default_factory=InputMetadataConfig)

    # ── 2-3-4 자세 지표 산출 ───────────────────────────────
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    frame_quality: FrameQualityConfig = Field(default_factory=FrameQualityConfig)
    ic: ICDetectorConfig = Field(default_factory=ICDetectorConfig)
    stride: StrideAggregationConfig = Field(default_factory=StrideAggregationConfig)
    foot_strike: FootStrikeConfig = Field(default_factory=FootStrikeConfig)
    knee_flexion: KneeFlexionConfig = Field(default_factory=KneeFlexionConfig)
    trunk_lean: TrunkLeanConfig = Field(default_factory=TrunkLeanConfig)

    # ── 2-3-5 Pose 후 품질 검사 ────────────────────────────
    visibility_check: VisibilityCheckConfig = Field(default_factory=VisibilityCheckConfig)
    side_view: SideViewConfig = Field(default_factory=SideViewConfig)
    analysis_side: AnalysisSideConfig = Field(default_factory=AnalysisSideConfig)
    tracking: TrackingStabilityConfig = Field(default_factory=TrackingStabilityConfig)
    stride_exclusion: StrideExclusionConfig = Field(default_factory=StrideExclusionConfig)
    variability: MetricVariabilityConfig = Field(default_factory=MetricVariabilityConfig)
    ic_validation: ICValidationConfig = Field(default_factory=ICValidationConfig)

    # ── 2-3-2/3/6/7 영역 — 해당 docs 확정 후 추가 ────────


app_config = AppConfig()
