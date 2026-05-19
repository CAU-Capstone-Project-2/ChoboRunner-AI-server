"""quality_gate.py unit 테스트 — docs/2-3-5 §5-2 + §5-3 + §5-4 (Phase 8-A/8-B-1/8-B-2).

본 파일은 Phase 8-A 신규 — `tmp/phase_8_a_sanity.py` 8 case 이식 + pytest 형식.

Phase 8-B-1 (δ 시그니처 통일):
- 누적 평가 함수 3종 반환 시그니처 변경: Optional[ReasonCode] /
  list[ReasonCode] → list[ReasonCodeEntry]
- 본 파일 backport: C/D 카테고리 3 case 수정 (Optional[ReasonCode] 가정 →
  list[ReasonCodeEntry] 검증)
- 신규 case 1: ReasonCodeEntry severity 정합 확인

Phase 8-B-2 (§5-4 측면 구도 invalid_view 2 severity):
- evaluate_frame_side_view (frame-level 3 sub-check)
- evaluate_side_view_accumulation (1차/보조 종합 판정, 2 severity 분기)
- 신규 F/G 카테고리 6 case (frame 3 + 누적 4 시나리오 중 핵심)

⚠️ §5-1 (evaluate_frame_visibility / evaluate_visibility_accumulation, Phase 4)는
   본 파일에서 다루지 않음 (Phase 4는 sanity script만 작성 — scripts/sanity/
   phase_4_c_integration.py). 향후 §5-1 pytest 이식은 별도 cleanup 후보.

15 case (9 → 15, +6 case for §5-4 Phase 8-B-2):
- A. evaluate_frame_body_inclusion (3, Phase 8-A)
- B. evaluate_frame_foot_cutoff (2, Phase 8-A)
- C. evaluate_body_inclusion_accumulation (2, 8-A + 8-B-1 δ)
- D. evaluate_foot_cutoff_accumulation (1, 8-A + 8-B-1 δ)
- E. ReasonCodeEntry severity 정합 (1, Phase 8-B-1)
- F. evaluate_frame_side_view (2, Phase 8-B-2 — 정상 + 1차 위반)
- G. evaluate_side_view_accumulation (4 시나리오, Phase 8-B-2)
"""
from __future__ import annotations

import pytest

from choborunner_ai.config import (
    ICValidationConfig,
    MetricVariabilityConfig,
    SideViewConfig,
    StrideExclusionConfig,
    TrackingStabilityConfig,
    VisibilityCheckConfig,
)
from choborunner_ai.pose_extractor import Landmark, LandmarkPair, PoseLandmarks
from choborunner_ai.quality_gate import (
    REASON_CODE_PRIORITY,
    REASON_CODE_SEVERITY,
    FrameGeometryResult,
    FrameSideViewResult,
    FrameVisibilityResult,
    ReasonCodeEntry,
    ResponseStatusResult,
    compute_response_status,
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
    evaluate_target_switch,
    evaluate_tracking_stability,
    evaluate_visibility_accumulation,
)


def _lm(x: float, y: float, vis: float = 0.9) -> Landmark:
    return Landmark(x=x, y=y, visibility=vis)


def _pair(x: float, y: float, vis: float = 0.9) -> LandmarkPair:
    return LandmarkPair(left=_lm(x - 0.05, y, vis), right=_lm(x + 0.05, y, vis))


def _normal_pl(
    nose_vis: float = 0.9, shoulder_x_offset: float = 0.0
) -> PoseLandmarks:
    """13점 정상 PoseLandmarks. shoulder_x_offset으로 좌표 out-of-range 케이스 생성."""
    return PoseLandmarks(
        shoulder=LandmarkPair(
            left=_lm(0.45 + shoulder_x_offset, 0.20),
            right=_lm(0.55 + shoulder_x_offset, 0.20),
        ),
        hip=_pair(0.50, 0.45),
        knee=_pair(0.50, 0.65),
        ankle=_pair(0.50, 0.85),
        heel=_pair(0.48, 0.88),
        foot_index=_pair(0.52, 0.88),
        nose=_lm(0.50, 0.10, nose_vis),
    )


@pytest.fixture
def cfg() -> VisibilityCheckConfig:
    return VisibilityCheckConfig()


# ============================================================
# A. evaluate_frame_body_inclusion (§5-2)
# ============================================================


def test_body_inclusion_normal(cfg: VisibilityCheckConfig):
    """정상 13점, nose+ankle visibility 0.9 → is_valid=True."""
    pl = _normal_pl()
    r = evaluate_frame_body_inclusion(pl, cfg)
    assert r.is_valid
    assert r.passed_checks["body_visibility"]
    assert r.passed_checks["body_coords"]
    assert r.failed_reasons == []
    assert abs(r.check_values["nose_visibility"] - 0.9) < 1e-9
    assert r.check_values["coord_out_of_range_count"] == 0.0


def test_body_inclusion_nose_visibility_fail(cfg: VisibilityCheckConfig):
    """nose visibility 0.5 < 0.6 임계 → body_visibility 실패."""
    pl = _normal_pl(nose_vis=0.5)
    r = evaluate_frame_body_inclusion(pl, cfg)
    assert not r.is_valid
    assert not r.passed_checks["body_visibility"]
    assert r.passed_checks["body_coords"]  # 좌표는 정상
    assert "body_not_fully_visible" in r.failed_reasons


def test_body_inclusion_coord_out_of_range(cfg: VisibilityCheckConfig):
    """shoulder x > 1.0 (out of [0,1] range) → body_coords 실패."""
    pl = _normal_pl(shoulder_x_offset=0.6)  # shoulder.x → 1.05, 1.15
    r = evaluate_frame_body_inclusion(pl, cfg)
    assert not r.is_valid
    assert r.passed_checks["body_visibility"]  # visibility는 정상
    assert not r.passed_checks["body_coords"]
    assert "body_not_fully_visible" in r.failed_reasons
    assert r.check_values["coord_out_of_range_count"] >= 1.0


# ============================================================
# B. evaluate_frame_foot_cutoff (§5-3)
# ============================================================


def test_foot_cutoff_normal(cfg: VisibilityCheckConfig):
    """정상 ankle/heel/foot y < 0.95 (0.85, 0.88) → is_valid=True."""
    pl = _normal_pl()
    r = evaluate_frame_foot_cutoff(pl, "left", cfg)
    assert r.is_valid
    assert r.passed_checks["foot_cutoff"]
    assert r.failed_reasons == []


def test_foot_cutoff_one_point_violation(cfg: VisibilityCheckConfig):
    """foot_index y=0.97 (5-7 α AND 해석: 1점 위반도 fail) → foot_out_of_frame."""
    pl = PoseLandmarks(
        shoulder=_pair(0.50, 0.20),
        hip=_pair(0.50, 0.45),
        knee=_pair(0.50, 0.65),
        ankle=_pair(0.50, 0.85),
        heel=_pair(0.48, 0.88),
        # 분석측(left) foot_index y=0.97 — 1점만 위반
        foot_index=LandmarkPair(left=_lm(0.47, 0.97), right=_lm(0.57, 0.88)),
        nose=_lm(0.50, 0.10, 0.9),
    )
    r = evaluate_frame_foot_cutoff(pl, "left", cfg)
    assert not r.is_valid
    assert not r.passed_checks["foot_cutoff"]
    assert "foot_out_of_frame" in r.failed_reasons
    assert abs(r.check_values["foot_index_y"] - 0.97) < 1e-9


# ============================================================
# C. evaluate_body_inclusion_accumulation (§5-2 누적)
# ============================================================


def test_body_inclusion_accumulation_pass(cfg: VisibilityCheckConfig):
    """70% 통과 (7 valid / 3 invalid) → [] (Phase 8-B-1 δ: 빈 list)."""
    results = (
        [FrameGeometryResult(is_valid=True) for _ in range(7)]
        + [FrameGeometryResult(is_valid=False) for _ in range(3)]
    )
    out = evaluate_body_inclusion_accumulation(results, cfg)
    assert out == []


def test_body_inclusion_accumulation_fail(cfg: VisibilityCheckConfig):
    """50% 통과 → [ReasonCodeEntry('body_not_fully_visible', 'failed')] (8-B-1 δ)."""
    results = (
        [FrameGeometryResult(is_valid=True) for _ in range(5)]
        + [FrameGeometryResult(is_valid=False) for _ in range(5)]
    )
    out = evaluate_body_inclusion_accumulation(results, cfg)
    assert len(out) == 1
    assert out[0].reason_code == "body_not_fully_visible"
    assert out[0].severity == "failed"


# ============================================================
# D. evaluate_foot_cutoff_accumulation (§5-3 누적)
# ============================================================


def test_foot_cutoff_accumulation_fail(cfg: VisibilityCheckConfig):
    """50% 통과 → [ReasonCodeEntry('foot_out_of_frame', 'failed')] (8-B-1 δ)."""
    results = (
        [FrameGeometryResult(is_valid=True) for _ in range(5)]
        + [FrameGeometryResult(is_valid=False) for _ in range(5)]
    )
    out = evaluate_foot_cutoff_accumulation(results, cfg)
    assert len(out) == 1
    assert out[0].reason_code == "foot_out_of_frame"
    assert out[0].severity == "failed"


# ============================================================
# E. ReasonCodeEntry severity 정합 (Phase 8-B-1 신규)
# ============================================================


def test_reason_code_entry_default_severity_lookup():
    """ReasonCodeEntry severity가 REASON_CODE_SEVERITY default와 정합.

    Phase 8-B-1 δ 도입 — 누적 평가 함수가 REASON_CODE_SEVERITY default를 사용해
    ReasonCodeEntry를 wrap. 본 case는 default 사전 lookup 정합 회귀.

    frozen=True 검증도 동시 — 누적 평가 결과 불변 보장.
    """
    # default 사전 lookup 정합
    assert REASON_CODE_SEVERITY["body_not_fully_visible"] == "failed"
    assert REASON_CODE_SEVERITY["foot_out_of_frame"] == "failed"
    assert REASON_CODE_SEVERITY["low_landmark_visibility"] == "low_confidence"
    assert REASON_CODE_SEVERITY["lower_body_not_visible"] == "low_confidence"

    # frozen=True 검증
    entry = ReasonCodeEntry(reason_code="body_not_fully_visible", severity="failed")
    with pytest.raises(Exception):  # FrozenInstanceError (dataclass frozen)
        entry.reason_code = "foot_out_of_frame"  # type: ignore[misc]


# ============================================================
# F. evaluate_frame_side_view (§5-4 Phase 8-B-2)
# ============================================================


@pytest.fixture
def side_cfg() -> SideViewConfig:
    return SideViewConfig()


def _side_pl(
    hip_x_dist: float,
    shoulder_x_dist: float,
    hip_y: float = 0.50,
    shoulder_y: float = 0.20,
) -> PoseLandmarks:
    """측면 구도 합성 PoseLandmarks (shoulder/hip만 의미 있음)."""
    return PoseLandmarks(
        shoulder=LandmarkPair(
            left=_lm(0.50 - shoulder_x_dist / 2, shoulder_y),
            right=_lm(0.50 + shoulder_x_dist / 2, shoulder_y),
        ),
        hip=LandmarkPair(
            left=_lm(0.50 - hip_x_dist / 2, hip_y),
            right=_lm(0.50 + hip_x_dist / 2, hip_y),
        ),
        knee=LandmarkPair(left=_lm(0.50, 0.65), right=_lm(0.50, 0.65)),
        ankle=LandmarkPair(left=_lm(0.50, 0.85), right=_lm(0.50, 0.85)),
        heel=LandmarkPair(left=_lm(0.48, 0.88), right=_lm(0.52, 0.88)),
        foot_index=LandmarkPair(left=_lm(0.52, 0.88), right=_lm(0.52, 0.88)),
    )


def test_side_view_frame_normal(side_cfg: SideViewConfig):
    """정상 측면: hip 0.03 + shoulder 0.05 + yaw 0.1 → 모두 통과."""
    pl = _side_pl(hip_x_dist=0.03, shoulder_x_dist=0.05)
    r = evaluate_frame_side_view(pl, side_cfg)
    assert r.is_valid
    assert r.passed_checks["primary_hip_x"]
    assert r.passed_checks["secondary_a_shoulder_x"]
    assert r.passed_checks["secondary_b_torso_yaw"]


def test_side_view_frame_torso_yaw_zero_division_guard(side_cfg: SideViewConfig):
    """torso yaw 분모 0 (hip_y == shoulder_y) → epsilon 가드 + b 자동 위반.

    catch 8-B-2-β: hip_shoulder_y_distance < 1e-6 → torso_yaw_ratio = inf → b 실패.
    a는 정상 통과 시 is_valid = primary AND a → True 유지.
    """
    pl = _side_pl(hip_x_dist=0.03, shoulder_x_dist=0.05, hip_y=0.30, shoulder_y=0.30)
    r = evaluate_frame_side_view(pl, side_cfg)
    assert r.check_values["torso_yaw_ratio"] == float("inf")
    assert not r.passed_checks["secondary_b_torso_yaw"]
    # primary AND (a OR b) = True AND (True OR False) = True
    assert r.is_valid


# ============================================================
# G. evaluate_side_view_accumulation (§5-4 4 시나리오, Phase 8-B-2)
# ============================================================


def _frame(primary: bool, sec_a: bool, sec_b: bool) -> FrameSideViewResult:
    """누적 평가 합성 입력 — passed_checks만 박음."""
    return FrameSideViewResult(
        passed_checks={
            "primary_hip_x": primary,
            "secondary_a_shoulder_x": sec_a,
            "secondary_b_torso_yaw": sec_b,
        },
        check_values={},
        is_valid=primary and (sec_a or sec_b),
    )


def test_side_view_accumulation_normal(side_cfg: SideViewConfig):
    """(a) 정상: 1차 80% + 보조 위반 0% → []."""
    results = [_frame(True, True, True) for _ in range(8)] + [
        _frame(False, False, False) for _ in range(2)
    ]
    out = evaluate_side_view_accumulation(results, side_cfg)
    assert out == []


def test_side_view_accumulation_low_confidence_only(side_cfg: SideViewConfig):
    """(b) low_conf만: 1차 80% + 보조 위반 50% (1차 통과 분모) → low_confidence."""
    results = (
        [_frame(True, True, True) for _ in range(4)]
        + [_frame(True, False, False) for _ in range(4)]
        + [_frame(False, False, False) for _ in range(2)]
    )
    out = evaluate_side_view_accumulation(results, side_cfg)
    assert len(out) == 1
    assert out[0].reason_code == "invalid_view"
    assert out[0].severity == "low_confidence"


def test_side_view_accumulation_failed_only(side_cfg: SideViewConfig):
    """(c) failed만: 1차 40% + 보조 위반 0% → failed."""
    results = (
        [_frame(True, True, True) for _ in range(4)]
        + [_frame(False, False, False) for _ in range(6)]
    )
    out = evaluate_side_view_accumulation(results, side_cfg)
    assert len(out) == 1
    assert out[0].reason_code == "invalid_view"
    assert out[0].severity == "failed"


def test_side_view_accumulation_both_triggers(side_cfg: SideViewConfig):
    """(d) 둘 다: 1차 40% + 보조 위반 50% (1차 통과 분모) → failed + low_confidence.

    catch 8-B-2-γ: list에 둘 다 entry. Phase 8-F (status 분기) 위임.
    """
    results = (
        [_frame(True, True, True) for _ in range(2)]
        + [_frame(True, False, False) for _ in range(2)]
        + [_frame(False, False, False) for _ in range(6)]
    )
    out = evaluate_side_view_accumulation(results, side_cfg)
    assert len(out) == 2
    severities = {e.severity for e in out}
    assert severities == {"failed", "low_confidence"}
    # 둘 다 reason_code는 'invalid_view'
    assert all(e.reason_code == "invalid_view" for e in out)


# ============================================================
# H. evaluate_camera_stability (§5-5 Phase 8-C)
# ============================================================


@pytest.fixture
def cam_cfg() -> StrideExclusionConfig:
    return StrideExclusionConfig()


def _camera_pl(hip_x: float) -> PoseLandmarks:
    """hip 양측 평균 x = hip_x인 합성 PoseLandmarks (camera_stability 입력)."""
    return PoseLandmarks(
        shoulder=LandmarkPair(left=_lm(hip_x, 0.20), right=_lm(hip_x, 0.20)),
        hip=LandmarkPair(
            left=_lm(hip_x - 0.02, 0.50), right=_lm(hip_x + 0.02, 0.50)
        ),
        knee=LandmarkPair(left=_lm(hip_x, 0.65), right=_lm(hip_x, 0.65)),
        ankle=LandmarkPair(left=_lm(hip_x, 0.85), right=_lm(hip_x, 0.85)),
        heel=LandmarkPair(left=_lm(hip_x - 0.02, 0.88), right=_lm(hip_x + 0.02, 0.88)),
        foot_index=LandmarkPair(left=_lm(hip_x, 0.88), right=_lm(hip_x, 0.88)),
    )


def test_camera_stability_normal(cam_cfg: StrideExclusionConfig):
    """2 stride 모두 pelvis_x 변동 < 30% → []."""
    # 20 frames, IC at 0/10/20. variation = 0.04/0.50 = 0.08
    landmarks = [_camera_pl(0.50 + 0.02 * ((-1) ** i)) for i in range(20)]
    out = evaluate_camera_stability(landmarks, [0, 10, 20], cam_cfg)
    assert out == []


def test_camera_stability_single_stride_violation(cam_cfg: StrideExclusionConfig):
    """lock 8-C-1 α: 단일 stride 위반도 → camera_unstable 트리거."""
    # stride 1 정상 + stride 2 큰 변동
    landmarks = [_camera_pl(0.50 + 0.02 * ((-1) ** i)) for i in range(10)]
    landmarks += [_camera_pl(0.30 + 0.04 * i) for i in range(10)]
    out = evaluate_camera_stability(landmarks, [0, 10, 20], cam_cfg)
    assert len(out) == 1
    assert out[0].reason_code == "camera_unstable"
    assert out[0].severity == "low_confidence"


def test_camera_stability_ic_indices_too_short(cam_cfg: StrideExclusionConfig):
    """catch 8-C-14: ic_indices 길이 < 2 → []."""
    landmarks = [_camera_pl(0.50) for _ in range(10)]
    out = evaluate_camera_stability(landmarks, [0], cam_cfg)
    assert out == []


# ============================================================
# I. evaluate_metric_variability (§5-6 Phase 8-C)
# ============================================================


@pytest.fixture
def var_cfg() -> MetricVariabilityConfig:
    return MetricVariabilityConfig()


def test_metric_variability_normal(var_cfg: MetricVariabilityConfig):
    """3 metric stddev 모두 임계 이내 → []."""
    foot = [1.0, 2.0, 1.5, 2.5, 1.8]
    knee = [17.0, 18.0, 17.5, 18.5, 17.2]
    trunk = [7.0, 7.5, 7.2, 7.8, 7.1]
    out = evaluate_metric_variability(foot, knee, trunk, var_cfg)
    assert out == []


def test_metric_variability_all_violate(var_cfg: MetricVariabilityConfig):
    """3 metric 모두 위반 → 3 entry (모두 low_confidence)."""
    foot = [0.0, 10.0, 20.0, 30.0, 5.0]  # stddev > 5
    knee = [10.0, 30.0, 15.0, 35.0, 12.0]  # stddev > 7
    trunk = [0.0, 8.0, 2.0, 10.0, 1.0]  # stddev > 4
    out = evaluate_metric_variability(foot, knee, trunk, var_cfg)
    assert len(out) == 3
    codes = {e.reason_code for e in out}
    assert codes == {"unstable_foot_angle", "unstable_knee_angle", "unstable_trunk_angle"}
    assert all(e.severity == "low_confidence" for e in out)


def test_metric_variability_n_one_skip(var_cfg: MetricVariabilityConfig):
    """입력 n=1 → stddev 미정의, 해당 metric skip → []."""
    out = evaluate_metric_variability([1.0], [10.0], [5.0], var_cfg)
    assert out == []


# ============================================================
# J. evaluate_ic_validation (§5-7 Phase 8-D, severity 혼합)
# ============================================================


@pytest.fixture
def ic_cfg() -> ICValidationConfig:
    return ICValidationConfig()


def test_ic_validation_normal(ic_cfg: ICValidationConfig):
    """정상: IC 5 high + window 0.8 → []."""
    out = evaluate_ic_validation(["high"] * 5, [0.8] * 5, ic_cfg)
    assert out == []


def test_ic_validation_insufficient_stride_only(ic_cfg: ICValidationConfig):
    """IC 2 high (count 2 < 3 trigger, high/medium=2 == min 통과) → insufficient_stride만."""
    out = evaluate_ic_validation(["high"] * 2, [0.8] * 5, ic_cfg)
    assert len(out) == 1
    assert out[0].reason_code == "insufficient_stride"
    assert out[0].severity == "failed"


def test_ic_validation_low_ic_confidence_only(ic_cfg: ICValidationConfig):
    """IC 5 + high=1 (high/medium=1 < 2 trigger) + window 0.8 → low_ic_confidence만."""
    out = evaluate_ic_validation(["high"] + ["low"] * 4, [0.8] * 5, ic_cfg)
    assert len(out) == 1
    assert out[0].reason_code == "low_ic_confidence"
    assert out[0].severity == "low_confidence"


def test_ic_validation_insufficient_window_only(ic_cfg: ICValidationConfig):
    """IC 5 high + window 1개 < 0.5 → insufficient_window만 (lock 8-D-3 α 단일 위반)."""
    out = evaluate_ic_validation(
        ["high"] * 5,
        [0.8, 0.8, 0.3, 0.8, 0.8],
        ic_cfg,
    )
    assert len(out) == 1
    assert out[0].reason_code == "insufficient_window"
    assert out[0].severity == "low_confidence"


def test_ic_validation_all_three_triggers_severity_mixed(ic_cfg: ICValidationConfig):
    """3 reason_code 모두 트리거 (severity 혼합 catch 7-2 검증).

    IC 1 low → insufficient_stride (failed) + low_ic_confidence (low_conf)
    window 0.3 → insufficient_window (low_conf)
    """
    out = evaluate_ic_validation(["low"], [0.3], ic_cfg)
    assert len(out) == 3
    codes = {e.reason_code for e in out}
    assert codes == {"insufficient_stride", "low_ic_confidence", "insufficient_window"}
    severities = {e.severity for e in out}
    assert severities == {"failed", "low_confidence"}


def test_ic_validation_empty_inputs(ic_cfg: ICValidationConfig):
    """lock 8-D-8 B + 8-D-9 A:
    ic_confidences=[] → insufficient_stride + low_ic_confidence 둘 다 트리거 (정보 보존)
    trunk_window_valid_ratios=[] → insufficient_window skip (false positive 회피)
    """
    out = evaluate_ic_validation([], [], ic_cfg)
    assert len(out) == 2
    codes = {e.reason_code for e in out}
    assert codes == {"insufficient_stride", "low_ic_confidence"}
    # insufficient_window는 trigger 안 됨 (8-D-9 A skip)
    assert "insufficient_window" not in codes


# ============================================================
# K. evaluate_tracking_stability (§4 Phase 8-E scope γ)
# ============================================================


@pytest.fixture
def track_cfg() -> TrackingStabilityConfig:
    return TrackingStabilityConfig()


def test_tracking_stability_normal(track_cfg: TrackingStabilityConfig):
    """정상 — 10초 visibility 0.9 전체 → []."""
    out = evaluate_tracking_stability([0.9] * 300, 30.0, track_cfg)
    assert out == []


def test_tracking_stability_target_lost_only(track_cfg: TrackingStabilityConfig):
    """target_lost trigger — sliding window of windows 보수성으로 6초 0.2 필요.

    lock 8-E-6 α 정확 해석: 5초 timestamp 동안 매 frame window 평균 < 0.4 필요.
    raw frame 0.2가 5초만 있어도 sliding window 평균은 늦게 < 0.4 진입.
    """
    visibility = [0.9] * 120 + [0.2] * 180  # 4초 0.9 + 6초 0.2
    out = evaluate_tracking_stability(visibility, 30.0, track_cfg)
    assert len(out) == 1
    assert out[0].reason_code == "target_lost"
    assert out[0].severity == "failed"


def test_tracking_stability_target_lost_no_trigger_4sec(
    track_cfg: TrackingStabilityConfig,
):
    """4초 0.2 < 5초 → trigger X (sliding window 보수성)."""
    visibility = [0.9] * 180 + [0.2] * 120  # 6초 0.9 + 4초 0.2
    out = evaluate_tracking_stability(visibility, 30.0, track_cfg)
    assert out == []


def test_tracking_stability_background_only(track_cfg: TrackingStabilityConfig):
    """background trigger — borderline 50% ≥ 30%."""
    visibility = [0.9] * 150 + [0.5] * 150  # 50% borderline (0.4 <= 0.5 < 0.6)
    out = evaluate_tracking_stability(visibility, 30.0, track_cfg)
    assert len(out) == 1
    assert out[0].reason_code == "background_person_interference"
    assert out[0].severity == "low_confidence"


def test_tracking_stability_both_triggers_severity_mixed(
    track_cfg: TrackingStabilityConfig,
):
    """severity 혼합 — target_lost (failed) + background (low_confidence) 둘 다 trigger."""
    # 0.5 100 frame (borderline ~32%) + 0.9 30 + 0.2 180 (target_lost 6초)
    visibility = [0.5] * 100 + [0.9] * 30 + [0.2] * 180
    out = evaluate_tracking_stability(visibility, 30.0, track_cfg)
    assert len(out) == 2
    codes = {e.reason_code for e in out}
    assert codes == {"target_lost", "background_person_interference"}
    severities = {e.severity for e in out}
    assert severities == {"failed", "low_confidence"}


def test_tracking_stability_empty_and_fps_fallback(
    track_cfg: TrackingStabilityConfig,
):
    """edge case — 빈 list 빈 반환 + fps=0 fps_safe fallback."""
    # 빈 list (lock 8-E-13)
    assert evaluate_tracking_stability([], 30.0, track_cfg) == []
    # fps=0 fallback to 30.0 (lock 8-E-14)
    out = evaluate_tracking_stability([0.9] * 30, 0.0, track_cfg)
    assert out == []  # 전체 0.9, trigger X (fallback 작동)


# ============================================================
# L. compute_response_status (§6 status + §8-7 primary, Phase 8-F SoT)
# ============================================================


def test_response_status_empty():
    """빈 entries → success/None/[] (lock 8-F-6)."""
    r = compute_response_status([])
    assert r == ResponseStatusResult(
        status="success", primary_reason_code=None, reason_codes=[]
    )


def test_response_status_failed_status():
    """failed severity 1개 → status='failed' (docs §6 1순위)."""
    r = compute_response_status([ReasonCodeEntry("foot_out_of_frame", "failed")])
    assert r.status == "failed"
    assert r.primary_reason_code == "foot_out_of_frame"


def test_response_status_low_confidence_status():
    """failed 0 + low_conf 1개 → status='low_confidence' (docs §6 2순위)."""
    r = compute_response_status([
        ReasonCodeEntry("camera_unstable", "low_confidence"),
    ])
    assert r.status == "low_confidence"
    assert r.primary_reason_code == "camera_unstable"


def test_response_status_docs_example_1_group_2():
    """docs §8-7-4 예시 1: foot_out_of_frame + foot_not_visible + low_landmark_visibility
    → primary=foot_out_of_frame (그룹 2 첫, 사용자 즉시 해결 가능 코드 우선)."""
    r = compute_response_status([
        ReasonCodeEntry("foot_out_of_frame", "failed"),
        ReasonCodeEntry("foot_not_visible", "failed"),
        ReasonCodeEntry("low_landmark_visibility", "low_confidence"),
    ])
    assert r.status == "failed"
    assert r.primary_reason_code == "foot_out_of_frame"
    # reason_codes PRIORITY 정렬: 그룹 2 foot_out > foot_not > 그룹 8 low_landmark
    assert r.reason_codes == [
        "foot_out_of_frame", "foot_not_visible", "low_landmark_visibility"
    ]


def test_response_status_docs_example_4_group_priority():
    """docs §8-7-4 예시 4: unstable_foot + unstable_knee + low_ic_confidence
    → primary=low_ic_confidence (그룹 9 > 그룹 10)."""
    r = compute_response_status([
        ReasonCodeEntry("unstable_foot_angle", "low_confidence"),
        ReasonCodeEntry("unstable_knee_angle", "low_confidence"),
        ReasonCodeEntry("low_ic_confidence", "low_confidence"),
    ])
    assert r.status == "low_confidence"
    assert r.primary_reason_code == "low_ic_confidence"


def test_response_status_invalid_view_dedup():
    """invalid_view 2 severity 동시 트리거 → reason_codes dedup (lock 8-F-3 α).

    catch Phase 8-B-2 γ 위임: invalid_view failed + low_confidence 동시 entry →
    응답 메시지 reason_codes에 1번만 등장. severity 정보는 status (failed)에 반영.
    primary는 그룹 4 (invalid_view failed) 우선.
    """
    r = compute_response_status([
        ReasonCodeEntry("invalid_view", "failed"),
        ReasonCodeEntry("invalid_view", "low_confidence"),
        ReasonCodeEntry("camera_unstable", "low_confidence"),
    ])
    assert r.status == "failed"
    assert r.primary_reason_code == "invalid_view"  # 그룹 4 (failed 우선)
    assert r.reason_codes == ["invalid_view", "camera_unstable"]


def test_response_status_priority_order_sort():
    """reason_codes PRIORITY 순서 정렬 (lock 8-F-4 α).

    입력 순서 무관 — PRIORITY list 순서 따라 정렬.
    """
    # 입력 순서: trunk → knee → foot → ic → camera (역순)
    r = compute_response_status([
        ReasonCodeEntry("unstable_trunk_angle", "low_confidence"),
        ReasonCodeEntry("unstable_knee_angle", "low_confidence"),
        ReasonCodeEntry("unstable_foot_angle", "low_confidence"),
        ReasonCodeEntry("low_ic_confidence", "low_confidence"),
        ReasonCodeEntry("camera_unstable", "low_confidence"),
    ])
    # PRIORITY 순서: 그룹 8 (camera) > 그룹 9 (low_ic) > 그룹 10 (foot > knee > trunk)
    assert r.reason_codes == [
        "camera_unstable", "low_ic_confidence",
        "unstable_foot_angle", "unstable_knee_angle", "unstable_trunk_angle",
    ]


def test_response_status_priority_list_length():
    """REASON_CODE_PRIORITY 길이 = 18 (17 ReasonCode + invalid_view 2 entry).

    Phase 8-F SoT base 17 → Phase 9-A `target_switch_detected` 추가 = 18.
    Phase 9-B + 메타데이터 진입 시 PRIORITY 확장 anchor (Phase 진입마다 1줄 갱신).
    """
    assert len(REASON_CODE_PRIORITY) == 18
    # invalid_view 2 entry 확인
    invalid_view_entries = [
        (c, s) for c, s in REASON_CODE_PRIORITY if c == "invalid_view"
    ]
    assert len(invalid_view_entries) == 2
    assert ("invalid_view", "failed") in invalid_view_entries
    assert ("invalid_view", "low_confidence") in invalid_view_entries


# ============================================================
# N. §5-1 visibility (Phase 4 — Phase E E-2 sanity → pytest 이식, lock 5건)
# ============================================================
#
# 이식 정합:
# - 원본: scripts/sanity/phase_4_c_integration.py (단일 main 시나리오)
# - 본 이식: 5 case parametrize 활용 (4 ReasonCode trigger + 누적 + severity lookup)
# - scripts/sanity 변경 X (데모 자산 보존, Phase 8-A 보고 §5-1 cleanup 후보 정합 해소)


def _make_pose_for_visibility(
    invalid_landmarks: set[str], invalid_vis: float, normal_vis: float
) -> PoseLandmarks:
    """6 LandmarkPair 합성 — sanity script make_pose 패턴 이식.

    Args:
        invalid_landmarks: visibility 낮출 LandmarkPair 이름 집합.
        invalid_vis: invalid 시 visibility.
        normal_vis: 그 외 visibility.
    """
    def pair(name: str) -> LandmarkPair:
        v = invalid_vis if name in invalid_landmarks else normal_vis
        return LandmarkPair(left=_lm(0.5, 0.5, v), right=_lm(0.5, 0.5, v))

    return PoseLandmarks(
        shoulder=pair("shoulder"),
        hip=pair("hip"),
        knee=pair("knee"),
        ankle=pair("ankle"),
        heel=pair("heel"),
        foot_index=pair("foot_index"),
    )


def test_visibility_frame_normal_pass(cfg: VisibilityCheckConfig):
    """§5-1 정상 — 4 카테고리 모두 통과 (visibility 0.9) → 빈 failed_reasons."""
    pl = _make_pose_for_visibility(set(), invalid_vis=0.0, normal_vis=0.9)
    r = evaluate_frame_visibility(pl, "left", cfg)
    assert r.is_valid
    assert r.failed_reasons == []


@pytest.mark.parametrize(
    "invalid_landmarks,expected_reason",
    [
        # hip+knee+ankle 미달 (lower_body)
        ({"hip", "knee", "ankle"}, "lower_body_not_visible"),
        # heel+foot_index 미달 (foot)
        ({"heel", "foot_index"}, "foot_not_visible"),
        # shoulder 미달 (upper_body)
        ({"shoulder"}, "upper_body_not_visible"),
        # 전체 미달 (overall_avg < 0.5) — 4 카테고리 모두 실패하지만 overall_avg가
        # 그중 가장 큰 시그널, low_landmark_visibility 포함 검증
        ({"shoulder", "hip", "knee", "ankle", "heel", "foot_index"},
         "low_landmark_visibility"),
    ],
)
def test_visibility_frame_per_reason_code(
    cfg: VisibilityCheckConfig,
    invalid_landmarks: set,
    expected_reason: str,
):
    """§5-1 4 ReasonCode trigger parametrize — 각 카테고리 단일 ReasonCode 검증."""
    pl = _make_pose_for_visibility(invalid_landmarks, invalid_vis=0.1, normal_vis=0.9)
    r = evaluate_frame_visibility(pl, "left", cfg)
    assert not r.is_valid
    assert expected_reason in r.failed_reasons


def test_visibility_accumulation_pass(cfg: VisibilityCheckConfig):
    """§5-1 누적 — 70% 통과 → 빈 list (lock 8-B-1 δ list[ReasonCodeEntry])."""
    results = (
        [FrameVisibilityResult(passed_categories={}, category_averages={},
                                failed_reasons=[], is_valid=True) for _ in range(7)]
        + [FrameVisibilityResult(passed_categories={}, category_averages={},
                                 failed_reasons=[], is_valid=False) for _ in range(3)]
    )
    out = evaluate_visibility_accumulation(results, cfg)
    assert out == []


def test_visibility_accumulation_fail(cfg: VisibilityCheckConfig):
    """§5-1 누적 — 50% 통과 → [ReasonCodeEntry('low_landmark_visibility', 'low_confidence')]."""
    results = (
        [FrameVisibilityResult(passed_categories={}, category_averages={},
                                failed_reasons=[], is_valid=True) for _ in range(5)]
        + [FrameVisibilityResult(passed_categories={}, category_averages={},
                                 failed_reasons=[], is_valid=False) for _ in range(5)]
    )
    out = evaluate_visibility_accumulation(results, cfg)
    assert len(out) == 1
    assert out[0].reason_code == "low_landmark_visibility"
    assert out[0].severity == "low_confidence"


def test_visibility_severity_lookup_5_1():
    """§5-1 4 ReasonCode REASON_CODE_SEVERITY 정합 — Phase 4 SoT."""
    assert REASON_CODE_SEVERITY["lower_body_not_visible"] == "low_confidence"
    assert REASON_CODE_SEVERITY["foot_not_visible"] == "failed"
    assert REASON_CODE_SEVERITY["upper_body_not_visible"] == "low_confidence"
    assert REASON_CODE_SEVERITY["low_landmark_visibility"] == "low_confidence"


# ============================================================
# O. §4-3 target_switch_detected (Phase 9-A — 3 신호 AND + 5 frame 연속)
# ============================================================
#
# 시그니처 매트릭스 (anchor B-1~B-4 lock):
# - pelvis 잔차 > 0.15 (sliding window 평균 대비 |차이|, B-3 결정 1 α)
# - |scale 변동률| > 0.20 (hip-shoulder 수직 거리, B-3 결정 2 γ)
# - visibility < 0.4 (frame-level 절대값, B-4 lock)
# - 5 frame 이상 연속 위반 (B-2 lock, target_switch_consecutive_frames=5)


def _pl_for_switch(
    hip_x: float = 0.5,
    hip_y: float = 0.5,
    shoulder_y: float = 0.2,
    vis: float = 0.9,
) -> PoseLandmarks:
    """Phase 9-A 3 신호 시나리오 합성 PoseLandmarks (sanity helper 패턴)."""
    return PoseLandmarks(
        shoulder=LandmarkPair(
            left=_lm(0.45, shoulder_y, vis), right=_lm(0.55, shoulder_y, vis)
        ),
        hip=LandmarkPair(
            left=_lm(hip_x - 0.02, hip_y, vis), right=_lm(hip_x + 0.02, hip_y, vis)
        ),
        knee=LandmarkPair(left=_lm(0.5, 0.65, vis), right=_lm(0.5, 0.65, vis)),
        ankle=LandmarkPair(left=_lm(0.5, 0.85, vis), right=_lm(0.5, 0.85, vis)),
        heel=LandmarkPair(left=_lm(0.48, 0.88, vis), right=_lm(0.52, 0.88, vis)),
        foot_index=LandmarkPair(
            left=_lm(0.52, 0.88, vis), right=_lm(0.52, 0.88, vis)
        ),
    )


@pytest.fixture
def track_switch_cfg() -> "TrackingStabilityConfig":
    """Phase 9-A target_switch_detected fixture (TrackingStabilityConfig 재사용)."""
    from choborunner_ai.config import TrackingStabilityConfig
    return TrackingStabilityConfig()


def _build_scenario(scenario: str) -> list:
    """Phase 9-A sanity 5 시나리오 합성 (parametrize 입력 builder).

    Scenarios:
    - 'normal_30f': 정상 30 frame visibility 0.9 → None
    - 'pass_through_4f': 옆 사람 통과 4 frame (5 미만) → None
    - 'switch_settle_10f': 옆 사람 정착 10 frame → failed
    - 'pelvis_only_10f': pelvis만 위반 (visibility 0.9 유지) → None
    - 'visibility_only_10f': visibility만 붕괴 (pelvis 0.5 유지) → None
    """
    if scenario == "normal_30f":
        return [_pl_for_switch() for _ in range(30)]
    if scenario == "pass_through_4f":
        return (
            [_pl_for_switch() for _ in range(15)]
            + [_pl_for_switch(hip_x=0.8, hip_y=0.3, vis=0.2) for _ in range(4)]
            + [_pl_for_switch() for _ in range(11)]
        )
    if scenario == "switch_settle_10f":
        return (
            [_pl_for_switch() for _ in range(15)]
            + [_pl_for_switch(hip_x=0.8, hip_y=0.3, vis=0.2) for _ in range(10)]
            + [_pl_for_switch() for _ in range(5)]
        )
    if scenario == "pelvis_only_10f":
        return (
            [_pl_for_switch() for _ in range(15)]
            + [_pl_for_switch(hip_x=0.8) for _ in range(10)]  # visibility 0.9 유지
            + [_pl_for_switch() for _ in range(5)]
        )
    if scenario == "visibility_only_10f":
        return (
            [_pl_for_switch() for _ in range(15)]
            + [_pl_for_switch(vis=0.2) for _ in range(10)]  # pelvis 0.5 유지
            + [_pl_for_switch() for _ in range(5)]
        )
    raise ValueError(f"unknown scenario: {scenario}")


@pytest.mark.parametrize(
    "scenario,expected_trigger",
    [
        ("normal_30f", False),
        ("pass_through_4f", False),  # 4 frame < 5 (lock B-2)
        ("switch_settle_10f", True),  # 10 frame >= 5, 메인 trigger
        ("pelvis_only_10f", False),  # AND 조건 (visibility 0.9 유지)
        ("visibility_only_10f", False),  # AND 조건 (pelvis 0.5 유지)
    ],
)
def test_evaluate_target_switch_sanity(
    track_switch_cfg, scenario, expected_trigger
):
    """Phase 9-A sanity 5 case parametrize — 3 신호 AND + 5 frame 연속 정책 검증.

    anchor lock:
    - B-1 β: TrackingStabilityConfig 단일 SoT
    - B-2: 5 frame 연속 정책 (false positive 방지)
    - B-3: 계산식 (pelvis α / scale γ / window α)
    - B-4: visibility frame-level 절대값
    """
    landmarks = _build_scenario(scenario)
    out = evaluate_target_switch(landmarks, 30.0, track_switch_cfg)
    if expected_trigger:
        assert out is not None
        assert out.reason_code == "target_switch_detected"
        assert out.severity == "failed"
    else:
        assert out is None


def test_evaluate_target_switch_fps_zero_safe(track_switch_cfg):
    """fps=0 → fps_safe = 30.0 fallback (Phase 7-A / 8-E 패턴 일관, failed-safe)."""
    landmarks = [_pl_for_switch() for _ in range(30)]
    out = evaluate_target_switch(landmarks, 0.0, track_switch_cfg)
    # 정상 frame이므로 fallback 후에도 trigger X
    assert out is None


def test_evaluate_target_switch_empty_input(track_switch_cfg):
    """빈 list → None (failed-safe, 빈 입력 guard)."""
    out = evaluate_target_switch([], 30.0, track_switch_cfg)
    assert out is None
