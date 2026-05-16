"""부록 D 합성 stride 시계열 헬퍼 (Phase 5-B 재사용 자산).

docs/2-3-4 부록 D-2 가정 1:1 재현:
- Gaussian noise σ = 0.003 ~ 0.012 (정규화 좌표 기준)
- Stride 주기 0.7s (러너별 변동성 미반영)
- FPS 30
- 분량 5초 (150 frame)
- Ground truth IC 7개 (stride 0.7s × 7 ≈ 4.9s)
- Heel forward extension peak가 실제 IC보다 1 frame lead
  (running biomechanics 일반 패턴)

본 헬퍼는 Phase 5-B IC detector 검증용 — Stage 1 단독 MAE 33.3ms / Stage 1+2
hybrid MAE 9.6ms 재현 sanity (부록 D-3).

⚠️ 미반영 (부록 D-5 한계 명시):
- Occlusion / motion blur / visibility drop
- Stride 주기 변동성 (가속/감속/피로)
- 신발 종류, 카메라 각도 변화
- Strike pattern별 차이 (RFS / MFS / FFS)

모션 모델:
- pelvis_x = 0.5 정적 (트레드밀 가정 + 카메라 고정)
- heel_rel_x(t) = A_x * cos(2π * (t - heel_peak_t) / stride_frames)
  · heel_peak_t = ic_frame - 1 (lead) → t=ic-1에 forward peak
- heel.y(t) = base_y + A_y * cos(2π * (t - ic_frame) / stride_frames)
  · t=ic_frame에 max (y 증가 = 화면 아래쪽 = 발이 가장 아래쪽 = 지면 닿음)
- foot_index.y = heel.y + 작은 offset (foot_index는 발끝, heel보다 약간 더 아래)
- 다른 landmark (shoulder, hip, knee, ankle) 정적 + visibility 0.9
- 반대측 다리는 정적 (self-occlusion 미반영)

사용 예:
    from tests.fixtures.synthetic_stride import (
        generate_synthetic_stride_series,
        DEFAULT_FPS, DEFAULT_GT_IC_FRAMES,
    )
    landmarks_series, gt_ic_frames = generate_synthetic_stride_series(noise_sigma=0.005)
"""
from __future__ import annotations

import math

import numpy as np

from choborunner_ai.pose_extractor import Landmark, LandmarkPair, PoseLandmarks


DEFAULT_FPS = 30
DEFAULT_STRIDE_SEC = 0.7
DEFAULT_DURATION_SEC = 5.0
DEFAULT_HEEL_PEAK_LEAD_FRAMES = 1  # heel forward peak lead 실제 IC보다 1 frame 일찍
DEFAULT_FIRST_IC_OFFSET = 10       # 첫 IC = frame 10 (warmup buffer 확보)


def default_gt_ic_frames(
    fps: int = DEFAULT_FPS,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    duration_sec: float = DEFAULT_DURATION_SEC,
    first_ic_offset: int = DEFAULT_FIRST_IC_OFFSET,
) -> list[int]:
    """기본 ground truth IC frame 인덱스 리스트.

    stride_frames = round(stride_sec * fps). 영상 길이 초과는 제외.
    """
    stride_frames = int(round(stride_sec * fps))
    n_frames = int(round(duration_sec * fps))
    return [
        first_ic_offset + i * stride_frames
        for i in range(7)
        if first_ic_offset + i * stride_frames < n_frames
    ]


DEFAULT_GT_IC_FRAMES = default_gt_ic_frames()  # [10, 31, 52, 73, 94, 115, 136]


def generate_synthetic_stride_series(
    fps: int = DEFAULT_FPS,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    duration_sec: float = DEFAULT_DURATION_SEC,
    noise_sigma: float = 0.005,
    heel_peak_lead_frames: int = DEFAULT_HEEL_PEAK_LEAD_FRAMES,
    analysis_side: str = "left",
    seed: int = 42,
) -> tuple[list[PoseLandmarks], list[int]]:
    """합성 PoseLandmarks 시퀀스 + ground truth IC frames 생성.

    부록 D-2 가정 정합. seed 고정으로 deterministic.

    Args:
        fps: 30 (부록 D 가정).
        stride_sec: 0.7 (부록 D 가정).
        duration_sec: 5.0 (부록 D 가정, 150 frame).
        noise_sigma: Gaussian noise σ (정규화 좌표). 부록 D 0.003~0.012 범위.
        heel_peak_lead_frames: heel forward peak가 실제 IC보다 일찍 N frame.
        analysis_side: 'left' 또는 'right' — 해당측 heel/foot만 motion.
        seed: numpy random seed.

    Returns:
        (landmarks_series, gt_ic_frames):
        - landmarks_series: list[PoseLandmarks] (len = duration_sec * fps)
        - gt_ic_frames: list[int] ground truth IC frame indices.
    """
    rng = np.random.default_rng(seed)
    n_frames = int(round(duration_sec * fps))
    stride_frames = stride_sec * fps  # 21.0 (float OK for cos cycle)

    gt_ic_frames = default_gt_ic_frames(fps, stride_sec, duration_sec)
    first_ic = gt_ic_frames[0] if gt_ic_frames else 0

    # 모션 진폭 (정규화 좌표)
    A_x_heel = 0.15  # heel forward/backward 진폭 (pelvis 대비)
    A_y_heel = 0.08  # heel vertical 진폭

    # 정적 base 좌표
    pelvis_x = 0.5
    hip_y = 0.55
    shoulder_y = 0.25
    knee_y = 0.7
    ankle_y = 0.85
    heel_base_y = 0.90
    foot_offset_y = 0.02  # foot_index는 heel보다 약간 더 아래
    foot_offset_x = 0.02  # foot_index는 heel보다 약간 더 앞

    heel_peak_t = first_ic - heel_peak_lead_frames  # heel.x peak 위치

    landmarks_series: list[PoseLandmarks] = []

    def static(x: float, y: float, vis: float = 0.9) -> Landmark:
        return Landmark(x=x, y=y, visibility=vis)

    for t in range(n_frames):
        # heel_rel_x: peak at t = heel_peak_t (실제 IC - 1)
        phase_x = 2 * math.pi * (t - heel_peak_t) / stride_frames
        heel_rel_x_clean = A_x_heel * math.cos(phase_x)
        heel_x_clean = pelvis_x + heel_rel_x_clean

        # heel.y: max (y 가장 큰값 = 가장 아래쪽) at t = first_ic
        phase_y = 2 * math.pi * (t - first_ic) / stride_frames
        heel_y_amp = A_y_heel * math.cos(phase_y)
        heel_y_clean = heel_base_y + heel_y_amp

        # Gaussian noise
        heel_x_noisy = heel_x_clean + rng.normal(0, noise_sigma)
        heel_y_noisy = heel_y_clean + rng.normal(0, noise_sigma)
        foot_x_noisy = (heel_x_clean + foot_offset_x) + rng.normal(0, noise_sigma)
        foot_y_noisy = (heel_y_clean + foot_offset_y) + rng.normal(0, noise_sigma)

        # 분석측 motion, 반대측 정적
        if analysis_side == "left":
            heel_pair = LandmarkPair(
                left=static(heel_x_noisy, heel_y_noisy),
                right=static(pelvis_x, heel_base_y),
            )
            foot_pair = LandmarkPair(
                left=static(foot_x_noisy, foot_y_noisy),
                right=static(pelvis_x + foot_offset_x, heel_base_y + foot_offset_y),
            )
        else:
            heel_pair = LandmarkPair(
                left=static(pelvis_x, heel_base_y),
                right=static(heel_x_noisy, heel_y_noisy),
            )
            foot_pair = LandmarkPair(
                left=static(pelvis_x + foot_offset_x, heel_base_y + foot_offset_y),
                right=static(foot_x_noisy, foot_y_noisy),
            )

        # 정적 hip/shoulder/knee/ankle (motion 무관, visibility 0.9)
        hip_pair = LandmarkPair(
            left=static(pelvis_x - 0.05, hip_y),
            right=static(pelvis_x + 0.05, hip_y),
        )
        shoulder_pair = LandmarkPair(
            left=static(pelvis_x - 0.05, shoulder_y),
            right=static(pelvis_x + 0.05, shoulder_y),
        )
        knee_pair = LandmarkPair(
            left=static(pelvis_x - 0.05, knee_y),
            right=static(pelvis_x + 0.05, knee_y),
        )
        ankle_pair = LandmarkPair(
            left=static(pelvis_x - 0.05, ankle_y),
            right=static(pelvis_x + 0.05, ankle_y),
        )

        pl = PoseLandmarks(
            shoulder=shoulder_pair,
            hip=hip_pair,
            knee=knee_pair,
            ankle=ankle_pair,
            heel=heel_pair,
            foot_index=foot_pair,
        )
        landmarks_series.append(pl)

    return landmarks_series, gt_ic_frames
