"""Measurement perturbations (design doc section 4.5).

Applied to a batch on the training device. Pose labels are never touched, and
nothing here changes image geometry, so the supervision stays valid.

Returns per-interval correspondence flags q_v / q_i in [0, 1]. They come from the
*known* perturbation settings and only weight the auxiliary losses during training;
the network never receives them.
"""
from dataclasses import dataclass
import random

import torch
import torch.nn.functional as F

VISUAL_TYPES = ('noise', 'photometric', 'occlusion', 'blur')
# Extra kinds, not in VISUAL_TYPES (the default kinds of test_crvio.py --test_perturb). 'blank': the frame is present
# but carries no correspondence — covered lens, darkness, a frozen stream or >= 90% occlusion (2026-11-09). PCM
# training uses it by default (crvio/config.py --visual_types): the gate loss needs its frozen frames.
EXTRA_VISUAL_TYPES = ('blank',)
ALL_VISUAL_TYPES = VISUAL_TYPES + EXTRA_VISUAL_TYPES
IMU_TYPES = ('noise', 'bias', 'missing')
IMU_FREQ = 10

# (low, high) per severity. Images live in [-0.5, 0.5]; IMU units are m/s^2 and rad/s.
RANGES = {
    'v_noise': {'light': (0.01, 0.03), 'severe': (0.05, 0.15)},
    'v_gain': {'light': (0.75, 1.3), 'severe': (0.35, 2.0)},
    'v_gamma': {'light': (0.8, 1.25), 'severe': (0.5, 2.0)},
    'v_occl_area': {'light': (0.03, 0.12), 'severe': (0.3, 0.6)},
    'v_blur_sigma': {'light': (0.5, 1.5), 'severe': (2.5, 5.0)},
    'acc_noise': {'light': (0.02, 0.1), 'severe': (0.2, 1.0)},
    'gyr_noise': {'light': (0.002, 0.01), 'severe': (0.02, 0.1)},
    'acc_bias': {'light': (0.02, 0.1), 'severe': (0.2, 0.5)},
    'gyr_bias': {'light': (0.001, 0.005), 'severe': (0.01, 0.05)},
    'imu_gap': {'light': (1, 2), 'severe': (6, 11)},   # missing samples per interval
}
Q_PHOTOMETRIC_SEVERE = 0.5     # strong exposure change: correspondence weakened, not removed


@dataclass
class PerturbConfig:
    mode: str = 'none'              # none | visual | imu | mixed
    visual_types: tuple = VISUAL_TYPES
    imu_types: tuple = IMU_TYPES
    p_clean: float = 0.4            # mixed: unperturbed share of the batch
    p_joint: float = 0.1            # mixed: both modalities perturbed
    p_severe: float = 0.3
    q_severe: float = 0.0           # weight of the aux losses on severely degraded intervals
    scale: float = 1.0              # >1 evaluates unseen strengths
    full_window: bool = False       # test: perturb the whole chunk instead of a random span
    misalign_samples: int = 0       # diagnostic only, never a positive pair


def _uniform(key, severity, scale, rng):
    lo, hi = RANGES[key][severity]
    return (lo + (hi - lo) * rng.random()) * scale


def _span(length, full, rng):
    if full:
        return 0, length
    size = rng.randint(1, length)
    start = rng.randint(0, length - size)
    return start, start + size


def _gaussian_blur(frames, sigma):
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=frames.dtype, device=frames.device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    c = frames.shape[1]
    kx = kernel.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = kernel.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    out = F.pad(frames, (radius, radius, 0, 0), mode='replicate')
    out = F.conv2d(out, kx, groups=c)
    out = F.pad(out, (0, 0, radius, radius), mode='replicate')
    return F.conv2d(out, ky, groups=c)


def _perturb_frames(frames, kind, severity, scale, rng, info=None):
    """frames: (F, C, H, W) in [-0.5, 0.5]. Geometry is preserved by every branch."""
    if kind == 'noise':
        sigma = _uniform('v_noise', severity, scale, rng)
        frames = frames + sigma * torch.randn_like(frames)
    elif kind == 'photometric':
        gain = _uniform('v_gain', severity, 1.0, rng)
        gamma = _uniform('v_gamma', severity, 1.0, rng)
        frames = (frames + 0.5).clamp(0, 1) ** gamma * gain - 0.5
    elif kind == 'occlusion':
        h, w = frames.shape[-2:]
        area = min(_uniform('v_occl_area', severity, scale, rng), 0.9)
        side_h = max(1, int(h * area ** 0.5))
        side_w = max(1, int(w * area ** 0.5))
        top, left = rng.randint(0, h - side_h), rng.randint(0, w - side_w)
        frames = frames.clone()
        frames[..., top:top + side_h, left:left + side_w] = rng.uniform(-0.5, 0.5)
    elif kind == 'blur':
        frames = _gaussian_blur(frames, _uniform('v_blur_sigma', severity, scale, rng))
    elif kind == 'blank':
        sub = rng.choice(['uniform', 'dark', 'frozen', 'cover'])
        if info is not None:
            info['sub'] = sub
        if sub == 'uniform':      # covered lens: one colour per channel plus slight sensor noise
            colour = frames.new_tensor([rng.uniform(-0.5, 0.5) for _ in range(frames.shape[1])]).view(1, -1, 1, 1)
            frames = colour + 0.01 * torch.randn_like(frames)
        elif sub == 'dark':       # near darkness: very low gain plus sensor noise
            frames = (frames + 0.5) * rng.uniform(0.0, 0.08) - 0.5 + rng.uniform(0.005, 0.03) * torch.randn_like(frames)
        elif sub == 'frozen':     # stalled stream: every frame of the span repeats its first frame
            frames = frames[:1].expand_as(frames).clone()
        else:                     # >= 90% of the frame occluded
            h, w = frames.shape[-2:]
            area = rng.uniform(0.9, 1.0)
            side_h, side_w = max(1, int(h * area ** 0.5)), max(1, int(w * area ** 0.5))
            top, left = rng.randint(0, h - side_h), rng.randint(0, w - side_w)
            frames = frames.clone()
            frames[..., top:top + side_h, left:left + side_w] = rng.uniform(-0.5, 0.5)
    else:
        raise ValueError(kind)
    return frames.clamp(-0.5, 0.5)


def _perturb_imu(imu, mask, kind, severity, scale, rng):
    """imu / mask: (S, 6) / (S, 1). Accelerometer and gyro use their own units."""
    if kind == 'noise':
        std = imu.new_tensor([_uniform('acc_noise', severity, scale, rng)] * 3
                             + [_uniform('gyr_noise', severity, scale, rng)] * 3)
        imu = imu + std * torch.randn_like(imu)
    elif kind == 'bias':
        bias = imu.new_tensor([_uniform('acc_bias', severity, scale, rng)] * 3
                              + [_uniform('gyr_bias', severity, scale, rng)] * 3)
        bias = bias * torch.randn(6, device=imu.device).sign()
        # A bias is time-correlated: constant offset plus a slow random walk.
        walk = torch.cumsum(0.02 * bias.abs() * torch.randn_like(imu), dim=0)
        imu = imu + bias + walk
    elif kind == 'missing':
        lo, hi = RANGES['imu_gap'][severity]
        gap = min(int(rng.randint(lo, hi) * scale), imu.shape[0])
        start = rng.randint(0, imu.shape[0] - gap)
        imu = imu.clone()
        mask = mask.clone()
        imu[start:start + gap] = 0.0     # the mask channel tells the encoder these are absent
        mask[start:start + gap] = 0.0
    else:
        raise ValueError(kind)
    return imu, mask


def apply_perturbation(imgs, imus, cfg, seed=None, return_blank=False, return_frozen=False):
    """imgs (B, T+1, C, H, W), imus (B, 10T+1, 6).

    Returns imgs', imus', imu_mask (B, 10T+1, 1), q_v (B, T), q_i (B, T), and with return_blank also
    blank (B, T): 1 on the intervals that read a 'blank' frame (no visual information at all).
    return_frozen (PCM, 2026-11-10) appends frozen (B, T): 1 on the intervals of a 'frozen' blank span that read
    two copies of its first frame (zero flow while the vehicle moves) or the copy and the next real frame (a jump):
    the confidently wrong images. The interval before the span reads real frames and is not marked (blank marks it).
    """
    out = _apply_perturbation(imgs, imus, cfg, seed)
    if return_frozen:
        return out
    return out[:6] if return_blank else out[:5]


def _apply_perturbation(imgs, imus, cfg, seed):
    B, n_frames = imgs.shape[:2]
    T = n_frames - 1
    imu_mask = torch.ones(*imus.shape[:2], 1, dtype=imus.dtype, device=imus.device)
    q_v = torch.ones(B, T, dtype=imus.dtype, device=imus.device)
    q_i = torch.ones(B, T, dtype=imus.dtype, device=imus.device)

    if cfg.misalign_samples:
        # Sync diagnostic: the windows no longer describe the same motion.
        imus = torch.roll(imus, shifts=cfg.misalign_samples, dims=1)
        return imgs, imus, imu_mask, torch.zeros_like(q_v), torch.zeros_like(q_i), torch.zeros_like(q_v), \
            torch.zeros_like(q_v)
    blank = torch.zeros_like(q_v)
    frozen = torch.zeros_like(q_v)
    if cfg.mode == 'none':
        return imgs, imus, imu_mask, q_v, q_i, blank, frozen

    rng = random.Random(seed) if seed is not None else random
    imgs, imus = imgs.clone(), imus.clone()
    for b in range(B):
        if cfg.mode == 'mixed':
            u = rng.random()
            single = (1.0 - cfg.p_clean - cfg.p_joint) / 2.0
            do_v = cfg.p_clean <= u < cfg.p_clean + single or u >= 1.0 - cfg.p_joint
            do_i = cfg.p_clean + single <= u < cfg.p_clean + 2 * single or u >= 1.0 - cfg.p_joint
        else:
            do_v, do_i = cfg.mode == 'visual', cfg.mode == 'imu'

        if do_v:
            severity = 'severe' if rng.random() < cfg.p_severe else 'light'
            start, end = _span(n_frames, cfg.full_window, rng)
            kind = rng.choice(list(cfg.visual_types))
            if kind == 'blank':
                severity = 'severe'     # no correspondence at all: q_v = q_severe
            info = {}
            imgs[b, start:end] = _perturb_frames(imgs[b, start:end], kind, severity, cfg.scale, rng, info)
            if info.get('sub') == 'frozen' and end - start >= 2:     # a one-frame span changes no frame
                frozen[b, start:min(T, end)] = 1.0
            if severity == 'severe':
                # Interval t reads frames t and t+1.
                q = Q_PHOTOMETRIC_SEVERE if kind == 'photometric' else cfg.q_severe
                q_v[b, max(0, start - 1):min(T, end)] = q
                if kind == 'blank':
                    blank[b, max(0, start - 1):min(T, end)] = 1.0

        if do_i:
            severity = 'severe' if rng.random() < cfg.p_severe else 'light'
            kind = rng.choice(list(cfg.imu_types))
            if cfg.full_window:
                s0, s1, t0, t1 = 0, imus.shape[1], 0, T
            else:
                t0, t1 = _span(T, False, rng)
                s0, s1 = t0 * IMU_FREQ, t1 * IMU_FREQ + 1
            if kind == 'missing':
                # One gap per interval, so 'severe' means most of that window is gone.
                for t in range(t0, t1):
                    a, c = t * IMU_FREQ, t * IMU_FREQ + IMU_FREQ + 1
                    imus[b, a:c], imu_mask[b, a:c] = _perturb_imu(
                        imus[b, a:c], imu_mask[b, a:c], kind, severity, cfg.scale, rng)
            else:
                imus[b, s0:s1], imu_mask[b, s0:s1] = _perturb_imu(
                    imus[b, s0:s1], imu_mask[b, s0:s1], kind, severity, cfg.scale, rng)
            if severity == 'severe':
                q_i[b, t0:t1] = cfg.q_severe
    return imgs, imus, imu_mask, q_v, q_i, blank, frozen
