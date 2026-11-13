"""Training recipe of PCM v3 (moved out of train_crvio.py on 2026-11-11): stillness windows,
speed-balanced sampling, the least-squares gyro map, AdamW parameter groups and the stage-2 freeze of the visual
encoder, the EMA of the weights, and BatchNorm recalibration (train_crvio.py --bn_recalib_batches, test_crvio.py
--bn_recalib)."""
import copy

import numpy as np
import torch
import torch.nn as nn


def is_visual_encoder_param(name):
    """FlowNet convolutions and visual_head: the expensive encoder that stage 2 may keep fixed."""
    return name.startswith('Feature_net.') and not name.startswith('Feature_net.inertial_encoder.')


def visual_encoder_modules(model):
    return [m for n, m in model.Feature_net.named_children() if n != 'inertial_encoder']


def param_groups(model, stage, opt, freeze_est=False):
    policy = list(model.Policy_net.parameters()) if model.Policy_net is not None else []
    policy_ids = {id(p) for p in policy}
    frozen_visual = stage == 2 and opt.stage2_freeze_visual
    est = []
    for n, p in model.named_parameters():
        if id(p) in policy_ids:
            continue
        if frozen_visual and is_visual_encoder_param(n):
            p.requires_grad = False
            continue
        est.append(p)
    if stage == 1:
        for p in policy:
            p.requires_grad = False
        for p in est:
            p.requires_grad = True
        return [{'params': est, 'lr': opt.lr_stage1, 'peak': opt.lr_stage1, 'name': 'est'}]
    for p in policy:
        p.requires_grad = True
    for p in est:
        p.requires_grad = not freeze_est
    return [{'params': policy, 'lr': opt.lr_policy, 'peak': opt.lr_policy, 'name': 'policy'},
            {'params': est, 'lr': opt.lr_est, 'peak': opt.lr_est, 'name': 'est'}]


def make_optimizer(groups, opt):
    """adam: coupled L2 as in B0 (weight_decay). adamw (the PCM recipe): decoupled decay on matrices only. Under coupled L2
    Adam normalizes the decay by the gradient scale, so parameters with tiny loss gradients (gates, G_s, S) were
    pulled to zero ~1000x faster than the regressor (design review 2026-11-10)."""
    if opt.optimizer == 'adamw':
        split = []
        for g in groups:
            split.append(dict(g, params=[p for p in g['params'] if p.ndim > 1], weight_decay=opt.adamw_wd))
            split.append(dict(g, params=[p for p in g['params'] if p.ndim <= 1], weight_decay=0.0,
                              name=g['name'] + '_nd'))
        return torch.optim.AdamW([g for g in split if g['params']], betas=(0.9, 0.999), eps=1e-8)
    return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=opt.weight_decay)


class EMA:
    """Exponential moving average of the weights (buffers copied). Validation and the saved 'model' use it."""

    def __init__(self, model, decay):
        self.decay, self.n = decay, 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.grad = None
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)


def apply_stillness(imgs, imus, gts, p):
    """Replace a share p of the windows by a stationary window. Frames repeat the first frame plus
    sensor noise, the accelerometer reads the window's mean specific force plus noise, the gyro reads noise,
    every label is the identity. The training set has two stops of ~20 frames; KITTI 07 stops for ~5 s, where
    every model kept moving (B0 ~9 m of false motion)."""
    if p <= 0:
        return imgs, imus, gts
    idx = (torch.rand(imgs.shape[0], device=imgs.device) < p).nonzero(as_tuple=True)[0]
    if len(idx) == 0:
        return imgs, imus, gts
    imgs, imus, gts = imgs.clone(), imus.clone(), gts.clone()
    imgs[idx] = (imgs[idx, :1] + 0.005 * torch.randn_like(imgs[idx])).clamp(-0.5, 0.5)
    acc = imus[idx, :, :3].mean(1, keepdim=True)
    imus[idx, :, :3] = acc + 0.02 * torch.randn_like(imus[idx, :, :3])
    imus[idx, :, 3:] = 0.002 * torch.randn_like(imus[idx, :, 3:])
    gts[idx] = 0.0
    return imgs, imus, gts


def speed_balanced_weights(train_set, bins=12):
    """Window weights count(speed bin)^-0.5 (mean speed of the window's relative translations)."""
    speeds = np.array([np.linalg.norm(np.asarray(s['gts'])[:, 3:], axis=1).mean() for s in train_set.samples])
    edges = np.linspace(speeds.min(), speeds.max() + 1e-9, bins + 1)
    idx = np.clip(np.digitize(speeds, edges) - 1, 0, bins - 1)
    w = np.bincount(idx, minlength=bins)[idx].astype(np.float64) ** -0.5
    return torch.as_tensor(w / w.mean(), dtype=torch.double), speeds


def fit_gyro_map(train_set, dt, stride=5):
    """Least-squares 3x3 map from the integrated gyro of an interval (IMU axes) to the rotation label (Euler, camera
    axes), on training windows only. Returns (M, per-axis R^2). On KITTI 00/02/08 M ~ [[0,-1,0],[0,0,-1],[1,0,0]]
    and the yaw R^2 is 0.994: the gyro alone explains yaw, so the shortcut starts there."""
    X, Y = [], []
    for s in train_set.samples[::stride]:
        imu, gts = np.asarray(s['imus'], dtype=np.float64), np.asarray(s['gts'], dtype=np.float64)
        g = imu[:, 3:6]
        for k in range(gts.shape[0]):
            w = g[10 * k:10 * k + 11]
            X.append(0.5 * (w[:-1] + w[1:]).sum(0) * dt / 10.0)
            Y.append(gts[k, :3])
    X, Y = np.array(X), np.array(Y)
    M = np.linalg.lstsq(X, Y, rcond=None)[0].T
    r2 = 1.0 - ((Y - X @ M.T) ** 2).mean(0) / Y.var(0).clip(min=1e-12)
    return M, r2


@torch.no_grad()
def recalibrate_batchnorm(model, loader, run_batch, max_batches):
    """Recompute every BatchNorm running mean / var as the plain average over `max_batches` batches
    of `loader`, each passed through `run_batch(batch)` in train mode. Used with clean windows after
    perturbation training, whose perturbed batches inflate the statistics. Returns the batch count."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns or max_batches <= 0:
        return 0
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None    # cumulative average
    was_training = model.training
    model.train()
    n = 0
    for batch in loader:
        if n >= max_batches:
            break
        run_batch(batch)
        n += 1
    for m, momentum in zip(bns, momenta):
        m.momentum = momentum
    model.train(was_training)
    return n
