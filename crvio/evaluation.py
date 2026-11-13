"""Sequence evaluation with real conditional visual execution.

Reuses the baseline KITTI data partition and metric code, so accuracy numbers stay
comparable to B0. Only the network call differs.
"""
from types import SimpleNamespace

import numpy as np
import torch

from utils.kitti_eval import data_partition, kitti_eval, plotPath_2D
from utils.utils import saveSequence

from .perturb import apply_perturbation


def make_policy_fn(kind, value=None, generator=None, dt=0.1):
    """Comparison policies; fn(s, elapsed) -> (a, p). They see only causal quantities (PCM passes s = None)."""
    if kind == 'random':
        def fn(s, elapsed):
            p = elapsed.new_full(elapsed.shape, float(value))
            return (torch.rand(p.shape, device=p.device, generator=generator) < p).float(), p
    elif kind == 'interval':
        def fn(s, elapsed):
            # Call every `value` intervals: value * dt seconds after the last call.
            a = (elapsed >= (float(value) - 0.5) * dt).float()
            return a, a
    elif kind == 'all':
        def fn(s, elapsed):
            a = torch.ones_like(elapsed)
            return a, a
    else:
        raise ValueError(kind)
    return fn


@torch.no_grad()
def run_sequence(model, partition, device, rule=None, policy_fn=None, perturb=None, perturb_seed=0):
    model.eval()
    state = None
    poses, poses0, a_all, elapsed_all, p_all = [], [], [], [], []
    for i in range(len(partition)):
        imgs, imus, _ = partition[i]
        imgs, imus = imgs.unsqueeze(0).to(device), imus.unsqueeze(0).to(device)
        imu_mask = None
        if perturb is not None:
            imgs, imus, imu_mask, _, _ = apply_perturbation(imgs, imus, perturb, seed=perturb_seed + i)
        out, state = model.forward_infer(imgs, imus, imu_mask=imu_mask, state=state, is_first=i == 0,
                                         rule=rule, policy_fn=policy_fn)
        poses.append(out['pose'][0].cpu().numpy())
        a_all.append(out['a'][0, :, 0].cpu().numpy())
        elapsed_all.append(out['elapsed'][0, :, 0].cpu().numpy())
        if 'pose0' in out:
            poses0.append(out['pose0'][0].cpu().numpy())
        if 'p' in out:
            p_all.append(out['p'][0, :, 0].cpu().numpy())
    join = lambda x, f: f(x) if len(x) else None
    return {'pose': np.vstack(poses), 'pose0': join(poses0, np.vstack),
            'a': np.hstack(a_all), 'elapsed': np.hstack(elapsed_all), 'p': join(p_all, np.hstack)}


class Evaluator:
    """Loads one partition per sequence once, like the baseline tester."""

    def __init__(self, opt, sequences):
        self.opt, self.sequences = opt, sequences
        args = SimpleNamespace(data_dir=str(opt.data_dir).rstrip('/'), seq_len=opt.seq_len,
                               img_h=opt.img_h, img_w=opt.img_w)
        self.partitions = {s: data_partition(args, s) for s in sequences}
        self.results = {}

    def evaluate(self, model, device, rule=None, policy_fn=None, perturb=None, perturb_seed=0):
        errors = {}
        self.results = {}
        for seq, part in self.partitions.items():
            r = run_sequence(model, part, device, rule, policy_fn, perturb, perturb_seed)
            # kitti_eval prepends the first decision itself, so drop it here (baseline format).
            est, gt_mat, t_rel, r_rel, t_rmse, r_rmse, usage, speed = kitti_eval(r['pose'], r['a'][1:], part.poses_rel)
            errors[seq] = dict(t_rel=t_rel, r_rel=r_rel, t_rmse=t_rmse, r_rmse=r_rmse, usage=usage)
            self.results[seq] = dict(est=est, gt=gt_mat, speed=speed, **r)
        errors['mean'] = {k: float(np.mean([errors[s][k] for s in self.partitions]))
                          for k in ('t_rel', 'r_rel', 't_rmse', 'r_rmse', 'usage')}
        return errors

    def save(self, out_dir, window_size=30):
        for seq, r in self.results.items():
            saveSequence(r['est'], f'{out_dir}/{seq}_pred.txt')
            np.savez_compressed(f'{out_dir}/{seq}_raw.npz', **{k: r[k] for k in ('pose', 'a', 'elapsed') if r[k] is not None})
            plotPath_2D(seq, r['gt'], r['est'], str(out_dir), r['a'][1:], r['speed'], window_size)


def calibrate_threshold(evaluator, model, device, target_usage, make_rule, lo, hi, iters=8):
    """Bisection on validation sequences for a threshold reaching a target usage (%)."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rule, policy_fn = make_rule(mid)
        usage = evaluator.evaluate(model, device, rule=rule, policy_fn=policy_fn)['mean']['usage']
        if best is None or abs(usage - target_usage) < abs(best[1] - target_usage):
            best = (mid, usage)
        if usage > target_usage:      # higher threshold, fewer calls
            lo = mid
        else:
            hi = mid
    return best
