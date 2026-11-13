"""Mechanism checks for PCM v3 checkpoints (2026-11-10, revised after review 2 and for v3 on 2026-11-11). Older designs:
PCM v1 / v2 (PCM, PCMC) in /root/VIO/snapshot_pcm12_20261111, the non-PCM designs in /root/VIO/legacy_crvio_20261111
(each snapshot has its own copy of this tool).

Pre-registered criteria: results/SUMMARY.md (PCM first experiment). Per test sequence, stage-1 weights:
  gains / variances per blank kind   vision at every step on a whole-sequence blank of one kind (uniform, dark, cover:
                                     textureless; frozen: confidently wrong), translation gain pi*k, rotation gain,
                                     sigma_v (std) and var_v ratios to the normal run, gate pi on frozen
  vision effect                      mean |t1 - t0| (translation of T1 minus T0 at called steps), normal vs each blank
                                     kind: how much the image actually moves the pose (independent of k)
  vision used                        t_rmse with every image rejected (pi = 0: gate_a set to -1e4) relative to full
                                     vision, and the same ratio for 0% vision (comparable with non-PCM runs)
  prior is cross-modal               R^2 of the prior translation readout (3 axes and forward) against the truth at
                                     random 25% vision, intervals >= 2 intervals after the last call, with the true IMU vs
                                     the IMU windows of another part of the same sequence (in distribution, wrong
                                     motion); both runs share one schedule. Pre-registered: the drop must be >= 0.2
  sigma_p calibrated (validation)    Spearman(log var_p, log prior error |trans_p - y / b_d|^2) over all intervals at
                                     random 25% vision, raw and after removing the rank dependence on the intervals
                                     since the last call (info['elapsed'] / dt)
  static images                      yaw slope (estimated vs true yaw) when every image pair is a frame with itself
  stop                               building2-style: on intervals with true speed < 0.045 m/s, the net and path
                                     length of the estimated translation at full vision, and the mean p_still (v3:
                                     scales the translation) there and on the moving intervals
Output: JSON on stdout and in --out.
Usage: VIO_SEG_LENGTHS=5,10,15,20 python tools/pcm_diag.py --ckpt results/<run>/checkpoints/stage1_last.pth [--out f]
"""
import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crvio.config import load_opt  # noqa: E402
from crvio.vio import build_model  # noqa: E402
from crvio.perturb import _perturb_frames  # noqa: E402
from utils.kitti_eval import data_partition, kitti_eval  # noqa: E402

BLANK_KINDS = ('uniform', 'dark', 'cover', 'frozen')
STILL_SPEED = 0.045      # m/s, as results/analysis/b0_translation (cond_attrib.py)


class ForcedBlank(random.Random):
    """random.Random whose choice of the blank sub-kind is fixed (the rest of _perturb_frames keeps its draws)."""

    @classmethod
    def make(cls, kind, seed):
        rng = cls(seed)          # random.Random.__new__ takes only the seed
        rng.kind = kind
        return rng

    def choice(self, seq):
        if list(seq) == ['uniform', 'dark', 'frozen', 'cover']:
            return self.kind
        return super().choice(seq)


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def spearman(x, y):
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]) if len(x) > 2 else float('nan')


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.stack([rz, np.ones_like(rz)], 1)
    res = lambda r: r - A @ np.linalg.lstsq(A, r, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def r2(pred, true):
    return float(1.0 - ((pred - true) ** 2).sum() / max(((true - true.mean()) ** 2).sum(), 1e-12))


@torch.no_grad()
def run(model, part, device, schedule, blank=None, static=False, imu_swap=False, max_chunks=0, seed=0):
    """Inference like PCMVIO.forward_infer, step by step. schedule: 0/1 per interval of the sequence (index 0 is always
    visual). blank: one of BLANK_KINDS applied to every chunk. imu_swap: IMU windows of the chunk half a sequence away."""
    model.eval()
    torch.manual_seed(seed)      # the uniform / dark blank kinds draw sensor noise from the global torch RNG
    state, rec, k = None, {}, 0
    n_chunks = len(part) if not max_chunks else min(max_chunks, len(part))
    for i in range(n_chunks):
        imgs, imus, _ = part[i]
        if imu_swap:     # same length as this chunk (the last chunk of a sequence is shorter)
            j = (i + len(part) // 2) % len(part)
            src = part[j][1]
            if src.shape[0] < imus.shape[0]:
                src = part[(j - 1) % len(part)][1]
            imus = src[:imus.shape[0]]
        imgs, imus = imgs.unsqueeze(0).to(device), imus.unsqueeze(0).to(device)
        if blank is not None:
            imgs = imgs.clone()
            imgs[0] = _perturb_frames(imgs[0], 'blank', 'severe', 1.0, ForcedBlank.make(blank, seed + i))
        mask = torch.ones(*imus.shape[:2], 1, device=device, dtype=imus.dtype)
        T = imgs.shape[1] - 1
        windows = model.imu_windows(imus, mask, T)
        state = model.init_state(1, device) if state is None else state
        for t in range(T):
            force = torch.as_tensor([[float(schedule[k]) if k > 0 else 1.0]], device=device)
            prev = imgs[:, t + 1] if static else imgs[:, t]
            pose, state, info = model.inference_step(prev, imgs[:, t + 1], windows[:, t], state, force_visual=force)
            items = [('pose', pose), ('a', info['a']), ('elapsed', info['elapsed'])]
            if info.get('pose0') is not None:
                items.append(('pose0', info['pose0']))
            items += [('pcm_' + key, v) for key, v in info.get('pcm', {}).items()]
            for key, v in items:
                rec.setdefault(key, []).append(v[0].cpu().numpy())
            k += 1
    return {key: np.stack(v) for key, v in rec.items()}


def errors(out, part):
    n = len(out['pose'])
    gt = np.asarray(part.poses_rel)[:n, :6]
    t_rmse = float(np.sqrt(((out['pose'][:, 3:] - gt[:, 3:]) ** 2).sum(1).mean()))
    if n < len(part.poses_rel):           # --max_chunks debug run
        return {'t_rmse': t_rmse}
    est, gtm, t_rel, r_rel, _, r_rmse, usage, speed = kitti_eval(out['pose'], out['a'][1:, 0], part.poses_rel)
    return dict(t_rel=float(t_rel), r_rel=float(r_rel), t_rmse=t_rmse, usage=float(usage))


def yaw_slope(pose, gt):
    x, y = gt[:, 1], pose[:, 1]          # camera frame: index 1 = rotation about y (yaw)
    k, b = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)[0]
    return float(k), float(np.mean(y - x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_dir', default=None, help='override the training machine data path')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--test_seq', nargs='+', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--max_chunks', type=int, default=0, help='debug: only the first chunks of each sequence')
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    opt = load_opt(ck['opt'])
    if args.data_dir is not None:
        opt.data_dir = args.data_dir
    if 'VIO_SEG_LENGTHS' not in os.environ and abs(opt.dt - 0.1) > 1e-6:
        os.environ['VIO_SEG_LENGTHS'] = '5,10,15,20'      # ground sequences are 25-208 m: no 100 m segments
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    model = build_model(opt)
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    pargs = SimpleNamespace(data_dir=str(opt.data_dir).rstrip('/'), seq_len=opt.seq_len, img_h=opt.img_h, img_w=opt.img_w)
    tests = args.test_seq or opt.test_seq
    mc = args.max_chunks
    res = {'ckpt': args.ckpt, 'pcm': True, 'seq': {}}      # 'pcm' kept: same JSON layout as before 2026-11-11
    for seq in tests:
        part = data_partition(pargs, seq)
        gt = np.asarray(part.poses_rel)[:, :6]
        n_int = len(gt) if not mc else mc * (opt.seq_len - 1)
        ones, zeros = np.ones(n_int), np.zeros(n_int)
        rng = np.random.default_rng(1)
        sched25 = (rng.random(n_int) < 0.25).astype(float)
        r = {}
        full = run(model, part, device, ones, max_chunks=mc)
        r['full'] = errors(full, part)
        r['no_vision'] = errors(run(model, part, device, zeros, max_chunks=mc), part)
        r['t_rmse_ratio_no_vision'] = r['no_vision']['t_rmse'] / max(r['full']['t_rmse'], 1e-12)
        n = len(full['pose'])
        speed = np.linalg.norm(gt[:n, 3:], axis=1) / opt.dt
        still = speed < STILL_SPEED
        est_t = full['pose'][:, 3:]
        r['stop'] = dict(frames=int(still.sum()), net_m=float(np.linalg.norm(est_t[still].sum(0))),
                         path_m=float(np.linalg.norm(est_t[still], axis=1).sum()))
        if 'pcm_p_still' in full:
            ps = full['pcm_p_still'][:n, 0]
            r['stop']['p_still'] = float(ps[still].mean()) if still.any() else float('nan')
            r['stop']['p_still_moving'] = float(ps[~still].mean()) if (~still).any() else float('nan')
        if 'pose0' in full:
            r['vision_effect_normal'] = float(np.linalg.norm(full['pose'][1:, 3:] - full['pose0'][1:, 3:], axis=1).mean())
        g = full['pcm_k'] * full['pcm_pi']
        r['gain_normal'] = [float(x) for x in g[1:].mean(0)]
        r['pi_normal'] = [float(x) for x in full['pcm_pi'][1:].mean(0)]
        lv_normal = full['pcm_lv_v'][1:]
        for kind in BLANK_KINDS:
            o = run(model, part, device, ones, blank=kind, max_chunks=mc)
            b = {'errors': errors(o, part)}
            if 'pose0' in o:
                b['vision_effect'] = float(np.linalg.norm(o['pose'][1:, 3:] - o['pose0'][1:, 3:], axis=1).mean())
            gb = (o['pcm_k'] * o['pcm_pi'])[1:]
            b['gain'] = [float(x) for x in gb.mean(0)]
            b['gain_trans_over_normal'] = float(gb[:, 1].mean() / max(g[1:, 1].mean(), 1e-12))
            b['pi'] = [float(x) for x in o['pcm_pi'][1:].mean(0)]
            d = np.nanmean(o['pcm_lv_v'][1:, 1]) - np.nanmean(lv_normal[:, 1])
            b['sigma_v_ratio'] = float(np.exp(0.5 * d))           # std; the criterion "sigma_v >= 3" uses this
            b['var_v_ratio_arith'] = float(np.nanmean(np.exp(o['pcm_lv_v'][1:, 1])) / np.nanmean(np.exp(lv_normal[:, 1])))
            b['nis_trans'] = float(np.nanmean(o['pcm_nis'][1:, 1]))
            r[f'blank_{kind}'] = b
        # every image rejected; the gate is switched on for this run if the model trains without one
        net = model.PCM_net
        saved = (net.gate, net.gate_a.detach().clone())
        with torch.no_grad():
            net.gate = True
            net.gate_a.fill_(-1e4)
        rej = run(model, part, device, ones, max_chunks=mc)
        with torch.no_grad():
            net.gate = saved[0]
            net.gate_a.copy_(saved[1])
        r['all_rejected'] = errors(rej, part)
        r['t_rmse_ratio_all_rejected'] = r['all_rejected']['t_rmse'] / max(r['full']['t_rmse'], 1e-12)
        for name, swap in (('imu', False), ('imu_swapped', True)):
            o = run(model, part, device, sched25, imu_swap=swap, max_chunks=mc, seed=1)
            m = len(o['pose'])
            calls = o['elapsed'][:m, 0] / opt.dt     # intervals since the last call, this one included
            sel = calls >= 2 - 1e-6
            pred = o['pcm_trans_p'][:m] * opt.b_d
            if sel.sum() > 10:
                r[f'prior_r2_fwd_{name}'] = r2(pred[sel, 2], gt[:m][sel, 5])
                r[f'prior_r2_3d_{name}'] = r2(pred[sel].ravel(), gt[:m][sel, 3:].ravel())
        if 'prior_r2_fwd_imu' in r and 'prior_r2_fwd_imu_swapped' in r:
            r['prior_r2_fwd_drop'] = r['prior_r2_fwd_imu'] - r['prior_r2_fwd_imu_swapped']
        st = run(model, part, device, ones, static=True, max_chunks=mc)
        r['static_yaw_slope'], r['static_yaw_bias'] = yaw_slope(st['pose'], gt[:len(st['pose'])])
        res['seq'][seq] = r
        print(seq, json.dumps(r), flush=True)
    for seq in opt.val_seq:
        part = data_partition(pargs, seq)
        gt = np.asarray(part.poses_rel)[:, :6]
        n_int = len(gt) if not mc else mc * (opt.seq_len - 1)
        sched = (np.random.default_rng(2).random(n_int) < 0.25).astype(float)
        o = run(model, part, device, sched, max_chunks=mc, seed=2)
        m = len(o['pose'])
        err = ((o['pcm_trans_p'][:m] - gt[:m, 3:] / opt.b_d) ** 2).sum(1)
        lvp, calls = o['pcm_lv_p'][:m, 1], o['elapsed'][:m, 0] / opt.dt
        ok = np.isfinite(err) & (err > 0)
        ok[0] = False
        res[f'val_{seq}'] = dict(spearman_varp_err=spearman(lvp[ok], np.log(err[ok])),
                                 partial_spearman_given_calls=partial_spearman(lvp[ok], np.log(err[ok]), calls[ok]),
                                 n=int(ok.sum()))
        print(seq, json.dumps(res[f'val_{seq}']), flush=True)
    normal = [s for s in tests if 'occlusion' not in s]
    flat = {}
    for s in normal:
        r = res['seq'][s]
        vals = {'t_rmse_ratio_no_vision': r['t_rmse_ratio_no_vision'], 'static_yaw_slope': r['static_yaw_slope'],
                'stop_net_m': r['stop']['net_m'], 'stop_p_still': r['stop'].get('p_still', float('nan'))}
        vals.update(t_rmse_ratio_all_rejected=r['t_rmse_ratio_all_rejected'],
                    prior_r2_fwd_drop=r.get('prior_r2_fwd_drop', float('nan')))
        for kind in BLANK_KINDS:
            vals[f'gain_ratio_{kind}'] = r[f'blank_{kind}']['gain_trans_over_normal']
            vals[f'sigma_v_ratio_{kind}'] = r[f'blank_{kind}']['sigma_v_ratio']
            vals[f'pi_trans_{kind}'] = r[f'blank_{kind}']['pi'][1]
        for key, v in vals.items():
            flat.setdefault(key, []).append(v)
    res['mean_normal_seqs'] = {key: float(np.nanmean(v)) for key, v in flat.items()}
    print('mean', json.dumps(res['mean_normal_seqs']))
    if args.out:
        json.dump(res, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
