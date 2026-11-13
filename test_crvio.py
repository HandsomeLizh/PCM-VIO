"""Test protocol. Every threshold and rule is fixed from validation data before test runs.

  python test_crvio.py --ckpt results/<run>/checkpoints/stage2_last.pth --data_dir ./data
PCM v3 checkpoints only; PCM / PCMC (v1 / v2) checkpoints: /root/VIO/snapshot_pcm12_20261111 (see crvio.vio.build_model).
  python test_crvio.py --ckpt ... --decision threshold --match_usage 20
  python test_crvio.py --ckpt ... --policy interval --policy_value 5
  python test_crvio.py --ckpt ... --test_perturb mixed --perturb_scale 2
"""
import argparse
import json
import math

import numpy as np
import torch
from path import Path

from crvio.config import load_opt
from crvio.evaluation import Evaluator, calibrate_threshold, make_policy_fn
from crvio.recipe import recalibrate_batchnorm
from crvio.vio import build_model
from crvio.perturb import ALL_VISUAL_TYPES, IMU_TYPES, VISUAL_TYPES, PerturbConfig


def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_dir', default=None, help='default: the training value')
    p.add_argument('--test_seq', nargs='+', default=None)
    p.add_argument('--val_seq', nargs='+', default=None, help='calibration sequences')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--experiment_name', default=None)
    p.add_argument('--policy', default='learned', choices=['learned', 'all', 'random', 'interval'])
    p.add_argument('--policy_value', type=float, default=None,
                   help='random: p; interval: period')
    p.add_argument('--decision', default='sample', choices=['sample', 'threshold'], help='learned policy rule')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--match_usage', type=float, default=None,
                   help='calibrate on val_seq so visual usage reaches this percentage')
    p.add_argument('--num_runs', type=int, default=1, help='repeat sampled decisions and report spread')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bn_recalib', type=int, default=0,
                   help='diagnostic: recompute BatchNorm statistics on this many clean training batches before '
                        'testing (the temperature assumes the last stage-2 epoch); 0 = off')
    p.add_argument('--max_skip', type=int, default=None)
    # Degradation evaluation
    p.add_argument('--test_perturb', default='none', choices=['none', 'visual', 'imu', 'mixed'])
    p.add_argument('--perturb_types', nargs='+', default=None, help='restrict to these perturbation kinds')
    p.add_argument('--perturb_scale', type=float, default=1.0, help='>1 for unseen strengths')
    p.add_argument('--p_severe', type=float, default=0.3)
    p.add_argument('--misalign_samples', type=int, default=0, help='sync diagnostic, in IMU samples')
    return p.parse_args()


def build_perturb(args):
    if args.misalign_samples:
        return PerturbConfig(misalign_samples=args.misalign_samples)
    if args.test_perturb == 'none':
        return None
    kinds = args.perturb_types
    return PerturbConfig(mode=args.test_perturb, scale=args.perturb_scale, p_severe=args.p_severe, full_window=True,
                         visual_types=tuple(k for k in (kinds or VISUAL_TYPES) if k in ALL_VISUAL_TYPES) or VISUAL_TYPES,
                         imu_types=tuple(k for k in (kinds or IMU_TYPES) if k in IMU_TYPES) or IMU_TYPES)


def main():
    args = parse()
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    opt = load_opt(ckpt['opt'])
    opt.data_dir = args.data_dir or opt.data_dir
    if args.max_skip is not None:
        opt.max_skip = args.max_skip
    test_seq = args.test_seq or opt.test_seq
    val_seq = args.val_seq or opt.val_seq
    if set(test_seq) & set(val_seq) and args.match_usage is not None:
        raise ValueError('calibration and test sequences overlap')
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    model = build_model(opt)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    if args.bn_recalib:
        from train_crvio import clean_batch_runner, make_train_set
        stage = ckpt.get('stage', 1)
        temp = opt.temp_init * math.exp(-opt.eta * (opt.epochs_stage2 - 1)) if stage == 2 else 1.0
        torch.manual_seed(args.seed)
        loader = torch.utils.data.DataLoader(make_train_set(opt), batch_size=opt.batch_size, shuffle=True,
                                             drop_last=True, num_workers=opt.workers,
                                             generator=torch.Generator().manual_seed(args.seed))
        n = recalibrate_batchnorm(model, loader, clean_batch_runner(model, stage, temp, device, opt), args.bn_recalib)
        print(f'BatchNorm statistics recomputed on {n} clean training batches (stage {stage} forward)')
        model.eval()

    def rule_for(threshold=None):
        """(rule, policy_fn) for the selected policy; `threshold` overrides during calibration."""
        if args.policy == 'learned':
            if args.decision == 'threshold':
                return {'kind': 'threshold', 'value': args.threshold if threshold is None else threshold}, None
            return {'kind': 'sample'}, None
        value = args.policy_value if threshold is None else threshold
        return None, make_policy_fn(args.policy, value, dt=opt.dt)

    threshold = None
    if args.match_usage is not None:
        if args.policy == 'learned' and args.decision == 'threshold':
            bounds = (0.0, 1.0)
        else:
            raise ValueError('--match_usage needs --decision threshold')
        threshold, usage = calibrate_threshold(Evaluator(opt, val_seq), model, device, args.match_usage, rule_for, *bounds)
        print(f'calibrated on {val_seq}: threshold={threshold:.5f}, validation usage={usage:.2f}%')

    perturb = build_perturb(args)
    evaluator = Evaluator(opt, test_seq)
    name = args.experiment_name or (Path(args.ckpt).parent.parent.name + '_test')
    out_dir = Path('./results') / name / 'files'
    out_dir.makedirs_p()

    runs = []
    for run in range(args.num_runs):
        torch.manual_seed(args.seed + run)
        rule, policy_fn = rule_for(threshold)
        runs.append(evaluator.evaluate(model, device, rule=rule, policy_fn=policy_fn,
                                       perturb=perturb, perturb_seed=args.seed + run * 100000))
    evaluator.save(out_dir)

    for seq in test_seq + ['mean']:
        msg = f'{seq}: '
        for key in ('t_rel', 'r_rel', 't_rmse', 'r_rmse', 'usage'):
            vals = [r[seq][key] for r in runs]
            msg += f'{key}={np.mean(vals):.4f}' + (f'±{np.std(vals):.4f} ' if args.num_runs > 1 else ' ')
        print(msg)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'args': vars(args), 'threshold': threshold, 'runs': runs}, f, indent=1, default=float)


if __name__ == '__main__':
    main()
