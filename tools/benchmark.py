"""Frozen PCM v3 evaluation suite. Run from any directory; results live in the repository.

python tools/benchmark.py --dataset ground --ckpt /path/stage1_last.pth --data-dir /path/data --name reproduction
Use --dry-run to inspect commands without loading a checkpoint or running inference.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'benchmarks/protocol_v1.json'
METRICS = ('t_rel', 'r_rel', 't_rmse', 'r_rmse', 'usage')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def aggregate(runs, sequences):
    """Equal weight per sequence, then mean/population std across decision seeds.

    Never average per-sequence standard deviations: first aggregate inside each run.
    """
    if not runs or not sequences:
        raise ValueError('empty runs or sequences')
    result = {}
    for metric in METRICS:
        values = [statistics.mean(float(r[s][metric]) for s in sequences) for r in runs]
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f'non-finite {metric}; check sequence lengths and evaluation')
        result[metric] = {'mean': statistics.mean(values), 'std': statistics.pstdev(values)}
    return result


def summarize(payload, spec):
    runs = payload['runs']
    return {
        'num_runs': len(runs),
        'primary': aggregate(runs, spec['primary']),
        'all_sequences': aggregate(runs, spec['test']),
        'per_sequence': {s: aggregate(runs, [s]) for s in spec['test']},
        'threshold': payload.get('threshold'),
    }


def validate_checkpoint(ckpt, spec, policy, protocol):
    opt = ckpt['opt']
    if opt.get('pcm_version') != 3:
        raise ValueError('This suite accepts PCM v3 checkpoints only')
    for key, expected in [('train_seq', spec['train']), ('val_seq', spec['val']), ('test_seq', spec['test'])]:
        if list(opt[key]) != expected:
            raise ValueError(f'{key} differs from frozen protocol: {opt[key]} != {expected}')
    for key, expected in [('img_h', protocol['image_hw'][0]), ('img_w', protocol['image_hw'][1]),
                          ('seq_len', protocol['seq_len']), ('max_skip', 0)]:
        if opt[key] != expected:
            raise ValueError(f'{key}: {opt[key]} != {expected}')
    if not math.isclose(opt['dt'], spec['dt'], abs_tol=1e-8):
        raise ValueError('checkpoint frame interval differs from dataset protocol')
    stage = ckpt['stage']
    if stage not in (1, 2) or ckpt['epoch'] != spec[f'stage{stage}_epochs'] - 1:
        raise ValueError('benchmark requires the final epoch of the declared stage')
    if policy and (stage != 2 or not opt.get('use_policy')):
        raise ValueError('--include-policy requires a trained stage-2 policy')
    if set(opt['train_seq']) & set(spec['val'] + spec['test']):
        raise ValueError('training/calibration/test overlap')


def commands(args, protocol):
    spec = protocol['datasets'][args.dataset]
    cases = protocol['estimator_cases'] + (protocol['policy_cases'] if args.include_policy else [])
    result = []
    for case in cases:
        name = f'{args.name}_{case["name"]}'
        cmd = [sys.executable, str(ROOT / 'test_crvio.py'), '--ckpt', str(Path(args.ckpt).resolve()),
               '--data_dir', str(Path(args.data_dir).resolve()), '--test_seq', *spec['test'],
               '--val_seq', *spec['val'], '--gpu', str(args.gpu), '--experiment_name', name,
               '--seed', str(protocol['evaluation_seed']), '--num_runs', str(case['runs']),
               '--bn_recalib', '0', '--max_skip', '0', *case['args']]
        result.append((case['name'], name, cmd))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['ground', 'kitti'], required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--name', required=True, help='new output prefix; existing outputs are refused')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--include-policy', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.name):
        p.error('--name must be a simple directory name')
    protocol = json.loads(PROTOCOL.read_text())
    spec = protocol['datasets'][args.dataset]
    jobs = commands(args, protocol)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(VIO_SEG_LENGTHS=','.join(map(str, spec['segment_lengths_m'])), MPLBACKEND='Agg',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    print('VIO_SEG_LENGTHS=' + env['VIO_SEG_LENGTHS'])
    for _, _, cmd in jobs:
        print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    sys.path.insert(0, str(ROOT))
    from tools.verify_data import verify
    data_manifest = ROOT / 'benchmarks/data' / f'{args.dataset}.json'
    verify(args.data_dir, data_manifest)
    import torch
    ckpt_path = Path(args.ckpt).resolve()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    validate_checkpoint(ckpt, spec, args.include_policy, protocol)
    suite = ROOT / 'results' / args.name
    for path in [suite] + [ROOT / 'results' / n for _, n, _ in jobs]:
        if path.exists():
            raise FileExistsError(f'Output exists: {path}; choose a new --name')
    for s in spec['val'] + spec['test']:
        for relative in [f'poses/{s}.txt', f'imus/{s}.mat', f'sequences/{s}/image_2']:
            if not (Path(args.data_dir) / relative).exists():
                raise FileNotFoundError(relative)
    suite.mkdir(parents=True)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True)
    manifest = {
        'protocol_id': protocol['protocol_id'], 'protocol_sha256': sha256(PROTOCOL), 'dataset': args.dataset,
        'checkpoint': str(ckpt_path), 'checkpoint_sha256': sha256(ckpt_path),
        'checkpoint_stage': ckpt['stage'], 'checkpoint_epoch_zero_based': ckpt['epoch'],
        'checkpoint_opt': ckpt['opt'], 'code_revision': revision.stdout.strip(),
        'code_dirty': bool(status.stdout.strip()), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else 'cpu',
        'segment_lengths_m': spec['segment_lengths_m'], 'commands': [cmd for _, _, cmd in jobs],
        'data_manifest_sha256': sha256(data_manifest),
        'status': 'running',
    }
    (suite / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    del ckpt
    summaries = {}
    try:
        for case, name, cmd in jobs:
            print('Running', case, flush=True)
            with (suite / f'{case}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            payload = json.loads((ROOT / 'results' / name / 'files/metrics.json').read_text())
            summaries[case] = summarize(payload, spec)
            (suite / 'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
        manifest['status'] = 'complete'
    except Exception:
        manifest['status'] = 'failed'
        raise
    finally:
        (suite / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Complete:', suite / 'summary.json')


if __name__ == '__main__':
    main()
