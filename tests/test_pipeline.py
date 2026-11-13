"""Training infrastructure: image cache, resume, end-to-end PCM v3 runs, collapse diagnostic. Synthetic data only."""
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import pytest
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from crvio.losses import effective_rank  # noqa: E402
from dataset.KITTI_dataset import KITTI  # noqa: E402
from tools import make_synthetic_kitti  # noqa: E402
from tools.cache_images import cache_sequence  # noqa: E402
from utils import custom_transform  # noqa: E402

SIZE = (64, 128)


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('kitti'))
    make_synthetic_kitti.main(root)
    cache = os.path.join(root, 'cache')
    os.makedirs(cache)
    with Pool(2) as pool:
        for seq in make_synthetic_kitti.LENGTHS:
            cache_sequence(root, cache, seq, SIZE, pool, chunk=7)
    return root, cache


def test_cached_frames_match_the_training_transform(synthetic):
    root, cache = synthetic
    live = KITTI(root, train_seqs=['00', '01'], transform=custom_transform.Compose(
        [custom_transform.ToTensor(), custom_transform.Resize(SIZE)]))
    cached = KITTI(root, train_seqs=['00', '01'], img_cache=cache,
                   transform=custom_transform.Compose([custom_transform.ToTensor()]))
    assert cached.cache_size == list(SIZE) and len(live) == len(cached)
    worst = 0.0
    for idx in (0, 7, len(live) - 1):
        a, b = live[idx], cached[idx]
        assert a[0].shape == b[0].shape == (11, 3) + SIZE
        worst = max(worst, (a[0] - b[0]).abs().max().item())
        for x, y in zip(a[1:], b[1:]):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert worst <= 0.5 / 255 + 1e-6


def test_cache_is_reused_and_rejects_a_different_frame_list(synthetic, tmp_path):
    root, cache = synthetic
    with Pool(1) as pool:   # second call finds an up-to-date cache and returns without writing
        mtime = os.path.getmtime(os.path.join(cache, '00.npy'))
        cache_sequence(root, cache, '00', SIZE, pool)
        assert os.path.getmtime(os.path.join(cache, '00.npy')) == mtime
    bad = tmp_path / 'cache'
    bad.mkdir()
    meta = json.load(open(os.path.join(cache, '00.json')))
    meta['files'] = meta['files'][:-1]
    json.dump(meta, open(bad / '00.json', 'w'))
    with pytest.raises(RuntimeError):
        KITTI(root, train_seqs=['00'], img_cache=str(bad))


def train(root, cache, cwd, name, *extra):
    cmd = [sys.executable, os.path.join(ROOT, 'train_crvio.py'), '--preset', 'PCM3', '--data_dir', root,
           '--img_cache', cache, '--img_h', str(SIZE[0]), '--img_w', str(SIZE[1]), '--gpu', '-1',
           '--workers', '0', '--batch_size', '4', '--limit_batches', '2', '--train_seq', '00', '01',
           '--val_seq', '09', '--eval_every', '1', '--print_frequency', '1', '--experiment_name', name, *extra]
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return os.path.join(cwd, 'results', name)


def test_resume_across_the_stage_boundary_matches_an_uninterrupted_run(synthetic, tmp_path):
    """Interrupted after the first of two stage-1 epochs; the resumed run must reach the same
    weights (mid-cosine lr, data order and RNG restored), which also checks that stage 2 starts
    from stage1_last in both runs. The logged lr must follow warmup + cosine."""
    from crvio.config import cosine_lr
    root, cache = synthetic
    args = ('--epochs_stage1', '2', '--epochs_stage2', '1')    # PCM recipe: speed-balanced sampler, AdamW, EMA
    full = train(root, cache, str(tmp_path), 'full', *args)
    part = train(root, cache, str(tmp_path), 'part', *args, '--stop_after_epoch', '0')
    assert not os.path.exists(os.path.join(part, 'checkpoints', 'stage2_last.pth'))
    train(root, cache, str(tmp_path), 'part', *args, '--resume', os.path.join(part, 'checkpoints', 'stage1_last.pth'))
    a = torch.load(os.path.join(full, 'checkpoints', 'stage2_last.pth'), weights_only=False)
    b = torch.load(os.path.join(part, 'checkpoints', 'stage2_last.pth'), weights_only=False)
    assert a['epoch'] == b['epoch'] == 0
    for key in ('model', 'model_raw'):
        for k in a[key]:
            assert torch.equal(a[key][k], b[key][k]), (key, k)
    log = open(os.path.join(full, 'train.log')).read()
    assert 'stage 2 starts from stage1_last.pth' in log
    # limit_batches 2: 2 iterations per epoch, 1 warmup epoch, the logged lr is the last step's
    assert f'stage 1 epoch 0 temp=1.000 lr_est={cosine_lr(1, 4, 2, 2.5e-4, 0.01):.2e}' in log
    assert f'stage 1 epoch 1 temp=1.000 lr_est={cosine_lr(3, 4, 2, 2.5e-4, 0.01):.2e}' in log
    lp, le = cosine_lr(1, 2, 2, 1e-4, 0.01), cosine_lr(1, 2, 2, 1e-5, 0.01)
    assert f'lr_policy={lp:.2e} lr_policy_nd={lp:.2e} lr_est={le:.2e} lr_est_nd={le:.2e}' in log


def test_effective_rank_detects_collapse():
    torch.manual_seed(0)
    assert effective_rank(torch.ones(160, 64)) == pytest.approx(1.0)
    assert effective_rank(torch.randn(160, 1) * torch.randn(1, 64)) == pytest.approx(1.0, abs=1e-3)
    assert effective_rank(torch.randn(4000, 64)) > 60


def train_b0(root, cache, cwd, name, *extra):
    cmd = [sys.executable, os.path.join(ROOT, 'train.py'), '--data_dir', root, '--img_cache', cache,
           '--img_h', str(SIZE[0]), '--img_w', str(SIZE[1]), '--gpu_ids', '0', '--workers', '0', '--batch_size', '4',
           '--limit_batches', '2', '--train_seq', '00', '01', '--val_seq', '09', '--eval_every', '1',
           '--print_frequency', '1', '--experiment_name', name,
           '--pretrain_flownet', os.path.join(ROOT, 'pretrain_models', 'flownets_bn_EPE2.459.pth.tar'), *extra]
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['VIO_DETERMINISTIC'] = '1'
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return os.path.join(cwd, 'results', name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='train.py is CUDA only')
def test_b0_two_stage_protocol_resumes_like_an_uninterrupted_run(synthetic, tmp_path):
    """train.py under the unified protocol: stage 1 random vision at the cosine lr, stage 2 policy
    with two param groups; interrupting after stage-1 epoch 0 and resuming reaches bitwise the same
    weights (deterministic cuDNN in this test) and the same epoch / lr bookkeeping."""
    from crvio.config import cosine_lr
    root, cache = synthetic
    # cuDNN keeps the LSTM's inter-layer dropout state outside torch's RNG state, so a resumed GPU run
    # can only match bitwise without that dropout; the dropout after the LSTM is still active.
    args = ('--epochs_stage1', '2', '--epochs_stage2', '1', '--rnn_dropout_between', '0')
    full = train_b0(root, cache, str(tmp_path), 'full', *args)
    part = train_b0(root, cache, str(tmp_path), 'part', *args, '--stop_after_epoch', '0')
    assert not os.path.exists(os.path.join(part, 'checkpoints', 'stage2_last.pth'))
    train_b0(root, cache, str(tmp_path), 'part', *args, '--resume', os.path.join(part, 'checkpoints', 'stage1_last.pth'))
    a = torch.load(os.path.join(full, 'checkpoints', 'stage2_last.pth'), weights_only=False)
    b = torch.load(os.path.join(part, 'checkpoints', 'stage2_last.pth'), weights_only=False)
    assert a['stage'] == b['stage'] == 2 and a['epoch'] == b['epoch'] == 0
    for k in a['model']:
        assert torch.equal(a['model'][k], b['model'][k]), k
    log = open(os.path.join(full, 'logs', 'train_full.txt')).read()
    assert f'stage 1 epoch 1 selection=random temp=5.000 lr_est={cosine_lr(3, 4, 2, 2.5e-4, 0.01):.2e}' in log
    assert f'lr_policy={cosine_lr(1, 2, 2, 1e-4, 0.01):.2e} lr_est={cosine_lr(1, 2, 2, 1e-5, 0.01):.2e}' in log
    assert 'stage 1 epoch 1 VAL' in log and 'stage 2 epoch 0 VAL' in log


def test_pcm3_trains_both_stages_and_tests_with_a_random_policy(synthetic, tmp_path):
    """PCM3 end to end: stage 1 with blank frames (the default visual_types; fusion loss active), stage 2 with the
    policy, then test_crvio.py."""
    root, cache = synthetic
    run = train(root, cache, str(tmp_path), 'pcm', '--epochs_stage1', '1', '--epochs_stage2', '1')
    log = open(os.path.join(run, 'train.log')).read()
    assert 'stage 2 finished' in log and 'xpred=' in log and 'k_trans=' in log
    assert 'fuse=' in log and 'alpha_trans=' in log and 'pstill_moving=' in log and 'pi_loss' not in log
    assert torch.load(os.path.join(run, 'checkpoints', 'stage1_last.pth'), weights_only=False)['opt']['pcm_version'] == 3
    ckpt = os.path.join(run, 'checkpoints', 'stage2_last.pth')
    env = {k: v for k, v in os.environ.items() if k != 'PYTHONPATH'}
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'test_crvio.py'), '--ckpt', ckpt, '--data_dir', root,
                        '--test_seq', '09', '--gpu', '-1', '--policy', 'random', '--policy_value', '0.25',
                        '--experiment_name', 'pcm_test'], cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
