"""Protect the handoff benchmark against split leakage and misleading aggregation."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.benchmark import aggregate, commands, validate_checkpoint

PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / 'benchmarks/protocol_v1.json').read_text())


def checkpoint():
    spec = PROTOCOL['datasets']['ground']
    return {'stage': 1, 'epoch': 39, 'opt': {
        'pcm_version': 3, 'train_seq': spec['train'], 'val_seq': spec['val'], 'test_seq': spec['test'],
        'dt': spec['dt'], 'img_h': 256, 'img_w': 512, 'seq_len': 11, 'max_skip': 0,
    }}


def test_summary_averages_sequences_inside_each_run_before_std():
    metrics = lambda x: dict.fromkeys(['t_rel', 'r_rel', 't_rmse', 'r_rmse', 'usage'], x)
    result = aggregate([{'a': metrics(0), 'b': metrics(10)}, {'a': metrics(10), 'b': metrics(0)}], ['a', 'b'])
    assert result['t_rel'] == {'mean': 5, 'std': 0}
    with pytest.raises(ValueError):
        aggregate([{'a': metrics(float('nan'))}], ['a'])


def test_protocol_rejects_training_on_validation_and_untrained_policy():
    spec = PROTOCOL['datasets']['ground']
    ck = checkpoint()
    validate_checkpoint(ck, spec, False, PROTOCOL)
    with pytest.raises(ValueError, match='stage-2'):
        validate_checkpoint(ck, spec, True, PROTOCOL)
    bad = copy.deepcopy(ck)
    bad['opt']['train_seq'].append('m2_tree')
    with pytest.raises(ValueError, match='train_seq'):
        validate_checkpoint(bad, spec, False, PROTOCOL)
    bad = copy.deepcopy(ck)
    bad['epoch'] = 19
    with pytest.raises(ValueError, match='final epoch'):
        validate_checkpoint(bad, spec, False, PROTOCOL)


def test_suite_fixes_seeds_rates_and_disables_recalibration():
    args = SimpleNamespace(dataset='ground', ckpt='a.pth', data_dir='data', name='trial', gpu=-1, include_policy=True)
    jobs = commands(args, PROTOCOL)
    assert len(jobs) == 9
    by_name = {case: cmd for case, _, cmd in jobs}
    for cmd in by_name.values():
        assert cmd[cmd.index('--seed') + 1] == '0'
        assert cmd[cmd.index('--bn_recalib') + 1] == '0'
        assert cmd[cmd.index('--max_skip') + 1] == '0'
    assert by_name['rand25'][by_name['rand25'].index('--num_runs') + 1] == '3'
    assert by_name['learned'][by_name['learned'].index('--num_runs') + 1] == '5'


def test_data_identity_detects_changed_labels_and_frame_inventory(tmp_path):
    from tools.verify_data import fingerprint, verify
    images = tmp_path / 'sequences/00/image_2'
    images.mkdir(parents=True)
    (images / '000000.png').write_bytes(b'image bytes')
    (tmp_path / 'poses').mkdir()
    (tmp_path / 'imus').mkdir()
    pose = tmp_path / 'poses/00.txt'
    pose.write_text('original labels')
    (tmp_path / 'imus/00.mat').write_bytes(b'imu bytes')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'sequences': {'00': fingerprint(tmp_path, '00')}}))
    verify(tmp_path, manifest)
    pose.write_text('changed labels')
    with pytest.raises(ValueError, match='poses_sha256'):
        verify(tmp_path, manifest)
    pose.write_text('original labels')
    (images / '000001.png').write_bytes(b'extra image')
    with pytest.raises(ValueError, match='frames'):
        verify(tmp_path, manifest)
