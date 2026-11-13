"""PCM v3 invariants, recipe pieces and the checkpoint contract, checked with small inputs on the CPU or GPU."""
import math
import json
import os
import sys
from argparse import Namespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crvio import losses as L  # noqa: E402
from crvio import pcm_losses as PL  # noqa: E402
from crvio import recipe as RC  # noqa: E402
from crvio.config import PRESETS, apply_preset, build_parser, load_opt  # noqa: E402
from crvio.evaluation import make_policy_fn  # noqa: E402
from crvio.geometry import euler_to_matrix, rotation_error, so3_exp, so3_log  # noqa: E402
from crvio.perturb import PerturbConfig, apply_perturbation  # noqa: E402
from crvio.pcm import PredictiveCommonMotion  # noqa: E402
from crvio.vio import LEGACY_SNAPSHOT, PCM12_SNAPSHOT, PCMVIO, VIOState, build_model  # noqa: E402
from utils.utils import eulerAnglesToRotationMatrix, euler_from_matrix  # noqa: E402

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
B, T, IMG_H, IMG_W = 3, 6, 64, 128
# The preset PCM3, and the same without the gated complement (pcm_complement false): both code paths of fuse.
BOTH = ['PCM3', 'PCM3_nocomp']


def make_opt(preset='PCM3', *extra):
    base = ['--preset', 'PCM3'] + (['--pcm_complement', 'false'] if preset == 'PCM3_nocomp' else [])
    return apply_preset(build_parser().parse_args(
        base + ['--img_w', str(IMG_W), '--img_h', str(IMG_H), '--b_R', '0.01', '--b_d', '0.5', *extra]))


def make_model(preset='PCM3', *extra):
    torch.manual_seed(0)
    return build_model(make_opt(preset, *extra)).to(DEVICE)


def make_batch(b=B, t=T):
    torch.manual_seed(1)
    imgs = 0.2 * torch.randn(b, t + 1, 3, IMG_H, IMG_W, device=DEVICE)
    imus = torch.randn(b, 10 * t + 1, 6, device=DEVICE)
    mask = torch.ones(b, 10 * t + 1, 1, device=DEVICE)
    gt = 0.01 * torch.randn(b, t, 6, device=DEVICE)
    q = torch.ones(b, t, device=DEVICE)
    return imgs, imus, mask, gt, q


@pytest.fixture
def exact(monkeypatch):
    """float64 on CPU: agreement between code paths is checked at 1e-10, so a real logic
    difference cannot hide behind float32 rounding, which untrained deep nets amplify."""
    monkeypatch.setattr(sys.modules[__name__], 'DEVICE', torch.device('cpu'))
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


def grad_norm(params):
    return sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)


def stage2_terms(model, temp=1.0):
    imgs, imus, mask, gt, q = make_batch()
    out = model.forward_train(imgs, imus, mask, stage=2, temp=temp)
    return out, L.stage2_loss(out, gt, q, model.opt)


# ------------------------------------------------------------------ geometry
def test_euler_matches_baseline_and_label_convention():
    for _ in range(20):
        e = np.random.uniform(-0.4, 0.4, 3)
        ours = euler_to_matrix(torch.as_tensor(e)).numpy()
        assert np.allclose(ours, eulerAnglesToRotationMatrix(e), atol=1e-12)
        M = np.eye(4)
        M[:3, :3] = ours
        assert np.allclose(euler_from_matrix(M), e, atol=1e-9)


def test_so3_log_exp_against_scipy():
    phi = np.random.uniform(-1.0, 1.0, (64, 3))
    R = Rotation.from_rotvec(phi).as_matrix()
    assert np.allclose(so3_exp(torch.as_tensor(phi)).numpy(), R, atol=1e-10)
    assert np.allclose(so3_log(torch.as_tensor(R)).numpy(), phi, atol=1e-8)


def test_so3_gradients_finite_at_identity():
    euler = torch.zeros(4, 3, requires_grad=True, dtype=torch.float32)
    err = rotation_error(euler_to_matrix(euler), euler_to_matrix(torch.zeros(4, 3)))
    (err ** 2).sum().backward()
    assert torch.isfinite(euler.grad).all()
    phi = torch.zeros(4, 3, requires_grad=True, dtype=torch.float32)
    so3_exp(phi).sum().backward()
    assert torch.isfinite(phi.grad).all()


# ------------------------------------------------------------------ checkpoint contract
CONTRACT = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'pcm3_state_keys.json')))


def test_state_dict_keys_order_and_shapes_match_the_v3_contract():
    """(h) Keys, their order (= parameters() order used by --resume optimizer states and the EMA) and shapes at the
    default sizes, as in tests/data/pcm3_state_keys.json. Rename or reorder nothing in PCMVIO / PCM; new parameters go
    after the existing ones."""
    model = build_model(apply_preset(build_parser().parse_args(['--preset', 'PCM3'])))
    assert [[k, list(v.shape)] for k, v in model.state_dict().items()] == CONTRACT['PCM3']['state_dict']
    assert [n for n, _ in model.named_parameters()] == CONTRACT['PCM3']['parameters']
    names = [n for n, _ in model.PCM_net.named_parameters()]
    assert names[:5] == ['gate_a', 'gate_c', 'gain_alpha', 'gain_beta', 'q_raw']
    assert not any('gate_counts' in k for k in model.state_dict())


def test_v3_checkpoint_options_load_and_older_designs_are_refused():
    v3 = vars(make_opt('PCM3'))
    model = build_model(load_opt(v3))
    assert isinstance(model, PCMVIO) and model.PCM_net.complement and v3['pcm_version'] == 3
    v2 = {k: v for k, v in v3.items() if k != 'pcm_version'}          # PCM / PCMC checkpoints carry no pcm_version
    for old in (dict(v2, preset='PCMC', pcm_phi_var=False, pcm_detach_gain=True, lambda_pi=0.01),
                dict(v3, preset='PCM', pcm_version=2)):
        with pytest.raises(ValueError, match=PCM12_SNAPSHOT):
            build_model(load_opt(old))
    with pytest.raises(ValueError, match=LEGACY_SNAPSHOT):
        build_model(load_opt(dict(v2, preset='BM', fusion='direct')))


def test_single_preset_keeps_the_recipe_and_drops_the_v2_options():
    assert sorted(PRESETS) == ['PCM3'] and build_parser().get_default('preset') == 'PCM3'
    a = vars(apply_preset(build_parser().parse_args([])))
    assert a['preset'] == 'PCM3' and a['pcm_version'] == 3 and a['lambda_fuse'] == 0.05 and a['pcm_complement']
    assert a['pcm_gate'] and a['pcm_tau_max'] == 9.0
    assert a['fusion'] == 'pcm' and a['direct_hidden'] == 1244 and a['gyro_shortcut'] and a['speed_balance']
    assert a['optimizer'] == 'adamw' and a['ema_decay'] > 0 and a['stage2_freeze_visual'] and a['target_usage'] > 0
    assert a['perturb'] == 'mixed' and 'blank' in a['visual_types'] and a['still_aug'] > 0
    assert a['policy_signal'] == 'full'
    for gone in (('--use_prior', 'true'), ('--pcm_phi_var', 'true'), ('--pcm_detach_gain', 'false'),
                 ('--lambda_pi', '0.01'), ('--preset', 'PCMC')):
        with pytest.raises(SystemExit):             # options of the older designs are gone
            make_opt('PCM3', *gone)
    with pytest.raises(ValueError):
        make_opt('PCM3', '--pcm_blocks', '16,32')


def test_ground_arm_command_line_parses():
    """The full training command of scripts/ground_arm.sh with the v3 preset."""
    args = ('--preset PCM3 --stage 1 --data_dir /root/VIO/data_ground --train_seq m2_bridge2 m2_street1 m2_switch '
            'm3_dynamic01 m3_shaturn01 m3_wheelfloat01 --val_seq m2_tree --test_seq m2_building2 m2_street2 '
            'm3_varyingillu01 m3_occlusion01 --img_cache /root/VIO/cache/ground_256x512 --dt 0.0666667 '
            '--epochs_stage1 40 --epochs_stage2 40 --workers 8 --gpu 0 --amp bf16 --batch_size 32 --lr_stage1 3.5e-4 '
            '--visual_types noise photometric occlusion blur blank --experiment_name pcm3_g --seed 1').split()
    opt = apply_preset(build_parser().parse_args(args))
    assert opt.pcm_complement and opt.amp == 'bf16' and opt.batch_size == 32 and opt.seed == 1
    assert opt.pcm_version == 3 and opt.lambda_fuse == 0.05 and opt.dt == pytest.approx(0.0666667)
    assert opt.visual_types == ['noise', 'photometric', 'occlusion', 'blur', 'blank']


# ------------------------------------------------------------------ inference invariants
@pytest.mark.parametrize('preset', BOTH)
def test_no_visual_path_does_not_depend_on_images(preset):
    model = make_model(preset).eval()
    imgs, imus, mask, _, _ = make_batch()
    windows = model.imu_windows(imus, mask, T)
    state = model.init_state(B, DEVICE)
    zero = torch.zeros(B, 1, device=DEVICE)
    a = model.inference_step(imgs[:, 0], imgs[:, 1], windows[:, 0], state, force_visual=zero)
    b = model.inference_step(imgs[:, 2] * 0.3, imgs[:, 4] + 0.2, windows[:, 0], state, force_visual=zero)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1].h, b[1].h) and torch.equal(a[1].tau, b[1].tau)
    assert torch.equal(a[2]['pcm']['lv_p'], b[2]['pcm']['lv_p'])


@pytest.mark.parametrize('preset', BOTH)
def test_visual_encoder_really_skipped(preset):
    model = make_model(preset).eval()
    calls = []
    model.Feature_net.conv1.register_forward_hook(lambda m, i, o: calls.append(i[0].shape[0]))
    imgs, imus, mask, _, _ = make_batch()
    windows = model.imu_windows(imus, mask, T)
    state = model.init_state(B, DEVICE)
    model.inference_step(imgs[:, 0], imgs[:, 1], windows[:, 0], state, force_visual=torch.zeros(B, 1, device=DEVICE))
    assert calls == []
    a = torch.tensor([[1.], [0.], [1.]], device=DEVICE)
    model.inference_step(imgs[:, 0], imgs[:, 1], windows[:, 0], state, force_visual=a)
    assert calls == [2]      # only the two samples that called vision


@pytest.mark.parametrize('preset', BOTH)
def test_mixed_batch_returns_the_right_candidate_and_no_state_mutation(exact, preset):
    model = make_model(preset).eval()
    imgs, imus, mask, _, _ = make_batch()
    windows = model.imu_windows(imus, mask, T)
    state = VIOState(torch.randn(2, B, 1024), torch.randn(2, B, 1024), torch.full((B, 1), 0.2))
    before = [x.clone() for x in state]
    ones, zeros = torch.ones(B, 1), torch.zeros(B, 1)
    mixed = torch.tensor([[1.], [0.], [1.]])
    call = lambda a: model.inference_step(imgs[:, 0], imgs[:, 1], windows[:, 0], state, force_visual=a)
    p1, s1, _ = call(ones)
    p0, s0, _ = call(zeros)
    pm, sm, info = call(mixed)
    assert all(torch.equal(x, y) for x, y in zip(state, before))
    assert not torch.allclose(p1, p0)
    for i, use in enumerate([True, False, True]):
        ref_p, ref_s = (p1, s1) if use else (p0, s0)
        assert torch.allclose(pm[i], ref_p[i], atol=1e-10)
        assert torch.allclose(sm.h[:, i], ref_s.h[:, i], atol=1e-10)
        assert torch.allclose(sm.tau[i], ref_s.tau[i], atol=1e-10)
    assert len(sm) == 3                                        # VIOState is (h, c, tau) again
    assert torch.allclose(info['elapsed'], torch.full((B, 1), 0.3))
    # (e) a call resets tau, a skipped interval adds dt
    assert sm.tau[[0, 2], 0].tolist() == [0.0, 0.0] and sm.tau[1, 0].item() == pytest.approx(0.3)


def train_infer_agreement(model, imgs, imus, mask, out, sched):
    """Inference with the schedule of a training forward must reproduce the selected candidate, tau and the prior."""
    res, _ = model.forward_infer(imgs, imus, mask, schedule=sched)
    sel = torch.where(sched[..., None] > 0.5, out['pose1'], out['pose0'])
    assert torch.allclose(res['pose'], sel, atol=1e-10)
    assert torch.allclose(res['elapsed'], out['tau'], atol=1e-12)
    assert torch.allclose(res['pcm_lv_p'], out['lv_p'], atol=1e-10)
    called = sched[..., None] > 0.5        # diagnostics of T^1 where vision was called, the prior's elsewhere
    assert torch.allclose(res['pcm_k'][called[..., 0]], out['k'][called[..., 0]], atol=1e-10)
    assert torch.allclose(res['pcm_p_still'][called], out['p_still'][called], atol=1e-10)
    return res


@pytest.mark.parametrize('preset', BOTH)
def test_train_and_inference_paths_agree(exact, preset):
    """(g) Stage 1 with a fixed schedule, and stage 2 with its own straight-through decisions replayed at inference."""
    model = make_model(preset).eval()
    imgs, imus, mask, gt, q = make_batch()
    sched = torch.ones(B, T)
    sched[:, 2:5] = 0
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=sched)
    res = train_infer_agreement(model, imgs, imus, mask, out, sched)
    # the prior reads log tau and steps: the masked run must change them
    assert res['elapsed'][0, :, 0].tolist() == pytest.approx([0.1, 0.1, 0.1, 0.2, 0.3, 0.4])
    torch.manual_seed(5)
    out2 = model.forward_train(imgs, imus, mask, stage=2, temp=1.0)
    a = out2['a'][..., 0]
    assert 0 < float(a[:, 1:].sum()) < a[:, 1:].numel()        # both decisions occur in the replay
    train_infer_agreement(model, imgs, imus, mask, out2, a)


@pytest.mark.parametrize('preset', BOTH)
def test_chunked_inference_carries_state_and_tau(exact, preset):
    model = make_model(preset).eval()
    imgs, imus, mask, _, _ = make_batch(b=1, t=T)
    schedule = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.float64)
    whole, _ = model.forward_infer(imgs, imus, mask, schedule=schedule)
    first, st = model.forward_infer(imgs[:, :4], imus[:, :31], mask[:, :31], schedule=schedule[:, :3])
    second, _ = model.forward_infer(imgs[:, 3:], imus[:, 30:], mask[:, 30:], state=st, is_first=False,
                                    schedule=schedule[:, 3:])
    assert torch.allclose(whole['pose'], torch.cat([first['pose'], second['pose']], 1), atol=1e-10)
    assert second['elapsed'][0, 0, 0].item() == pytest.approx(0.3)


def test_full_vision_without_policy_never_builds_the_no_visual_path():
    model = make_model('PCM3', '--use_policy', 'false').eval()
    assert model.Policy_net is None
    calls = []
    orig = model._decode
    model._decode = lambda *a: (calls.append(1), orig(*a))[1]
    imgs, imus, mask, _, _ = make_batch()
    out, _ = model.forward_infer(imgs, imus, mask)
    assert len(calls) == T and 'pose0' not in out and out['a'].all()


@pytest.mark.parametrize('preset', BOTH)
def test_rejected_vision_equals_no_vision(exact, preset):
    """(f) pi = 0: T^1 and T^0 of every interval are exactly equal (pose, state), in training and inference."""
    model = make_model(preset).eval()
    with torch.no_grad():
        model.PCM_net.gate_a.fill_(-1e4)          # pi = 0: every image rejected
    imgs, imus, mask, gt, q = make_batch()
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=torch.ones(B, T))
    assert torch.equal(out['pose1'], out['pose0'])
    assert float(out['pi'].abs().max()) == 0.0
    windows = model.imu_windows(imus, mask, T)
    state = VIOState(torch.randn(2, B, 1024), torch.randn(2, B, 1024), torch.full((B, 1), 0.2))
    p1, s1, _ = model.inference_step(imgs[:, 0], imgs[:, 1], windows[:, 0], state, force_visual=torch.ones(B, 1))
    p0, s0, i0 = model.inference_step(imgs[:, 0], imgs[:, 1], windows[:, 0], state, force_visual=torch.zeros(B, 1))
    assert torch.equal(p1, p0) and torch.equal(s1.h, s0.h) and torch.equal(s1.c, s0.c)
    assert torch.equal(p0, i0['pose0'])
    assert (s1.tau == 0).all() and torch.allclose(s0.tau, torch.full((B, 1), 0.3))   # a rejected call is still a call


@pytest.mark.parametrize('preset', BOTH)
def test_no_vision_path_ignores_images_over_a_sequence(preset):
    model = make_model(preset).eval()
    imgs, imus, mask, gt, q = make_batch()
    sched = torch.zeros(B, T, device=DEVICE)
    sched[:, 0] = 1
    r1, _ = model.forward_infer(imgs, imus, mask, schedule=sched)
    imgs2 = imgs.clone()
    imgs2[:, 2:] = torch.randn_like(imgs2[:, 2:])
    r2, _ = model.forward_infer(imgs2, imus, mask, schedule=sched)
    assert torch.allclose(r1['pose'][:, 2:], r2['pose'][:, 2:], atol=1e-5)


def test_prior_mean_is_capped_in_tau_and_sigma_p_grows_linearly_past_the_cap(exact):
    """(c) z_hat and the V_p part read the tau feature capped at pcm_tau_max (9); var_p = exp(V_p) + q * steps keeps
    growing, linearly in steps, past the cap."""
    model = make_model('PCM3').eval()
    fi = torch.randn(2, 256)
    h = torch.randn(2, 1024)
    dt = model.opt.dt
    at = lambda steps: model._inertial_branch(fi, h, torch.full((2, 1), steps * dt))
    assert not torch.allclose(at(9)['z_hat'], at(3)['z_hat'])
    var = {n: at(n)['lv_p'].exp() for n in (9, 12, 20, 50)}
    for n in (12, 20, 50):
        assert torch.allclose(at(n)['z_hat'], at(9)['z_hat'], atol=1e-12)
    q = model.PCM_net.q()
    assert torch.allclose(q, torch.full((2,), 0.01), atol=1e-12)
    for n in (12, 20, 50):
        assert torch.allclose(var[n] - var[9], (n - 9) * q.expand(2, 2), atol=1e-10)
    with torch.no_grad():
        model.PCM_net.q_raw.fill_(0.0)                          # q = log 2: the slope follows q
    v2 = {n: at(n)['lv_p'].exp() for n in (20, 50)}
    assert torch.allclose(v2[50] - v2[20], torch.full((2, 2), 30 * np.log(2.0)), atol=1e-10)


def test_innovation_is_centred_and_the_interval_rule_follows_tau(exact):
    model = make_model('PCM3').train()
    imgs, imus, mask, gt, q = make_batch()
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=torch.ones(B, T))
    expect = (out['vb']['z'] - out['z_hat']).reshape(-1, model.PCM_net.z_dim).mean(0)
    assert torch.allclose(model.PCM_net.eps_mean, expect)
    net = model.PCM_net.eval()
    zh = torch.randn(4, net.z_dim)
    with torch.no_grad():
        net.eps_mean.copy_(torch.randn(net.z_dim))
    _, info = net.fuse(torch.randn(4, 256), zh, torch.zeros(4, 2), zh + net.eps_mean, torch.zeros(4, 2),
                       torch.randn(4, 256))
    assert float(info['nis'].abs().max()) < 1e-12                  # a constant offset is no innovation
    # tau restarts at every call (rejected or not); the interval rule reads it; training replays the same tau
    model.eval()
    with torch.no_grad():
        model.PCM_net.gate_a.fill_(-1e4)
    res, st = model.forward_infer(imgs, imus, mask, policy_fn=make_policy_fn('interval', 3, dt=model.opt.dt))
    assert res['a'][0, 1:, 0].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]   # every 3rd interval
    assert torch.allclose(res['elapsed'][:, :, 0],
                          torch.tensor([1, 1, 2, 3, 1, 2], dtype=torch.float64).expand(B, T) * model.opt.dt)
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=res['a'][..., 0])
    assert torch.allclose(out['tau'], res['elapsed'])


def test_max_skip_forces_a_call_after_max_skip_intervals(exact):
    model = make_model('PCM3', '--max_skip', '3').eval()
    imgs, imus, mask, gt, q = make_batch()
    never = lambda s, elapsed: (torch.zeros_like(elapsed), torch.zeros_like(elapsed))
    res, _ = model.forward_infer(imgs, imus, mask, policy_fn=never)
    assert res['a'][0, :, 0].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def test_stillness_probability_scales_the_translation(exact):
    """(d) p_still -> 1 (stillness logit +50 for every code): the translation of both candidates is ~0 in training and
    inference, the rotation is unchanged; p_still -> 0 restores the regressor's translation."""
    model = make_model('PCM3').eval()
    imgs, imus, mask, gt, q = make_batch()
    sched = torch.ones(B, T)
    sched[:, 2:4] = 0
    outs = {}
    for bias in (50.0, -50.0):
        with torch.no_grad():
            model.PCM_net.H_still.weight.zero_()
            model.PCM_net.H_still.bias.fill_(bias)
        outs[bias] = (model.forward_train(imgs, imus, mask, stage=1, schedule=sched),
                      model.forward_infer(imgs, imus, mask, schedule=sched)[0])
    (still, still_inf), (moving, moving_inf) = outs[50.0], outs[-50.0]
    assert float(still['p_still'].min()) > 1 - 1e-12 and float(moving['p_still'].max()) < 1e-12
    for key in ('pose1', 'pose0'):
        assert float(still[key][..., 3:].abs().max()) < 1e-12
        assert float(moving[key][..., 3:].abs().max()) > 1e-3
        assert torch.allclose(still[key][..., :3], moving[key][..., :3], atol=1e-12)
    assert float(still_inf['pose'][..., 3:].abs().max()) < 1e-12
    assert float(still_inf['pcm_p_still'].min()) > 1 - 1e-12


# ------------------------------------------------------------------ PCM+ complement
def test_complement_carries_the_image_feature_scaled_by_the_gain():
    model = make_model('PCM3').eval()
    net = model.PCM_net
    assert net.complement and net.Phi[0].in_features == net.z_dim + 2 * 256
    assert not make_model('PCM3_nocomp').PCM_net.complement
    x_i, zh, lv = torch.randn(3, 256), torch.randn(3, net.z_dim), torch.zeros(3, 2)
    z_v = zh + net.eps_mean                                    # no innovation: only the complement differs
    xa, xb = torch.randn(3, 256), torch.randn(3, 256)
    fa, ia = net.fuse(x_i, zh, lv, z_v, lv, xa)
    fb, _ = net.fuse(x_i, zh, lv, z_v, lv, xb)
    assert float((fa - fb).abs().max()) > 1e-4                 # the image feature reaches Phi
    with pytest.raises(AssertionError):                        # PCM+ needs the image feature with every image
        net.fuse(x_i, zh, lv, z_v, lv)
    with torch.no_grad():
        net.gate_a.fill_(-1e4)                                 # rejected: the complement vanishes too
    fa0, _ = net.fuse(x_i, zh, lv, z_v, lv, xa)
    fb0, _ = net.fuse(x_i, zh, lv, z_v, lv, xb)
    f_none, _ = net.fuse(x_i, zh, lv)
    assert torch.allclose(fa0, fb0) and torch.allclose(fa0, f_none)


FUSE_SCALARS = ('gain_alpha', 'gain_beta', 'gate_a', 'gate_c')


def test_lstm_input_uses_the_detached_gain_and_mu_loss_the_live_one():
    """The pose loss must not open the gate (or raise the gain) to get more image: the LSTM input f reaches none of
    alpha, beta, a, c (sg g in mu and in the complement); mu_loss = sg(z_hat) + gain(g) * sg(eps) reaches only them."""
    net = make_model('PCM3').PCM_net.train()
    x_i, zh, lv = torch.randn(3, 256), torch.randn(3, net.z_dim, requires_grad=True), torch.zeros(3, 2)
    z_v = (zh.detach() + torch.randn(3, net.z_dim)).requires_grad_()
    f, info = net.fuse(x_i, zh, lv, z_v, lv + 0.5, torch.randn(3, 256))
    f.sum().backward()
    for n in FUSE_SCALARS:
        assert getattr(net, n).grad is None or float(getattr(net, n).grad.abs().max()) == 0.0, n
    w = net.Phi[0].weight.grad
    assert float(w[:, -256:].abs().sum()) > 0                 # Phi learns from the complement
    assert float(z_v.grad.abs().sum()) > 0                    # and from the innovation
    net.zero_grad()
    zh.grad = z_v.grad = None
    info['mu_loss'].pow(2).sum().backward()
    assert zh.grad is None and z_v.grad is None
    assert all(float(getattr(net, n).grad.abs().sum()) > 0 for n in FUSE_SCALARS)
    assert sorted(n for n, p in net.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0) \
        == sorted(FUSE_SCALARS)


def test_gain_and_gate_start_at_the_v2_values_with_the_parameter_scale():
    net = make_model('PCM3').PCM_net
    assert net.PARAM_SCALE == 10.0
    alpha, beta = net.gain_params()
    a, c = net.gate_params()
    for v, ref in ((alpha, 1.0), (beta, 0.0), (a, 4.0), (c, 1.6)):
        assert torch.allclose(v, torch.full((2,), ref), atol=1e-6)
    assert torch.allclose(torch.sigmoid(a - c * 2.5), torch.full((2,), 0.5), atol=1e-6)     # pi = 0.5 at NIS 2.5
    with torch.no_grad():                                      # one unit of the raw parameter = PARAM_SCALE units
        net.gain_alpha.add_(0.1)
        net.gate_a.add_(0.1)
    assert torch.allclose(net.gain_params()[0], torch.full((2,), 2.0)) and torch.allclose(net.gate_params()[0],
                                                                                          torch.full((2,), 5.0))


# ------------------------------------------------------------------ gradient plumbing and objectives
@pytest.mark.parametrize('preset', BOTH)
def test_every_module_trains_in_both_stages_and_infers(preset):
    model = make_model(preset).train()
    imgs, imus, mask, gt, q = make_batch()
    blank = torch.zeros(B, T, device=DEVICE)
    blank[0, 1:4] = 1
    frozen = torch.zeros(B, T, device=DEVICE)
    frozen[0, 2:4] = 1
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=L.stage1_schedule(B, T, model.opt, DEVICE))
    terms = L.stage1_loss(out, gt, q, model.opt, blank=blank, frozen=frozen)
    for key in ('pose', 'pose0', 'ground_v', 'ground_p', 'var_v', 'var_p', 'xpred', 'still', 'fuse', 'total'):
        assert torch.isfinite(terms[key]), key
    assert 'pi_loss' not in terms
    terms['total'].backward()
    L.diagnostics(out)
    dead = [n for n, p in model.named_parameters() if p.grad is None and not n.startswith('Policy_net')]
    assert dead == [], dead
    for n in FUSE_SCALARS + ('q_raw',):
        assert float(getattr(model.PCM_net, n).grad.abs().sum()) > 0, n
    model.zero_grad()
    out, terms = stage2_terms(model)
    terms['total'].backward()
    assert torch.isfinite(terms['total']) and 'p' in out
    assert model.Policy_net.net[0].in_features == model.PCM_net.z_dim + 1024 + 6 + 1
    model.eval()
    res, _ = model.forward_infer(imgs, imus, mask)
    assert res['pose'].shape == (B, T, 6)
    off = make_model(preset, '--pcm_gate', 'false').train()          # ablation: pi = 1, a and c unused
    out = off.forward_train(imgs, imus, mask, stage=1, schedule=torch.ones(B, T, device=DEVICE))
    assert float(out['pi'].min()) == 1.0
    L.stage1_loss(out, gt, q, off.opt, blank=blank, frozen=frozen)['total'].backward()
    assert off.PCM_net.gate_a.grad is None and float(off.PCM_net.gain_alpha.grad.abs().sum()) > 0


def test_straight_through_is_hard_forward_and_soft_backward():
    model = make_model('PCM3').train()
    out, terms = stage2_terms(model)
    assert set(out['a'].detach().unique().tolist()) <= {0.0, 1.0}
    terms['pose'].backward()
    assert grad_norm(model.Policy_net.parameters()) > 0


def test_policy_cost_does_not_train_the_estimator_by_default():
    for extra, leaks in (((), False), (('--policy_detach_features', 'false'), True)):
        model = make_model('PCM3', *extra).train()
        _, terms = stage2_terms(model)
        terms['cost'].backward()
        est = [p for n, p in model.named_parameters() if not n.startswith('Policy_net')]
        assert grad_norm(model.Policy_net.parameters()) > 0
        assert (grad_norm(est) > 0) == leaks
        # the prior log variances reach the policy only as a detached signal
        assert grad_norm(model.PCM_net.V_p.parameters()) == 0


def test_policy_signal_zero_removes_the_prior_variances_from_the_policy_input():
    full, zero = make_model('PCM3').eval(), make_model('PCM3', '--policy_signal', 'zero').eval()
    ib = full._inertial_branch(torch.randn(2, 256, device=DEVICE), torch.randn(2, 1024, device=DEVICE),
                               torch.full((2, 1), 0.3, device=DEVICE))
    seen = []
    for m in (full, zero):
        m.Policy_net.net[0].register_forward_hook(lambda mod, inp, out: seen.append(inp[0]))
        m._policy_logits(ib, torch.randn(2, 1024, device=DEVICE), torch.full((2, 1), 0.3, device=DEVICE))
    z = full.PCM_net.z_dim + 1024
    assert torch.equal(seen[0][:, z:z + 2], ib['lv_p']) and (seen[0][:, z + 2:z + 6] == 0).all()
    assert (seen[1][:, z:z + 6] == 0).all()


def test_stage1_policy_frozen_and_schedule_followed():
    model = make_model('PCM3').train()
    imgs, imus, mask, gt, q = make_batch()
    sched = L.sample_deploy_schedule(B, T, 0.1, 10.0, 0.1, DEVICE)
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=sched)
    L.stage1_loss(out, gt, q, model.opt)['total'].backward()
    assert grad_norm(model.Policy_net.parameters()) == 0
    assert torch.equal(out['a'][..., 0], sched.index_fill(1, torch.tensor([0], device=DEVICE), 1.0))


def test_stage1_supervises_the_no_visual_candidate_from_t0_from_step():
    model = make_model('PCM3').train()
    imgs, imus, mask, gt, q = make_batch()
    sched = L.sample_deploy_schedule(B, T, 0.1, 10.0, 0.1, DEVICE)
    with torch.no_grad():   # only the gradient reaching T^0 is inspected
        out = model.forward_train(imgs, imus, mask, stage=1, schedule=sched)
    pose0 = out['pose0'].clone().requires_grad_()
    terms = L.stage1_loss(dict(out, pose0=pose0), gt, q, model.opt)
    terms['total'].backward()
    ref = pose0.detach().clone().requires_grad_()
    L.pose_loss_per_step(ref[:, 1:], gt[:, 1:]).mean().backward()
    assert terms['pose0'].item() > 0
    assert torch.allclose(pose0.grad, ref.grad, rtol=1e-5, atol=0)
    pose0.grad = None
    L.stage1_loss(dict(out, pose0=pose0), gt, q, make_opt('PCM3', '--lambda_0', '0'))['total'].backward()
    assert pose0.grad.abs().max() == 0


def test_no_visual_supervision_skips_the_zero_state_step():
    """Step 0 starts from the zero state: T^0 cannot observe speed there, so its pose loss (stage 1 and 2) may not
    reach it; --t0_from_step 0 restores the every-interval loss."""
    model = make_model('PCM3').train()
    imgs, imus, mask, gt, q = make_batch()
    sched = L.sample_deploy_schedule(B, T, 0.1, 10.0, 0.1, DEVICE)
    with torch.no_grad():
        out1 = model.forward_train(imgs, imus, mask, stage=1, schedule=sched)
        out2 = model.forward_train(imgs, imus, mask, stage=2, temp=1.0)
    for opt_args, step0_free in (((), True), (('--t0_from_step', '0'), False)):
        opt = make_opt('PCM3', *opt_args)
        p1, p2 = out1['pose0'].clone().requires_grad_(), out2['pose0'].clone().requires_grad_()
        L.stage1_loss(dict(out1, pose0=p1), gt, q, opt)['total'].backward()
        L.stage2_loss(dict(out2, pose0=p2), gt, q, opt)['total'].backward()
        for g in (p1.grad, p2.grad):
            assert (g[:, 0].abs().max() == 0) == step0_free and g[:, 1:].abs().max() > 0


def test_stage2_anchor_supervises_both_candidates_and_keeps_the_policy_signal():
    model = make_model('PCM3').train()
    imgs, imus, mask, gt, q = make_batch()
    with torch.no_grad():
        out = model.forward_train(imgs, imus, mask, stage=2, temp=1.0)
    a = torch.ones(B, T, 1, device=DEVICE).requires_grad_()
    pose0, pose1 = out['pose0'].clone().requires_grad_(), out['pose1'].clone().requires_grad_()
    L.stage2_loss(dict(out, a=a, pose0=pose0, pose1=pose1), gt, q, model.opt)['total'].backward()
    assert (pose1.grad.abs().sum(-1) > 0).all()                 # T1 learns at every interval
    assert (pose0.grad[:, 1:].abs().sum(-1) > 0).all() and (pose0.grad[:, 0] == 0).all()   # from t0_from_step
    pose1.data[:, 1] = gt[:, 1]                                 # vision would help at step 1
    a.grad = None
    L.stage2_loss(dict(out, a=a, pose0=pose0, pose1=pose1), gt, q, model.opt)['total'].backward()
    assert bool((a.grad[:, 1, 0] < 0).all())                    # pushes towards calling vision there


def test_pcm_terms_weight_and_stage2_scaling():
    """Stage 2 adds the PCM terms with eta_r; stage 1 at full weight."""
    model = make_model('PCM3').train()
    imgs, imus, mask, gt, q = make_batch()
    with torch.no_grad():
        o1 = model.forward_train(imgs, imus, mask, stage=1, schedule=torch.ones(B, T, device=DEVICE))
        o2 = model.forward_train(imgs, imus, mask, stage=2, temp=1.0)
    t1 = L.stage1_loss(o1, gt, q, model.opt)
    assert torch.allclose(t1['total'], t1['pose'] + model.opt.lambda_0 * t1['pose0'] + t1['pcm'])
    t2 = L.stage2_loss(o2, gt, q, model.opt)
    assert torch.allclose(t2['total'], t2['pose'] + model.opt.lambda_c * t2['cost'] + model.opt.eta_r * t2['pcm'])


# ------------------------------------------------------------------ PCM losses and statistics
def test_gain_starts_at_the_variance_ratio_and_calibration_trains_only_the_variance_heads():
    """k = sigmoid(alpha (lp - lv) + beta) from detached log variances, = var_p / (var_p + var_v) at init; the V heads
    read detached features, so only the calibration NLL trains them (and q), and it never reaches the encoders."""
    model = make_model('PCM3').train()
    imgs, imus, mask, gt, q = make_batch()
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=torch.ones(B, T, device=DEVICE))
    assert torch.allclose(out['k'], torch.sigmoid(out['lv_p'] - out['vb']['lv_v']), atol=1e-6)
    L.pose_loss_per_step(out['pose1'], gt).mean().backward()
    for n, p in model.PCM_net.named_parameters():
        if n.startswith(('V_p.', 'V_v.', 'q_raw')):
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, n
    model.zero_grad()
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=torch.ones(B, T, device=DEVICE))
    terms = PL.pcm_terms(out, gt, q, None, model.opt)
    (terms['var_v'] + terms['var_p']).backward()
    assert float(model.PCM_net.V_v[-1].weight.grad.abs().sum()) > 0
    assert float(model.PCM_net.V_p[-1].weight.grad.abs().sum()) > 0
    assert float(model.PCM_net.q_raw.grad.abs().sum()) > 0          # q is trained by the var_p NLL
    for n, p in model.named_parameters():      # calibration trains the V heads and q, nothing else
        if not n.startswith(('PCM_net.V_p.', 'PCM_net.V_v.', 'PCM_net.q_raw')):
            assert p.grad is None or float(p.grad.abs().max()) == 0.0, n


def test_pose_loss_never_trains_the_fusion_scalars_and_the_fusion_loss_trains_only_them():
    """(a) The pose losses (T1, T0, stage-2 selection) reach none of alpha, beta, a, c; the fusion loss reaches exactly
    alpha, beta, a, c (not the encoders, P_i / P_v, the readout heads, Phi or the LSTM)."""
    model = make_model('PCM3').train()
    imgs, imus, mask, gt, q = make_batch()
    blank = torch.zeros(B, T, device=DEVICE)
    blank[0, 1:4] = 1
    sched = torch.ones(B, T, device=DEVICE)
    sched[:, 3:5] = 0
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=sched)
    terms = L.stage1_loss(out, gt, q, model.opt, blank=blank)
    (terms['pose'] + terms['pose0']).backward()
    for n in FUSE_SCALARS:
        assert getattr(model.PCM_net, n).grad is None, n
    model.zero_grad()
    out, t2 = stage2_terms(model)
    t2['pose'].backward()
    for n in FUSE_SCALARS:
        assert getattr(model.PCM_net, n).grad is None, n
    model.zero_grad()
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=sched)
    terms = L.stage1_loss(out, gt, q, model.opt, blank=blank)
    (model.opt.lambda_fuse * terms['fuse']).backward()
    touched = {n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert touched == {'PCM_net.' + n for n in FUSE_SCALARS}, sorted(touched)


def small_pcm(gate=False):
    torch.manual_seed(0)
    return PredictiveCommonMotion(8, 8, (16, 32, 8, 0), 16, 8, gate=gate, complement=True)


def code_errors(gen, n, z_dim, shared, own):
    """Per-dimension errors with a shared component: (n, z_dim)."""
    return shared * torch.randn(n, z_dim, generator=gen) + own * torch.randn(n, z_dim, generator=gen)


def test_learned_gain_finds_the_optimum_under_correlated_errors():
    """(b) Prior and image errors share a component (e_p = s + n_p, e_v = s + n_v). The fused error e_p + k (e_v - e_p)
    is smallest at k* = var(n_p) / (var(n_p) + var(n_v)), below the independent-error gain var_p / (var_p + var_v) when
    the prior's own error is the smaller one (the v2 audit: k_rot 0.39 vs 0.16-0.23). Two groups with different
    variance ratios (so alpha and beta are both identified); only alpha, beta train, on the fusion loss as in
    pcm_terms (readout of mu_loss with detached heads against the truth)."""
    net = small_pcm(gate=False)
    groups = [dict(s2=1.0, p2=0.25, v2=1.0), dict(s2=1.0, p2=0.5, v2=0.75)]
    for gr in groups:
        var_p, var_v = gr['s2'] + gr['p2'], gr['s2'] + gr['v2']
        gr['lp'], gr['lv'] = np.log(var_p), np.log(var_v)
        gr['k_opt'] = gr['p2'] / (gr['p2'] + gr['v2'])
        gr['k_indep'] = var_p / (var_p + var_v)
    optim = torch.optim.Adam([net.gain_alpha, net.gain_beta], lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    n, z = 256, net.z_dim
    for step in range(1500):
        zs, zh, zv, lp, lv = [], [], [], [], []
        for gr in groups:
            z_true = torch.randn(n, z, generator=gen)
            shared = np.sqrt(gr['s2']) * torch.randn(n, z, generator=gen)
            zs.append(z_true)
            zh.append(z_true + shared + np.sqrt(gr['p2']) * torch.randn(n, z, generator=gen))
            zv.append(z_true + shared + np.sqrt(gr['v2']) * torch.randn(n, z, generator=gen))
            lp.append(torch.full((n, 2), gr['lp']))
            lv.append(torch.full((n, 2), gr['lv']))
        zs, zh, zv, lp, lv = (torch.cat(x) for x in (zs, zh, zv, lp, lv))
        _, info = net.fuse(torch.zeros(2 * n, 8), zh, lp, zv, lv, torch.zeros(2 * n, 8))
        r, t, _ = net.readout(info['mu_loss'], detach_params=True)
        r0, t0, _ = net.readout(zs)
        loss = ((r - r0) ** 2).mean() + ((t - t0) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    alpha, beta = (x.detach() for x in net.gain_params())
    for gr in groups:
        k = torch.sigmoid(alpha * (gr['lp'] - gr['lv']) + beta)
        assert torch.allclose(k, torch.full((2,), gr['k_opt']), atol=0.04), (k, gr)
    k_a = torch.sigmoid(alpha * (groups[0]['lp'] - groups[0]['lv']) + beta)
    assert bool((groups[0]['k_indep'] - k_a > 0.12).all()), (k_a, groups[0]['k_indep'])


def test_fusion_loss_teaches_the_gate_to_close_on_confidently_wrong_images():
    """No gate label since v3: with clean images (small NIS, vision helps) and frozen-like ones (large NIS, the image
    is wrong), the fusion loss alone must close the gate on the wrong images and keep it open on the clean ones."""
    net = small_pcm(gate=True)
    optim = torch.optim.Adam([net.gain_alpha, net.gain_beta, net.gate_a, net.gate_c], lr=3e-3)
    gen = torch.Generator().manual_seed(1)
    n, z = 256, net.z_dim
    lp, lv = torch.zeros(2 * n, 2), torch.zeros(2 * n, 2)
    for step in range(1500):
        z_true = torch.randn(2 * n, z, generator=gen)
        zh = z_true + torch.randn(2 * n, z, generator=gen)
        e_v = 0.5 * torch.randn(2 * n, z, generator=gen)
        e_v[n:] = 3.0 * torch.randn(n, z, generator=gen)        # wrong images: large error, no relation to the truth
        _, info = net.fuse(torch.zeros(2 * n, 8), zh, lp, z_true + e_v, lv, torch.zeros(2 * n, 8))
        if step == 0:
            g0 = info['k'] * info['pi']          # clean ~0.49, wrong ~0.38
        r, t, _ = net.readout(info['mu_loss'], detach_params=True)
        r0, t0, _ = net.readout(z_true)
        loss = ((r - r0) ** 2).mean() + ((t - t0) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    # optimum per group: 0.8 (clean) and 0.1 (wrong); the per-sample NIS of the two groups overlaps, so the reachable
    # means are ~0.65 and ~0.19
    g = info['k'] * info['pi']
    clean, wrong = float(g[:n].mean()), float(g[n:].mean())
    assert clean > 0.55 and wrong < 0.25 and wrong < 0.6 * float(g0[n:].mean()) and clean - wrong > 0.35, (clean, wrong)


def test_xpred_target_is_detached_and_scale_free():
    z_v = torch.randn(4, 5, 8, requires_grad=True)
    z_hat = torch.randn(4, 5, 8, requires_grad=True)
    steps = torch.tensor([1.0, 2.0, 5.0, 9.0, 1.0]).expand(4, 5)
    w = torch.ones(4, 5)
    var = z_v.detach().reshape(-1, 8).var(0, unbiased=False)
    PL.xpred_loss(z_hat, z_v, steps, w, var).backward()
    assert z_v.grad is None and z_hat.grad is not None
    const = PL.xpred_loss(torch.zeros(4, 5, 8) + z_v.detach().reshape(-1, 8).mean(0), z_v, steps, w, var)
    assert 0.6 < float(const) < 1.4          # predicting the mean scores ~1 per dimension
    scaled = PL.xpred_loss(10 * z_hat.detach(), 10 * z_v.detach(), steps, w, 100 * var)
    assert torch.allclose(scaled, PL.xpred_loss(z_hat.detach(), z_v.detach(), steps, w, var), rtol=1e-4)


def test_zv_variance_is_an_ema_across_batches():
    model = make_model('PCM3').train()
    imgs, imus, mask, gt, q = make_batch()
    assert float(model.PCM_net.zv_var_n) == 0
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=torch.ones(B, T, device=DEVICE))
    first = out['zv_var'].clone()
    assert torch.allclose(first, out['vb']['z'].detach().reshape(-1, model.PCM_net.z_dim).var(0, unbiased=False))
    model.forward_train(2 * imgs, imus, mask, stage=1, schedule=torch.ones(B, T, device=DEVICE))
    assert float(model.PCM_net.zv_var_n) == 2 and not torch.equal(model.PCM_net.zv_var, first)
    model.eval()
    model.forward_train(imgs, imus, mask, stage=1, schedule=torch.ones(B, T, device=DEVICE))
    assert float(model.PCM_net.zv_var_n) == 2                  # eval forwards leave the statistics alone


def test_nan_mean_marks_empty_classes():
    v = torch.ones(2, 3)
    assert torch.isnan(PL.nan_mean(v, torch.zeros(2, 3)))
    assert float(PL.nan_mean(v * 2, torch.ones(2, 3))) == 2.0


def test_adaptive_weights_are_bounded_and_stillness_is_class_balanced():
    ref = torch.tensor(1e-3)
    w_solved = PL.adaptive_weight(0.1, ref, torch.tensor(1e-9), PL.STILL_FLOOR)
    assert float(w_solved) == pytest.approx(0.1 * 1e-3 / PL.STILL_FLOOR)      # no blow-up when a term is solved
    assert float(PL.adaptive_weight(0.1, ref, torch.tensor(2.0), 0.1)) == pytest.approx(0.1 * 1e-3 / 2.0)
    target = torch.zeros(1, 40)
    target[0, :2] = 1.0                       # 5% positives
    logit = torch.full((1, 40), -3.0)
    w = torch.ones(1, 40)
    bal = PL.balanced_bce(logit, target, w)
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
    assert float(bal) > 3 * float(plain)      # missed positives count as much as all negatives together
    none = PL.balanced_bce(logit, torch.zeros(1, 40), w)
    assert torch.allclose(none, torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.zeros(1, 40)))


def test_weighted_mean_is_normalized_by_valid_samples():
    v = torch.tensor([[1.0, 100.0]])
    assert L.weighted_mean(v, torch.tensor([[1.0, 0.0]])).item() == pytest.approx(1.0)
    assert L.weighted_mean(v, torch.zeros(1, 2)).item() == 0.0


# ------------------------------------------------------------------ recipe
def test_gyro_shortcut_starts_at_zero_and_integrates_the_gyro():
    model = make_model('PCM3')
    assert torch.count_nonzero(model.gyro_W.weight) == 0
    w = torch.zeros(2, 4, 11, 7)
    w[..., 5] = 0.3          # rad/s about z for one interval of dt = 0.1 s
    rv = model.gyro_rotvec(w)
    assert torch.allclose(rv[..., 2], torch.full((2, 4), 0.03)) and torch.allclose(rv[..., :2], torch.zeros(2, 4, 2))


def test_gyro_shortcut_adds_to_both_candidates_in_train_and_inference(exact):
    model = make_model('PCM3').eval()
    with torch.no_grad():
        model.gyro_W.weight.copy_(torch.eye(3))
    imgs, imus, mask, gt, q = make_batch()
    sched = torch.ones(B, T)
    sched[:, 2:4] = 0
    out = model.forward_train(imgs, imus, mask, stage=1, schedule=sched)
    res, _ = model.forward_infer(imgs, imus, mask, schedule=sched)
    sel = torch.where(sched[..., None] > 0.5, out['pose1'], out['pose0'])
    assert torch.allclose(res['pose'], sel, atol=1e-10)
    rv = model.gyro_rotvec(model.imu_windows(imus, mask, T))
    with torch.no_grad():
        model.gyro_W.weight.zero_()
    out0 = model.forward_train(imgs, imus, mask, stage=1, schedule=sched)
    assert torch.allclose(out['pose1'][:, 0, :3] - out0['pose1'][:, 0, :3], rv[:, 0], atol=1e-10)


def test_gyro_map_fit_recovers_a_known_axis_map():
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    M = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)
    samples = []
    for _ in range(50):
        w = rng.normal(0, 0.2, (101, 6))
        w[:, 3:6] = np.repeat(rng.normal(0, 0.2, (10, 3)), 10, axis=0)[np.r_[0:100, 99]]
        g = w[:, 3:6]
        rot = np.array([0.5 * (g[10 * k:10 * k + 11][:-1] + g[10 * k:10 * k + 11][1:]).sum(0) * 0.01 for k in range(10)])
        gts = np.zeros((10, 6))
        gts[:, :3] = rot @ M.T
        samples.append({'imus': w, 'gts': gts})
    fit, r2 = RC.fit_gyro_map(SimpleNamespace(samples=samples), 0.1, stride=1)
    assert np.allclose(fit, M, atol=1e-6) and (r2 > 0.999).all()


def test_stillness_windows_have_identity_labels_and_still_sensors():
    torch.manual_seed(0)
    imgs, imus, mask, gt, q = make_batch(b=64)
    imus[..., 2] += 9.8
    i2, m2, g2 = RC.apply_stillness(imgs, imus, gt, 0.5)
    still = (g2.abs().sum((1, 2)) == 0)
    assert 10 < int(still.sum()) < 54
    s = still.nonzero()[0, 0]
    assert (i2[s] - i2[s, :1]).abs().max() < 0.05 and m2[s, :, 3:].abs().max() < 0.02
    assert abs(float(m2[s, :, 2].mean()) - float(imus[s, :, 2].mean())) < 0.05
    assert torch.equal(i2[~still], imgs[~still]) and torch.equal(g2[~still], gt[~still])


def test_adamw_groups_skip_decay_on_vectors_and_stage2_freezes_the_visual_encoder():
    model = make_model('PCM3')
    opt = model.opt
    groups = RC.param_groups(model, 2, opt)
    optim = RC.make_optimizer(groups, opt)
    assert isinstance(optim, torch.optim.AdamW)
    for g in optim.param_groups:
        assert (g['weight_decay'] == 0) == g['name'].endswith('_nd')
        assert all((p.ndim <= 1) == g['name'].endswith('_nd') for p in g['params'])
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen and all(RC.is_visual_encoder_param(n) for n in frozen)
    assert not any(n.startswith('Feature_net.inertial_encoder') for n in frozen)
    in_opt = {id(p) for g in optim.param_groups for p in g['params']}
    assert all(id(p) not in in_opt for n, p in model.named_parameters() if n in frozen)
    stage1 = RC.param_groups(model, 1, opt)
    assert [g['name'] for g in stage1] == ['est'] and all(p.requires_grad for p in stage1[0]['params'])
    assert not any(p.requires_grad for p in model.Policy_net.parameters())


def test_ema_tracks_weights_and_copies_buffers():
    model = make_model('PCM3')
    ema = RC.EMA(model, 0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
        for b in model.buffers():
            if b.dtype.is_floating_point:
                b.add_(3.0)
    before = [p.clone() for p in ema.model.parameters()]
    ema.update(model)      # n = 1: decay min(0.5, 2/11)
    d = 2.0 / 11.0
    for pe, p0, p in zip(ema.model.parameters(), before, model.parameters()):
        assert torch.allclose(pe, d * p0 + (1 - d) * p)
    assert all(torch.equal(be, b) for be, b in zip(ema.model.buffers(), model.buffers()))


def test_batchnorm_recalibration_is_the_clean_average_and_keeps_momentum():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)).train()
    for _ in range(20):                       # "perturbed" training batches inflate the statistics
        net(torch.randn(32, 4) * 5)
    clean = [torch.randn(32, 4) for _ in range(6)]
    n = RC.recalibrate_batchnorm(net.eval(), clean, net, max_batches=4)
    bn = net[1]
    with torch.no_grad():
        h = [net[0](x) for x in clean[:4]]
    assert n == 4 and bn.momentum == 0.1 and not net.training
    assert torch.allclose(bn.running_mean, torch.stack([x.mean(0) for x in h]).mean(0), atol=1e-6)
    assert torch.allclose(bn.running_var, torch.stack([x.var(0) for x in h]).mean(0), atol=1e-5)


def test_cosine_lr_warms_up_linearly_then_decays_to_the_floor():
    from crvio.config import cosine_lr
    peak, total, warmup = 2.5e-4, 1000, 100
    lrs = [cosine_lr(s, total, warmup, peak, 0.01) for s in range(total)]
    assert lrs[0] == pytest.approx(peak / warmup) and lrs[warmup - 1] == pytest.approx(peak)
    assert lrs[warmup] == pytest.approx(peak) and lrs[-1] == pytest.approx(0.01 * peak)
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1:]))      # monotone after warmup
    mid = warmup + (total - warmup - 1) / 2
    assert cosine_lr(mid, total, warmup, peak, 0.01) == pytest.approx(0.505 * peak)
    assert cosine_lr(5000, total, warmup, peak, 0.01) == pytest.approx(0.01 * peak)


def test_stage2_resume_continues_the_usage_dual_variable():
    import train_crvio as TC
    opt = make_opt('PCM3')
    assert TC.resumed_lambda_c(opt, None) == opt.lambda_c
    stage1 = {'stage': 1, 'opt': dict(vars(opt), lambda_c=0.05)}
    stage2 = {'stage': 2, 'opt': dict(vars(opt), lambda_c=0.05)}
    assert TC.resumed_lambda_c(opt, stage1) == opt.lambda_c      # stage 2 starts from the initial value anyway
    assert TC.resumed_lambda_c(opt, stage2) == 0.05


def test_deploy_schedule_reaches_low_rates_and_whole_masked_windows():
    torch.manual_seed(0)
    a = L.sample_deploy_schedule(4000, 10, 0.1, 10.0, 0.1, torch.device('cpu'))
    assert (a[:, 0] == 1).all()
    rate = a[:, 1:].mean(1)
    assert 0.4 < float(rate.mean()) < 0.75
    assert float((rate == 0).float().mean()) > 0.09        # p_none plus chance
    assert float((rate < 0.3).float().mean()) > 0.2


def test_b0_markov_schedule_is_mostly_visual_with_contiguous_runs():
    """crvio.losses.sample_visual_schedule is kept for B0 (train_b0_mask.py)."""
    torch.manual_seed(0)
    a = L.sample_visual_schedule(4000, 10, 0.15, 3.0, torch.device('cpu'))
    assert (a[:, 0] == 1).all()
    assert 0.6 < a[:, 1:].mean() < 0.85
    prev, cur = a[:, :-1], a[:, 1:]
    assert 0.6 < ((prev == 0) & (cur == 0)).float().sum() / (prev == 0).float().sum() < 0.72


# ------------------------------------------------------------------ perturbation
def clean_batch(b=8, t=10):
    imgs = 0.3 * torch.rand(b, t + 1, 3, 32, 64) - 0.15
    imus = torch.randn(b, 10 * t + 1, 6)
    return imgs, imus


def test_clean_mode_is_identity():
    imgs, imus = clean_batch()
    i2, u2, m, qv, qi = apply_perturbation(imgs, imus, PerturbConfig())
    assert i2 is imgs and u2 is imus and m.all() and qv.all() and qi.all()


def test_single_modality_modes_touch_only_that_modality():
    imgs, imus = clean_batch()
    i2, u2, m, qv, qi = apply_perturbation(imgs, imus, PerturbConfig(mode='visual'), seed=3)
    assert torch.equal(u2, imus) and not torch.equal(i2, imgs) and qi.all() and m.all()
    i2, u2, m, qv, qi = apply_perturbation(imgs, imus, PerturbConfig(mode='imu'), seed=3)
    assert torch.equal(i2, imgs) and not torch.equal(u2, imus) and qv.all()


def test_perturbation_is_reproducible_in_range_and_keeps_input():
    imgs, imus = clean_batch()
    img0, imu0 = imgs.clone(), imus.clone()
    torch.manual_seed(0)
    a = apply_perturbation(imgs, imus, PerturbConfig(mode='mixed'), seed=7)
    torch.manual_seed(0)
    b = apply_perturbation(imgs, imus, PerturbConfig(mode='mixed'), seed=7)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(imgs, img0) and torch.equal(imus, imu0)
    assert a[0].min() >= -0.5 and a[0].max() <= 0.5


def test_severe_flags_interval_and_missing_mask():
    imgs, imus = clean_batch()
    _, _, _, qv, _ = apply_perturbation(imgs, imus, PerturbConfig(mode='visual', p_severe=1.0, visual_types=('blur',),
                                                                  full_window=True), seed=1)
    assert (qv == 0).all()
    _, _, _, qv, _ = apply_perturbation(imgs, imus, PerturbConfig(mode='visual', p_severe=0.0, full_window=True), seed=1)
    assert (qv == 1).all()
    _, u2, m, _, qi = apply_perturbation(imgs, imus, PerturbConfig(mode='imu', p_severe=1.0, imu_types=('missing',),
                                                                   full_window=False), seed=2)
    assert (m == 0).any() and (qi == 0).any() and (u2[m[..., 0] == 0] == 0).all()


def test_misalignment_is_never_a_positive_pair():
    imgs, imus = clean_batch()
    _, u2, _, qv, qi = apply_perturbation(imgs, imus, PerturbConfig(misalign_samples=3))
    assert (qv == 0).all() and (qi == 0).all() and not torch.equal(u2, imus)


def test_perturbation_frozen_mask_marks_the_copied_frames_and_keeps_everything_else():
    cfg = PerturbConfig(mode='visual', visual_types=('blank',))
    imgs = torch.randn(40, 11, 3, 8, 16)
    imus = torch.randn(40, 101, 6)
    torch.manual_seed(0)       # sensor noise of the blank kinds uses the torch generator
    a = apply_perturbation(imgs, imus, cfg, seed=3, return_blank=True)
    torch.manual_seed(0)
    b = apply_perturbation(imgs, imus, cfg, seed=3, return_frozen=True)
    assert len(a) == 6 and len(b) == 7
    for x, y in zip(a, b[:6]):
        assert torch.equal(x, y)
    frozen, blank, out = b[6], b[5], b[0]
    assert float(frozen.sum()) > 0 and bool((frozen <= blank).all())
    for i in range(40):
        idx = frozen[i].nonzero().flatten()
        if len(idx) == 0:
            continue
        s0 = int(idx[0])
        for t in idx.tolist():          # every marked interval starts on the copied first frame of the span
            assert torch.equal(out[i, t], out[i, s0])
        if s0 > 0:
            assert blank[i, s0 - 1] == 1 and frozen[i, s0 - 1] == 0   # reads two real frames


def test_still_probability_starts_at_the_prior_and_is_bayes_corrected():
    net = make_model('PCM3').PCM_net
    assert float(net.H_still.weight.abs().max()) == 0.0 and float(net.H_still.bias.abs().max()) == 0.0
    assert float(net.still_prob(torch.zeros(1))) == pytest.approx(net.STILL_PRIOR)
    logit = torch.tensor([math.log(net.STILL_PRIOR / (1 - net.STILL_PRIOR))])     # balanced-BCE logit of the prior
    assert float(net.still_prob(-logit)) == pytest.approx(0.5)
