"""PCM v3 visual-inertial odometry: shared backbone + PredictiveCommonMotion fusion + visual-call policy.

  f_v = FlowNet(I_{t-1}, I_t), f_i = IMU encoder(window, mask)        Feature_net (baseline encoders)
  x_v = A_v(f_v), x_i = A_i([f_i; state_proj(h_{t-1})])               modality projections
  z_hat, lv_p = PCM.prior(x_i, f_i, log tau, steps)    z_v, lv_v = PCM.observe(x_v, |f_v|)
  f, info = PCM.fuse(...)  ->  LSTM  ->  regressor (+ gyro shortcut on rotation), translation * (1 - p_still)
  policy p(call) = Policy_net(z_hat, h_{t-1}, lv_p, log tau), before any visual encoding.
Both candidates of an interval (T1 with vision, T0 without) start from the same h_{t-1}; each scales its translation by
the p_still of its own fuse.

State (h, c, tau): tau = seconds since the last visual call, reset by a call (tau_next = (1 - a) * elapsed,
elapsed = tau + dt). The prior, the policy, the interval rule and max_skip all read elapsed; the prior gets the tau
feature capped at pcm_tau_max and steps = elapsed / dt uncapped (sigma_p grows linearly in it). v2 kept a second,
acceptance-weighted tau (rejected calls did not count as fresh vision) next to a call counter; v3 dropped it (see the
PredictiveCommonMotion docstring).

Checkpoint contract: attribute names and registration order below are the state_dict keys and their order (also the
parameters() order that --resume optimizer states and the EMA rely on) and the order of the init RNG draws. Do not
rename or reorder: tests/test_pcm.py checks them against tests/data/pcm3_state_keys.json (PCM v3; v1 / v2
checkpoints do not load and are refused by build_model).
"""
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Encoder, initialization
from .pcm import PredictiveCommonMotion, _branch, parse_blocks


class VIOState(NamedTuple):
    h: torch.Tensor     # (layers, B, H)
    c: torch.Tensor     # (layers, B, H)
    tau: torch.Tensor   # (B, 1) seconds since the last visual call


def select_state(a, cand0, cand1):
    """(1 - a) * cand0 + a * cand1 for an LSTM tensor (layers, B, H), a is (B, 1). With a
    straight-through a this is an exact forward selection that still passes the soft gradient."""
    a = a.view(1, -1, 1)
    return (1.0 - a) * cand0 + a * cand1


class CallPolicy(nn.Module):
    """p_t = pi(z_hat, h_{t-1}, lv_p, log tau). Index 0 = call vision, 1 = skip (baseline order).
    The 6-wide signal slot holds the two prior log variances and 4 zeros (kept for checkpoint shapes)."""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 2))

    def forward(self, z_hat, h_top, signal, tau_feat):
        return self.net(torch.cat([z_hat, h_top, signal, tau_feat], -1))


class PCMVIO(nn.Module):
    pcm = True      # tools/pcm_diag.py switches its PCM metrics on this

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        H, x_dim = opt.rnn_hidden_size, opt.x_dim
        f_dim = opt.v_f_len + opt.i_f_len      # the baseline LSTM input size

        # Baseline backbones, reused. The extra IMU channel carries the sample mask.
        self.Feature_net = Encoder(opt)
        self.Feature_net.inertial_encoder.encoder_conv[0] = nn.Conv1d(7, 64, kernel_size=3, padding=1)
        self.state_proj = nn.Linear(H, opt.h_proj_dim)
        self.A_v = _branch(opt.v_f_len, x_dim, x_dim)
        self.A_i = _branch(opt.i_f_len + opt.h_proj_dim, x_dim, x_dim)
        self.PCM_net = PredictiveCommonMotion(
            x_dim, opt.i_f_len, parse_blocks(opt.pcm_blocks), opt.direct_hidden, f_dim,
            gate=opt.pcm_gate, complement=opt.pcm_complement)
        self.rnn = nn.LSTM(input_size=f_dim, hidden_size=H, num_layers=2,
                           dropout=opt.rnn_dropout_between, batch_first=True)
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(nn.Linear(H, 128), nn.LeakyReLU(0.1, inplace=True), nn.Linear(128, 6))
        self.Policy_net = CallPolicy(self.PCM_net.z_dim + H + 6 + 1) if opt.use_policy else None
        # Rotation shortcut from the integrated gyro of each interval; zero here, least squares in train_crvio.py.
        self.gyro_W = nn.Linear(3, 3, bias=False) if getattr(opt, 'gyro_shortcut', False) else None

        initialization(self)
        if self.gyro_W is not None:
            nn.init.zeros_(self.gyro_W.weight)
        self.PCM_net.reset_heads()

    # ------------------------------------------------------------------ encoders
    def encode_visual_pair(self, pair):
        """pair: (N, 6, H, W). The expensive call that inference must be able to skip.
        --amp bf16 runs only this encoder under bf16 autocast; its output and every later module stay float32."""
        amp = getattr(self.opt, 'amp', 'none') == 'bf16' and pair.is_cuda
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            feat = self.Feature_net.encode_image(pair)
            out = self.Feature_net.visual_head(feat.flatten(1))
        return out.float() if amp else out

    @staticmethod
    def imu_windows(imus, imu_mask, T):
        x = torch.cat([imus, imu_mask], -1)
        return torch.stack([x[:, t * 10:t * 10 + 11] for t in range(T)], 1)  # (B, T, 11, 7)

    def encode_imu(self, windows):
        """windows: (B, T, 11, 7) -> (B, T, 256)."""
        return self.Feature_net.inertial_encoder(windows)

    def gyro_rotvec(self, windows):
        """Trapezoidal integral of the gyro over each interval: (..., 11, 7) -> (..., 3), IMU axes, radians."""
        if self.gyro_W is None:
            return None
        g = windows[..., 3:6]
        return 0.5 * (g[..., :-1, :] + g[..., 1:, :]).sum(-2) * (self.opt.dt / 10.0)

    # ------------------------------------------------------------------ branches
    def _tau_cap(self):
        """The longest skip of a training window: neither the prior nor the policy extrapolates in tau."""
        return getattr(self.opt, 'pcm_tau_max', 9) * self.opt.dt

    def _tau_feature(self, elapsed):
        return torch.log1p(elapsed / self.opt.dt)

    def _visual_branch(self, fv):
        x = self.A_v(fv)
        z, lv_v = self.PCM_net.observe(x, fv.norm(dim=-1, keepdim=True))
        return {'x': x, 'z': z, 'lv_v': lv_v}

    def _inertial_branch(self, fi, h_top, elapsed):
        """elapsed: seconds since the last visual call, this interval included. The prior reads the capped tau feature
        and the uncapped interval count steps = elapsed / dt (>= 1)."""
        h_proj = self.state_proj(h_top)
        x = self.A_i(torch.cat([fi, h_proj], -1))
        z_hat, lv_p = self.PCM_net.prior(x, fi, self._tau_feature(elapsed.clamp(max=self._tau_cap())),
                                         elapsed / self.opt.dt)
        return {'x': x, 'z_hat': z_hat, 'lv_p': lv_p, 'h_proj': h_proj}

    def _fuse(self, ib, vb=None):
        if vb is None:
            return self.PCM_net.fuse(ib['x'], ib['z_hat'], ib['lv_p'])
        return self.PCM_net.fuse(ib['x'], ib['z_hat'], ib['lv_p'], vb['z'], vb['lv_v'], vb['x'])

    def _decode(self, f, h, c, gyro=None, p_still=None):
        """p_still (B, 1): soft zero-velocity constraint, the translation is scaled by (1 - p_still)."""
        out, (h1, c1) = self.rnn(f.unsqueeze(1), (h, c))
        pose = self.regressor(self.rnn_drop_out(out[:, 0]))
        if p_still is not None:
            pose = torch.cat([pose[:, :3], pose[:, 3:] * (1.0 - p_still)], -1)
        if gyro is not None:
            pose = pose + F.pad(self.gyro_W(gyro), (0, 3))
        return pose, h1, c1

    def _policy_logits(self, ib, h_top, elapsed):
        z_hat, signal = ib['z_hat'], F.pad(ib['lv_p'], (0, 4)).detach()
        if self.opt.policy_signal == 'zero':
            signal = torch.zeros_like(signal)
        if self.opt.policy_detach_features:
            z_hat, h_top = z_hat.detach(), h_top.detach()
        return self.Policy_net(z_hat, h_top, signal, self._tau_feature(elapsed.clamp(max=self._tau_cap())))

    def _max_skip_reached(self, elapsed):
        """max_skip intervals since the last call (half an interval of slack against float accumulation in tau)."""
        return elapsed >= (self.opt.max_skip - 0.5) * self.opt.dt

    def init_state(self, B, device):
        H = self.opt.rnn_hidden_size
        zeros = torch.zeros(2, B, H, device=device)
        return VIOState(zeros, zeros.clone(), torch.zeros(B, 1, device=device))

    # ------------------------------------------------------------------ training path
    def forward_train(self, imgs, imus, imu_mask, stage, temp=1.0, schedule=None):
        """Both candidates at every interval; the visual encoder runs on all intervals.
        stage=1: a_t from `schedule` (B, T), first interval always visual. stage=2: a_t from the policy via
        straight-through Gumbel-Softmax. The state is committed along the selected path in both stages.
        out['ro_mu']: readout of the fused code sg(z_hat) + gain(g) * sg(eps) with detached readout weights, the
        fusion loss's input (the only loss that trains the gain and gate scalars)."""
        B, n_frames = imgs.shape[:2]
        T = n_frames - 1
        pairs = torch.cat([imgs[:, :-1], imgs[:, 1:]], 2)
        fv = self.encode_visual_pair(pairs.flatten(0, 1)).view(B, T, -1)
        vb_all = self._visual_branch(fv)
        windows = self.imu_windows(imus, imu_mask, T)
        fi = self.encode_imu(windows)
        gyro = self.gyro_rotvec(windows)

        h, c, tau = self.init_state(B, imgs.device)
        need_no_visual = stage == 2 or schedule is not None
        rec = {k: [] for k in ('pose0', 'pose1', 'a', 'p', 'tau', 'z_hat', 'lv_p', 'k', 'pi', 'nis', 'mu_loss',
                               'p_still')}
        for t in range(T):
            h_top = h[-1]
            elapsed = tau + self.opt.dt
            ib = self._inertial_branch(fi[:, t], h_top, elapsed)
            vb = {k: v[:, t] for k, v in vb_all.items()}
            g_t = None if gyro is None else gyro[:, t]
            f1, finfo = self._fuse(ib, vb)
            pose1, h1, c1 = self._decode(f1, h, c, g_t, finfo['p_still'])      # from h_{t-1}, never from h0
            if need_no_visual:
                f0, info0 = self._fuse(ib)
                pose0, h0, c0 = self._decode(f0, h, c, g_t, info0['p_still'])
            forced = t == 0
            if stage == 2 and not forced:
                logits = self._policy_logits(ib, h_top, elapsed)
                a = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)[:, :1]
                rec['p'].append(F.softmax(logits, -1)[:, :1])
            else:
                a = imgs.new_ones(B, 1) if schedule is None or forced else schedule[:, t:t + 1]
            h = select_state(a, h0, h1) if need_no_visual else h1
            c = select_state(a, c0, c1) if need_no_visual else c1
            tau = (1.0 - a.detach()) * elapsed

            rec['pose1'].append(pose1); rec['a'].append(a); rec['tau'].append(elapsed)
            if need_no_visual:
                rec['pose0'].append(pose0)
            # finfo of the fused candidate T^1 (vision exists at every training interval)
            rec['z_hat'].append(ib['z_hat']); rec['lv_p'].append(ib['lv_p'])
            rec['k'].append(finfo['k']); rec['pi'].append(finfo['pi']); rec['nis'].append(finfo['nis'])
            rec['mu_loss'].append(finfo['mu_loss']); rec['p_still'].append(finfo['p_still'])

        out = {k: torch.stack(v, 1) for k, v in rec.items() if len(v)}
        # shared readouts of both sides (grounding, calibration), of the fused code with detached heads (fusion loss),
        # EMA statistics (L_x, innovation centring)
        out['ro_v'] = self.PCM_net.readout(vb_all['z'])
        out['ro_p'] = self.PCM_net.readout(out['z_hat'])
        out['ro_mu'] = self.PCM_net.readout(out['mu_loss'], detach_params=True)
        if self.training:
            self.PCM_net.update_zv_var(vb_all['z'].detach())
            self.PCM_net.update_eps_mean(vb_all['z'].detach() - out['z_hat'].detach())
        out['zv_var'] = self.PCM_net.zv_var.clone()
        out['fuse_params'] = self.PCM_net.fuse_params()
        out['vb'] = vb_all
        return out

    # ------------------------------------------------------------------ inference path
    def _decide(self, p, rule):
        if rule['kind'] == 'threshold':
            return (p >= rule['value']).float()
        return (torch.rand_like(p) < p).float()     # same law as the baseline's hard Gumbel test

    @torch.no_grad()
    def inference_step(self, image_prev, image_now, imu_window, state, force_visual=None, rule=None, policy_fn=None):
        """One interval. The visual encoder runs only for samples with a_t = 1.
        image_*: (B, C, H, W); imu_window: (B, 11, 7); `state` is never modified in place."""
        rule = rule or {'kind': 'sample'}
        B = imu_window.shape[0]
        h, c, tau = state
        h_top = h[-1]
        fi = self.encode_imu(imu_window.unsqueeze(1))[:, 0]
        gyro = self.gyro_rotvec(imu_window)
        elapsed = tau + self.opt.dt
        ib = self._inertial_branch(fi, h_top, elapsed)

        full_visual = self.Policy_net is None and policy_fn is None \
            and (force_visual is None or bool(force_visual.all()))
        pose0, h0, c0 = None, None, None
        if not full_visual:
            f0, info0 = self._fuse(ib)
            pose0, h0, c0 = self._decode(f0, h, c, gyro, info0['p_still'])

        if force_visual is not None:
            a, p = force_visual, None
        elif policy_fn is not None:
            a, p = policy_fn(None, elapsed)      # comparison policies: (s, elapsed); PCM has no s
        elif self.Policy_net is not None:
            p = F.softmax(self._policy_logits(ib, h_top, elapsed), -1)[:, :1]
            a = self._decide(p, rule)
        else:
            a, p = imu_window.new_ones(B, 1), None
        if self.opt.max_skip > 0:
            a = torch.where(self._max_skip_reached(elapsed), torch.ones_like(a), a)

        pose, h_new, c_new = pose0, h0, c0
        diag = self.PCM_net.diag_init(ib['z_hat'], ib['lv_p'])
        idx = a[:, 0].nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            pair = torch.cat([image_prev[idx], image_now[idx]], 1)
            vb = self._visual_branch(self.encode_visual_pair(pair))
            ib_sub = {k: v[idx] for k, v in ib.items()}
            f1, finfo = self._fuse(ib_sub, vb)
            self.PCM_net.diag_fill(diag, idx, finfo, vb['z'], vb['lv_v'])
            pose1, h1, c1 = self._decode(f1, h[:, idx], c[:, idx], None if gyro is None else gyro[idx],
                                         finfo['p_still'])
            if full_visual:
                pose, h_new, c_new = pose1, h1, c1
            else:
                pose, h_new, c_new = pose0.clone(), h0.clone(), c0.clone()
                pose[idx], h_new[:, idx], c_new[:, idx] = pose1, h1, c1

        new_state = VIOState(h_new, c_new, (1.0 - a) * elapsed)
        info = {'a': a, 'p': p, 'pose0': pose0, 'elapsed': elapsed, 'pcm': diag}
        return pose, new_state, info

    @torch.no_grad()
    def forward_infer(self, imgs, imus, imu_mask=None, state: Optional[VIOState] = None, is_first=True,
                      rule=None, policy_fn=None, schedule=None):
        """Chunked inference over (B, T+1, ...) inputs, carrying state between chunks."""
        B, n_frames = imgs.shape[:2]
        T = n_frames - 1
        if imu_mask is None:
            imu_mask = torch.ones(*imus.shape[:2], 1, device=imus.device, dtype=imus.dtype)
        windows = self.imu_windows(imus, imu_mask, T)
        state = self.init_state(B, imgs.device) if state is None else state
        rec = {'pose': [], 'a': [], 'p': [], 'pose0': [], 'elapsed': []}
        for t in range(T):
            force = None
            if t == 0 and is_first:
                force = imgs.new_ones(B, 1)   # one fixed visual initialization, as in the baseline
            elif schedule is not None:
                force = schedule[:, t:t + 1]
            pose, state, info = self.inference_step(imgs[:, t], imgs[:, t + 1], windows[:, t], state,
                                                    force_visual=force, rule=rule, policy_fn=policy_fn)
            rec['pose'].append(pose)
            if info['pose0'] is not None:
                rec['pose0'].append(info['pose0'])
            rec['a'].append(info['a'])
            rec['elapsed'].append(info['elapsed'])
            if info['p'] is not None:
                rec['p'].append(info['p'])
            for key, v in info['pcm'].items():
                rec.setdefault('pcm_' + key, []).append(v)
        out = {k: torch.stack(v, 1) for k, v in rec.items() if len(v)}
        return out, state


LEGACY_SNAPSHOT = '/root/VIO/legacy_crvio_20261111'      # B1-B7, B1P, B7R, BM, BMC
PCM12_SNAPSHOT = '/root/VIO/snapshot_pcm12_20261111'     # PCM v1 / v2: presets PCM, PCMC


def build_model(opt):
    """PCMVIO for PCM v3 options (opt.fusion == 'pcm', opt.pcm_version == 3); older checkpoints are refused."""
    fusion = getattr(opt, 'fusion', None)
    if fusion != 'pcm':
        raise ValueError(f"preset {getattr(opt, 'preset', '?')} (fusion={fusion}) is an older design that this code no "
                         f"longer serves; evaluate its checkpoints with the frozen snapshot: cd {LEGACY_SNAPSHOT} && "
                         f"python test_crvio.py --ckpt <checkpoint> ...")
    version = getattr(opt, 'pcm_version', None)
    if version != 3:
        raise ValueError(f"preset {getattr(opt, 'preset', '?')} is PCM v1 / v2 (pcm_version={version}); this code builds "
                         f"PCM v3 only. Evaluate old PCM / PCMC checkpoints with the snapshot: cd {PCM12_SNAPSHOT} && "
                         f"python test_crvio.py --ckpt <checkpoint> ...")
    return PCMVIO(opt)
