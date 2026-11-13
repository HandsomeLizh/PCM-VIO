"""Predictive Common Motion, version 3 (PCM v3, 2026-11-11). Preset PCM3 (crvio/config.py).

Interface used by the backbone (crvio/vio.py):
  prior(x_i, f_i, tau_feat, steps) -> z_hat, lv_p    before the image (IMU + carried state)
  observe(x_v, fv_norm)            -> z_v, lv_v      image only
  fuse(x_i, z_hat, lv_p[, z_v, lv_v, x_v]) -> f, info   LSTM input; z_v None = no vision
  readout(z[, detach_params])      -> rotation, translation, stillness logit
  diag_init / diag_fill            -> per-sample inference diagnostics (tools/pcm_diag.py)
Losses: crvio/pcm_losses.py. Checkpoints of PCM v1 / v2 (presets PCM, PCMC) are evaluated with the snapshot named in
crvio.vio.build_model.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.LeakyReLU(0.1, inplace=True),
                         nn.Linear(hidden, out_dim))


def _branch(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.LeakyReLU(0.1, inplace=True),
                         nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))


def parse_blocks(spec):
    """'16,32,8,8' -> (16, 32, 8, 8): rotation, translation, stillness, free."""
    blocks = tuple(int(b) for b in str(spec).split(','))
    if len(blocks) != 4 or min(blocks[:3]) < 1 or blocks[3] < 0:
        raise ValueError(f'pcm_blocks needs four sizes rot,trans,still,free (free may be 0), got {spec}')
    return blocks


def bounded_logvar(raw, lo=-12.0, hi=6.0):
    return lo + (hi - lo) * torch.sigmoid(raw)


def _inv_softplus(y):
    return math.log(math.expm1(y))


class PredictiveCommonMotion(nn.Module):
    """PCM v3: the common motion is what the IMU and the carried state predict before the image arrives; the image
    corrects it by a learned, gated share of the innovation.

    Code layout (blocks): rotation | translation | stillness | free. The rotation and translation blocks have shared
    linear readouts (rotation / b_R, translation / b_d), the stillness block a shared logit readout.
      prior   z_hat = P_i([x_i; log tau]); the stillness block reads f_i only (P_s: no state, so memory cannot keep a
              stopped vehicle moving). Prior variance per readout block (rotation, translation):
                var_p = exp(bounded(V_p(sg[x_i; log tau]))) + q_b * steps,   q_b = softplus(q_raw_b), q_b(0) = 0.01
              steps = intervals since the last visual call (>= 1, the current one included). The tau feature read by
              P_i, V_p and the policy is capped at the longest skip of a training window (pcm_tau_max); the q term is
              not capped, so sigma_p keeps growing past the cap.
      observe z_v = P_v(x_v); log var_v = bounded(V_v(sg[x_v; log |f_v|])), image only (|f_v| is a texture cue)
      update  eps = z_v - z_hat - eps_mean (EMA of the innovation mean: a constant offset is no innovation)
              gain  k_b  = sigmoid(alpha_b * (sg lp_b - sg lv_b) + beta_b)          lp, lv: the two log variances
              gate  pi_b = sigmoid(a_b - c_b * NIS_b), NIS_b = |W_b eps_b|^2 / (d (var_p + var_v)), detached, in readout
                    units (W_b: the readout weights of block b)
              g_b = pi_b * k_b; the rotation gain acts on the rotation block, the translation gain on all other blocks
              alpha = S * gain_alpha (alpha(0) = 1), beta = S * gain_beta (beta(0) = 0), a = S * gate_a (a(0) = 4),
              c = softplus(S * gate_c) (c(0) = 1.6, so pi = 0.5 at NIS = 2.5), S = PARAM_SCALE = 10: one Adam step
              moves these scalars ~10x further than an ordinary weight.
      fuse    mu = z_hat + gain(sg g) * eps;  f = Phi([mu; x_i; sg(g_trans) * x_v])  (complement=True, preset PCM3)
              Without vision mu = z_hat and the complement slot is zero; pi * k = 0 gives exactly the no-vision input.
      losses  only the fusion loss trains alpha, beta, a, c: info['mu_loss'] = sg(z_hat) + gain(g) * sg(eps) is read
              out with detached readout weights (readout(..., detach_params=True)) and regressed on the motion
              (crvio/pcm_losses.py, lambda_fuse). The LSTM path uses sg(g), so the pose loss cannot reach them, and the
              fusion loss reaches neither the encoders nor the readout heads.
      still   info['p_still'] = sg sigmoid(H_still(still block of mu)) (mu with sg g with vision, z_hat without);
              the backbone scales the predicted translation by (1 - p_still) (soft zero-velocity constraint).

    Changes from v2 (PCM / PCMC, 2026-11-10/11) and why (internal signal audit and test-time ablations of the trained
    PCM+ = PCMC on the ground data):
      1. gain. v2: k = var_p / (var_p + var_v) = sigmoid(lp - lv), which is optimal only if the two errors are
         independent. Measured NIS on normal frames averaged 0.44 instead of ~1: the errors are positively correlated,
         and vision was over-trusted (k_rot 0.39 vs 0.16-0.23 optimal from the actual errors, k_trans 0.57 vs ~0.44).
         v3: k = sigmoid(alpha (lp - lv) + beta), starting exactly at the v2 gain, with alpha, beta learned from the
         actual fused error (fusion loss). The variance heads stay calibrated by their own NLL.
      2. sigma_p over long skips. v2: var_p = exp(V_p(...)) only, and the tau feature is capped at 9 intervals, so
         past ~10 skipped intervals the error / predicted variance ratio was ~1.6. v3 adds q * steps (random-walk
         growth, uncapped), q trained by the existing var_p NLL.
      3. gate. v2 trained pi by a hand-made frozen-frame label (frozen and moving -> 0, clean -> 1), class-balanced BCE
         with EMA class counts (buffers gate_counts, gate_counts_n), lambda_pi and GATE_SCALE = 100; on frozen frames
         g_rot stayed at 0.20 vs ~0.04 optimal. v3 removes the label, the counts and lambda_pi: the gate is trained,
         together with the gain, by the fusion loss on every interval (perturbed ones included, which is where it has
         to learn to close). PARAM_SCALE = 10 replaces GATE_SCALE = 100 and also scales alpha, beta.
      4. stillness. v2 detected stops well (AUC 0.95-0.97) but the detection never reached the pose: false motion at
         stops was 0.33-0.55 m, worse than the control without a stillness block (0.21 m). v3 feeds p_still to the
         translation output.
      Removed with them: the options phi_var (log var_post into Phi), detach_gain (the gain now always reads detached
      log variances; its learned part is alpha, beta) and lv_post; acceptance() and the backbone's split between
      an acceptance-weighted tau and a call counter (rejected calls not counting as fresh vision): the gate almost
      never closes to 0, so that bookkeeping changed nothing but complicated training and inference. tau is again the
      time since the last visual call.
    Kept from v2: innovation centring (eps_mean), the zv_var EMA for L_x, the gated complement of PCM+, V_v reading
    log |f_v|, detached inputs of the variance heads, diag_init / diag_fill.
    """

    PARAM_SCALE = 10.0     # alpha = S * gain_alpha, beta = S * gain_beta, a = S * gate_a, c = softplus(S * gate_c)
    Q_INIT = 0.01          # softplus(q_raw) at init: prior variance growth per interval, readout units

    def __init__(self, x_dim, i_dim, blocks, hidden, f_dim, gate=True, complement=True):
        super().__init__()
        self.blocks = tuple(int(b) for b in blocks)
        self.z_dim = sum(self.blocks)
        nr, nt, ns, nf = self.blocks
        self.P_i = _mlp(x_dim + 1, x_dim, nr + nt + nf)
        self.P_s = _mlp(i_dim, 64, ns)
        self.V_p = _mlp(x_dim + 1, 64, 2)
        self.P_v = _mlp(x_dim, x_dim, self.z_dim)
        self.V_v = _mlp(x_dim + 1, 64, 2)
        self.H_rot = nn.Linear(nr, 3)
        self.H_trans = nn.Linear(nt, 3)
        self.H_still = nn.Linear(ns, 1)
        self.gate = gate
        s = self.PARAM_SCALE
        self.gate_a = nn.Parameter(torch.full((2,), 4.0 / s))
        self.gate_c = nn.Parameter(torch.full((2,), _inv_softplus(1.6) / s))
        # v3 parameters, registered after the v2 ones (no init RNG draws: the other modules start as before)
        self.gain_alpha = nn.Parameter(torch.full((2,), 1.0 / s))
        self.gain_beta = nn.Parameter(torch.zeros(2))
        self.q_raw = nn.Parameter(torch.full((2,), _inv_softplus(self.Q_INIT)))
        self.complement = complement      # PCM+: Phi also reads the image feature, scaled by the accepted gain
        self.Phi = _mlp(self.z_dim + x_dim + (x_dim if complement else 0), hidden, f_dim)
        self.register_buffer('zv_var', torch.ones(self.z_dim))
        self.register_buffer('zv_var_n', torch.zeros(()))
        self.register_buffer('eps_mean', torch.zeros(self.z_dim))
        self.register_buffer('eps_mean_n', torch.zeros(()))

    # ------------------------------------------------------------------ parameters and statistics
    def gate_params(self):
        return self.PARAM_SCALE * self.gate_a, F.softplus(self.PARAM_SCALE * self.gate_c)

    def gain_params(self):
        return self.PARAM_SCALE * self.gain_alpha, self.PARAM_SCALE * self.gain_beta

    def q(self):
        return F.softplus(self.q_raw)

    @torch.no_grad()
    def fuse_params(self):
        """Current alpha, beta, a, c, q per block (rotation, translation), detached: training monitors."""
        alpha, beta = self.gain_params()
        a, c = self.gate_params()
        return {'alpha': alpha.detach().clone(), 'beta': beta.detach().clone(), 'a': a.detach().clone(),
                'c': c.detach().clone(), 'q': self.q().detach().clone()}

    @torch.no_grad()
    def update_eps_mean(self, eps, momentum=0.99):
        """EMA of the innovation mean (z_v - z_hat over every interval of the batch, bias-corrected start)."""
        self.eps_mean_n += 1
        m = min(momentum, 1.0 - 1.0 / float(self.eps_mean_n))
        self.eps_mean.mul_(m).add_(eps.reshape(-1, eps.shape[-1]).float().mean(0), alpha=1.0 - m)

    @torch.no_grad()
    def update_zv_var(self, z_v, momentum=0.99):
        """EMA of the per-dimension variance of z_v over training batches (bias-corrected start)."""
        var = z_v.reshape(-1, z_v.shape[-1]).float().var(0, unbiased=False)
        self.zv_var_n += 1
        m = min(momentum, 1.0 - 1.0 / float(self.zv_var_n))
        self.zv_var.mul_(m).add_(var, alpha=1.0 - m)

    STILL_PRIOR = 0.05      # share of still intervals in training (5% stillness windows + rare real stops)

    def reset_heads(self):
        """Called after the generic initialization: log variances start at 0 (unit readout scale); the stillness logit
        starts at 0 (p_still = STILL_PRIOR after the prior correction below), so an untrained detector does not
        shrink the translation."""
        for V in (self.V_p, self.V_v):
            nn.init.zeros_(V[-1].weight)
            nn.init.constant_(V[-1].bias, math.log(12.0 / 6.0))   # bounded_logvar(.) = 0
        nn.init.zeros_(self.H_still.weight)
        nn.init.zeros_(self.H_still.bias)

    def still_prob(self, logit):
        """P(still) for the soft zero-velocity constraint. The stillness BCE is class-balanced (a 50/50 prior), so the
        logit is shifted by the log prior odds of still intervals before the sigmoid (Bayes correction)."""
        return torch.sigmoid(logit + math.log(self.STILL_PRIOR / (1.0 - self.STILL_PRIOR)))

    def split(self, z):
        nr, nt, ns, nf = self.blocks
        return z[..., :nr], z[..., nr:nr + nt], z[..., nr + nt:nr + nt + ns], z[..., nr + nt + ns:]

    def _gain(self, g):
        """(B, 2) block gains -> (B, z_dim): rotation gain on the rotation block, translation gain on the rest."""
        nr = self.blocks[0]
        B = g.shape[0]
        return torch.cat([g[:, :1].expand(B, nr), g[:, 1:].expand(B, self.z_dim - nr)], -1)

    # ------------------------------------------------------------------ the three steps
    def prior(self, x_i, f_i, tau_feat, steps):
        """tau_feat: capped log tau feature (B, 1); steps: intervals since the last visual call, >= 1 (B, 1)."""
        nr, nt, ns, nf = self.blocks
        inp = torch.cat([x_i, tau_feat], -1)
        zr = self.P_i(inp)
        z_hat = torch.cat([zr[..., :nr + nt], self.P_s(f_i), zr[..., nr + nt:]], -1)
        lv_base = bounded_logvar(self.V_p(inp.detach()))
        var_p = torch.exp(lv_base) + self.q().to(lv_base.dtype) * steps
        return z_hat, torch.log(var_p)

    def observe(self, x_v, fv_norm):
        cue = torch.log(fv_norm.detach() + 1e-3)       # textureless frames: ~3.2, normal / still / frozen: 3.6-3.7
        return self.P_v(x_v), bounded_logvar(self.V_v(torch.cat([x_v.detach(), cue], -1)))

    def readout(self, z, detach_params=False):
        """detach_params: readout weights and biases detached (the fusion loss must not train the heads)."""
        zr, zt, zs, _ = self.split(z)
        if detach_params:
            head = lambda m, x: F.linear(x, m.weight.detach(), m.bias.detach())
        else:
            head = lambda m, x: m(x)
        return head(self.H_rot, zr), head(self.H_trans, zt), head(self.H_still, zs)[..., 0]

    def fuse(self, x_i, z_hat, lv_p, z_v=None, lv_v=None, x_v=None):
        """x_v (PCM+ only): the image feature, entering Phi as sg(g_trans) * x_v; zero without vision or when rejected
        (g = 0). With complement, x_v is required whenever z_v is given.
        info: k, pi, nis (B, 2), detached; mu_loss (with vision): sg(z_hat) + gain(g) * sg(eps), the only path to
        alpha, beta, a, c; p_still (B, 1), detached."""
        B = z_hat.shape[0]
        if z_v is None:
            zero = z_hat.new_zeros(B, 2)
            mu, g_sg = z_hat, zero
            info = {'k': zero, 'pi': zero, 'nis': zero}
        else:
            assert x_v is not None or not self.complement, 'PCM+ (complement) needs the image feature x_v with z_v'
            eps = z_v - z_hat - self.eps_mean.to(z_v.dtype)
            lp, lv = lv_p.detach(), lv_v.detach()
            alpha, beta = self.gain_params()
            k = torch.sigmoid(alpha * (lp - lv) + beta)
            er, et, _, _ = self.split(eps)
            ro = torch.stack([(F.linear(er, self.H_rot.weight) ** 2).mean(-1),
                              (F.linear(et, self.H_trans.weight) ** 2).mean(-1)], -1)
            nis = (ro / (lp.exp() + lv.exp())).detach()
            if self.gate:
                a, c = self.gate_params()
                pi = torch.sigmoid(a - c * nis)
            else:
                pi = torch.ones_like(k)
            g = pi * k
            g_sg = g.detach()
            mu = z_hat + self._gain(g_sg) * eps
            info = {'k': k.detach(), 'pi': pi.detach(), 'nis': nis, 'ro': ro.detach(),
                    'mu_loss': z_hat.detach() + self._gain(g) * eps.detach()}
        info['p_still'] = self.still_prob(self.H_still(self.split(mu)[2])).detach()
        inp = [mu, x_i]
        if self.complement:     # gain detached: the pose loss must not open the gate to get more image feature
            inp.append(torch.zeros_like(x_i) if z_v is None else g_sg[:, 1:] * x_v)
        return self.Phi(torch.cat(inp, -1)), info

    # ------------------------------------------------------------------ inference diagnostics
    @torch.no_grad()
    def diag_init(self, z_hat, lv_p):
        """Per-sample diagnostics of one inference interval; zeros / nan where vision was not called."""
        B = z_hat.shape[0]
        z2 = lv_p.new_zeros(B, 2)
        ro_p = self.readout(z_hat)
        nan3 = torch.full_like(ro_p[1], float('nan'))
        return {'k': z2, 'pi': z2.clone(), 'nis': z2.clone(), 'ro': torch.full_like(z2, float('nan')),
                'lv_p': lv_p, 'lv_v': torch.full_like(z2, float('nan')),
                'rot_p': ro_p[0], 'trans_p': ro_p[1], 'still_p': ro_p[2][:, None],
                'rot_v': nan3.clone(), 'trans_v': nan3.clone(), 'still_v': torch.full_like(z2[:, :1], float('nan')),
                'p_still': self.still_prob(ro_p[2][:, None])}

    @torch.no_grad()
    def diag_fill(self, diag, idx, info, z_v, lv_v):
        for key in ('k', 'pi', 'nis', 'ro', 'p_still'):
            diag[key][idx] = info[key]
        diag['lv_v'][idx] = lv_v
        ro_v = self.readout(z_v)
        diag['rot_v'][idx], diag['trans_v'][idx] = ro_v[0], ro_v[1]
        diag['still_v'][idx] = ro_v[2][:, None]
