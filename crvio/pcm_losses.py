"""PCM v3 losses: readout grounding, variance calibration, L_x, stillness, fusion loss (gain correction and gate),
plus the epoch monitors. Each term is normalized by its own valid-sample count.

v3 (2026-11-11) removed the common-cause gate BCE (gate_loss, lambda_pi, the EMA class counts and the frozen-and-moving
targets); the gate and the gain correction are trained by the fusion loss instead (crvio/pcm.py)."""
import torch
import torch.nn.functional as F

ANGLE_WEIGHT = 100.0     # baseline: 100 * angle MSE + translation MSE


def weighted_mean(values, q):
    return (values * q).sum() / q.sum().clamp(min=1.0)


PCM_STEP_BINS = (1.5, 4.5)     # skip-length bins in intervals: 1 | 2-4 | 5+ (a training window skips at most 9)


def xpred_loss(z_hat, z_v, steps, w, var):
    """L_x (PCM): |sg(z_v) - z_hat|^2 per interval divided by the EMA variance of z_v per dimension (scale-free: a
    constant prediction at the mean scores ~1 per dimension), equal weight for each skip-length bin. Inside masked
    runs z_hat is a multi-step prediction, so velocity enters the common code. No host synchronization."""
    zt = z_v.detach()
    err = ((z_hat - zt) ** 2 / (var.detach().to(zt.dtype) + 1e-6)).mean(-1)
    bins = torch.bucketize(steps.contiguous(), torch.tensor(PCM_STEP_BINS, device=steps.device, dtype=steps.dtype))
    means, used = [], []
    for b in range(len(PCM_STEP_BINS) + 1):
        wb = w * (bins == b).to(w.dtype)
        n = wb.sum()
        means.append((err * wb).sum() / n.clamp(min=1.0))
        used.append((n > 0).to(err.dtype))
    means, used = torch.stack(means), torch.stack(used)
    return (means * used).sum() / used.sum().clamp(min=1.0)


def gaussian_nll(pred, target, logvar):
    """Per-interval Gaussian NLL with one variance per block (d = last dim)."""
    d = pred.shape[-1]
    return 0.5 * (((pred - target) ** 2).sum(-1) * torch.exp(-logvar) + d * logvar)


XPRED_FLOOR, STILL_FLOOR = 0.5, 0.1     # typical sizes of L_x and the stillness BCE once learned


def adaptive_weight(kappa, ref, loss, floor):
    """kappa * sg(ref) / max(sg(loss), floor): the weighted term stays ~kappa * ref, so its gradient scales with the
    pose loss on both datasets. The floor keeps a nearly solved term from being amplified (with loss -> 0 the ratio
    would push e.g. classifier logits towards infinity at a constant rate)."""
    return kappa * ref.detach() / loss.detach().clamp(min=floor)


def balanced_bce(logit, target, w):
    """BCE with positives and negatives weighted equally (a batch without positives puts full weight on negatives).
    Still intervals are a few percent of the windows; plain BCE drove the logits to -inf between them."""
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    pos, neg = w * target, w * (1.0 - target)
    has_pos = (pos.sum() > 0).to(loss.dtype)
    return 0.5 * (weighted_mean(loss, pos) * has_pos + weighted_mean(loss, neg) * (2.0 - has_pos))


# Monitors only (no loss target since v3): |rotation| / b_R, |translation| / b_d above which a frozen image is wrong.
ROT_MOVING, TRANS_MOVING = 1.0, 0.5


def nan_mean(v, m):
    """Masked mean, NaN when the mask is empty (monitors: the epoch average skips NaN instead of averaging zeros)."""
    n = m.sum()
    return torch.where(n > 0, (v * m).sum() / n.clamp(min=1.0), torch.full_like(n, float('nan')))


def pcm_terms(out, gt, q_v, blank, opt, frozen=None, pose_ref=None):
    """PCM v3 losses.

    ground   the shared readouts of z_v and z_hat regress the motion in pose-loss units (100 b_R^2 * rotation MSE +
             b_d^2 * translation MSE, on normalized readouts), image side weighted by q_v (no blank or severe frames),
             prior side from t0_from_step; lambda_ground is dimensionless, the same on KITTI and ground data
    var      Gaussian NLL of var_v / var_p on the detached readout errors; the V heads read detached features, so this
             trains only the V heads. Image side on every interval but the frozen ones (a frozen pair looks like a
             still pair, where the image is right), prior side on every interval, step 0 included
    xpred    L_x with weight kappa_x * sg(pose) / sg(L_x): its gradient scales with the pose loss on both datasets
    still    class-balanced stillness BCE (image side without blank frames, prior side everywhere: it reads only f_i), same weighting
    fuse     readout of the fused code sg(z_hat) + pi k sg(eps) with detached readout weights (out['ro_mu']) against
             the motion, in the units of ground, on every interval incl. perturbed ones (blank, frozen, severe: the gate
             and the gain have to learn to close there); lambda_fuse. It reaches only alpha, beta, a, c.
    xpred is binned by out['tau'] / dt, the intervals since the last visual call.
    """
    B, T = gt.shape[:2]
    k0 = getattr(opt, 't0_from_step', 0)
    step_ok = torch.zeros(B, T, device=gt.device, dtype=gt.dtype)
    step_ok[:, k0:] = 1.0
    blank = torch.zeros_like(q_v) if blank is None else blank
    frozen = torch.zeros_like(q_v) if frozen is None else frozen
    vb = out['vb']
    y_rot, y_trans = gt[..., :3] / opt.b_R, gt[..., 3:] / opt.b_d
    still = (gt[..., 3:].norm(dim=-1) < opt.pcm_still_thresh * opt.b_d).to(gt.dtype)
    (rv, tv, sv), (rp, tp, sp) = out['ro_v'], out['ro_p']
    lv_v, lv_p = vb['lv_v'], out['lv_p']
    wr, wt = ANGLE_WEIGHT * opt.b_R ** 2, opt.b_d ** 2
    pose_units = lambda r, t: wr * ((r - y_rot) ** 2).mean(-1) + wt * ((t - y_trans) ** 2).mean(-1)
    ground_v = weighted_mean(pose_units(rv, tv), q_v)
    ground_p = weighted_mean(pose_units(rp, tp), step_ok)
    nll = lambda r, t, lv: gaussian_nll(r.detach(), y_rot, lv[..., 0]) + gaussian_nll(t.detach(), y_trans, lv[..., 1])
    textureless = blank * (1.0 - frozen)       # uniform / dark / cover: ~2.5% of the intervals, balanced with the rest
    has_tl = (textureless.sum() > 0).to(gt.dtype)
    nll_v = nll(rv, tv, lv_v)
    var_v = 0.5 * (weighted_mean(nll_v, textureless) * has_tl
                   + weighted_mean(nll_v, (1.0 - frozen) * (1.0 - textureless)) * (2.0 - has_tl))
    var_p = nll(rp, tp, lv_p).mean()
    still_loss = balanced_bce(sv, still, 1.0 - blank) + balanced_bce(sp, still, torch.ones_like(still))
    x = xpred_loss(out['z_hat'], vb['z'], out['tau'][..., 0] / opt.dt, q_v * step_ok, out['zv_var'])
    rm, tm, _ = out['ro_mu']
    fuse = weighted_mean(pose_units(rm, tm), torch.ones_like(q_v))
    ref = pose_ref if pose_ref is not None else gt.new_ones(())
    lam_x = adaptive_weight(opt.kappa_x, ref, x, XPRED_FLOOR)
    lam_s = adaptive_weight(opt.kappa_still, ref, still_loss, STILL_FLOOR)
    terms = {'ground_v': ground_v, 'ground_p': ground_p, 'var_v': var_v, 'var_p': var_p, 'xpred': x,
             'still': still_loss, 'fuse': fuse}
    terms['pcm'] = opt.lambda_ground * (ground_v + ground_p) + opt.lambda_var * (var_v + var_p) + lam_x * x \
        + lam_s * still_loss + opt.lambda_fuse * fuse
    with torch.no_grad():     # mechanism monitors (epoch means in the log): blank vs clean, frozen vs clean
        pi = out['pi']
        clean = (q_v >= 1.0).to(gt.dtype) * (1.0 - blank)
        # frozen intervals of a moving vehicle, per block (rotating / translating): there the image is confidently wrong
        moving = torch.stack([(y_rot.norm(dim=-1) > ROT_MOVING), (y_trans.norm(dim=-1) > TRANS_MOVING)], -1).to(gt.dtype)
        shut = frozen[..., None] * moving
        g_tr = (out['k'] * out['pi'])[..., 1]
        terms['gtr_clean'] = nan_mean(g_tr, clean)
        terms['gtr_blank'] = nan_mean(g_tr, textureless)
        terms['gtr_frozen'] = nan_mean(g_tr, shut[..., 1])
        terms['pi_frozen_rot'] = nan_mean(pi[..., 0], shut[..., 0])
        terms['pi_frozen_trans'] = nan_mean(pi[..., 1], shut[..., 1])
        terms['pi_clean'] = nan_mean(pi.mean(-1), clean)
        terms['nis_frozen'] = nan_mean(out['nis'][..., 1], shut[..., 1])
        terms['nis_clean'] = nan_mean(out['nis'][..., 1], clean)
        terms['lvv_blank'] = nan_mean(lv_v[..., 1], textureless)
        terms['lvv_clean'] = nan_mean(lv_v[..., 1], clean)
        w = clean[..., None]
        terms['eps_mean'] = (((vb['z'] - out['z_hat']) * w).sum((0, 1)) / w.sum().clamp(min=1.0)).abs().mean()
        terms['k_rot'], terms['k_trans'] = out['k'][..., 0].mean(), out['k'][..., 1].mean()
        for name, v in out.get('fuse_params', {}).items():      # alpha, beta, a, c, q per block
            terms[f'{name}_rot'], terms[f'{name}_trans'] = v[0], v[1]
        if 'p_still' in out:      # T^1 (vision at every training interval)
            ps = out['p_still'][..., 0]
            terms['pstill_still'] = nan_mean(ps, still)
            terms['pstill_moving'] = nan_mean(ps, 1.0 - still)
    return terms
