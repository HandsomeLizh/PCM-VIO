"""Training objectives of the two stages (PCM v3) and the stage-1 visual schedules.

  stage 1: l(T1) + lambda_0 * l(T0 from t0_from_step) + PCM terms
  stage 2: l(T1) + lambda_0 * l(T0) + selected-path loss on detached candidates (teaches the policy through the
           straight-through a) + lambda_c * expected call cost + eta_r * PCM terms
The PCM terms (crvio/pcm_losses.py) include the fusion loss, the only loss that trains the gain correction and the gate;
the pose losses cannot reach them (the LSTM path uses the detached gain).
"""
import torch

from .pcm_losses import ANGLE_WEIGHT, pcm_terms, weighted_mean  # noqa: F401


def pose_loss_per_step(pose, gt):
    """The baseline pose loss, kept per (B, T) so candidates can be selected per sample."""
    angle = ((pose[..., :3] - gt[..., :3]) ** 2).mean(-1)
    trans = ((pose[..., 3:] - gt[..., 3:]) ** 2).mean(-1)
    return ANGLE_WEIGHT * angle + trans


def stage1_loss(out, gt, q_v, opt, blank=None, frozen=None):
    terms = {'pose': pose_loss_per_step(out['pose1'], gt).mean()}
    total = terms['pose']
    k = opt.t0_from_step     # T^0 from the zero state is unobservable, see config
    if 'pose0' in out:
        terms['pose0'] = pose_loss_per_step(out['pose0'][:, k:], gt[:, k:]).mean()
        total = total + opt.lambda_0 * terms['pose0']
    terms.update(pcm_terms(out, gt, q_v, blank, opt, frozen, pose_ref=terms['pose']))
    terms['total'] = total + terms['pcm']
    return terms


def stage2_loss(out, gt, q_v, opt, blank=None, frozen=None):
    a = out['a'][..., 0]
    l0 = pose_loss_per_step(out['pose0'], gt)
    l1 = pose_loss_per_step(out['pose1'], gt)
    k = opt.t0_from_step
    terms = {'select': ((1.0 - a) * l0.detach() + a * l1.detach()).mean()}
    terms['pose'] = l1.mean() + opt.lambda_0 * l0[:, k:].mean() + terms['select']
    terms['cost'] = opt.c_extra * out['p'].mean()      # C_base is constant here and enters the report
    terms.update(pcm_terms(out, gt, q_v, blank, opt, frozen, pose_ref=l1.mean()))
    terms['total'] = terms['pose'] + opt.lambda_c * terms['cost'] + opt.eta_r * terms['pcm']
    return terms


@torch.no_grad()
def effective_rank(z):
    """exp(entropy) of the centered covariance spectrum (Roy & Vetterli 2007); 1 = one direction used."""
    sv = torch.linalg.svdvals((z - z.mean(0)).float())
    p = sv ** 2 / (sv ** 2).sum().clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()


def diagnostics(out):
    """Gains, gate, variances (rotation / translation), stillness, code ranks: logged every print_frequency
    iterations."""
    diag = {}
    zv, zh = out['vb']['z'].flatten(0, 1).detach(), out['z_hat'].flatten(0, 1).detach()
    for i, name in enumerate(('rot', 'trans')):
        diag[f'k_{name}'] = out['k'][..., i].mean().item()
        diag[f'pi_{name}'] = out['pi'][..., i].mean().item()
        diag[f'lvv_{name}'] = out['vb']['lv_v'][..., i].mean().item()
        diag[f'lvp_{name}'] = out['lv_p'][..., i].mean().item()
    diag['nis'] = out['nis'].mean().item()
    if 'p_still' in out:
        diag['p_still'] = out['p_still'].mean().item()
    diag['zv_erank'] = effective_rank(zv)
    diag['zhat_erank'] = effective_rank(zh)
    return diag


def sample_deploy_schedule(B, T, rate_min, run_max, p_none, device):
    """Stage-1 availability at deployment-like rates. Per window: a visual rate rho ~ U[rate_min, 1] and a mean
    masked-run length L ~ U[1, run_max] set a two-state Markov chain with stationary visual share rho
    (p_drop = (1/L)(1 - rho)/rho, capped at 1); with probability p_none the window is visual at step 0 only."""
    rho = rate_min + (1.0 - rate_min) * torch.rand(B, device=device)
    leave = 1.0 / (1.0 + (run_max - 1.0) * torch.rand(B, device=device))
    p_drop = (leave * (1.0 - rho) / rho).clamp(max=1.0)
    a = torch.ones(B, T, device=device)
    for t in range(1, T):
        u = torch.rand(B, device=device)
        a[:, t] = torch.where(a[:, t - 1] > 0.5, (u >= p_drop).float(), (u < leave).float())
    none = torch.rand(B, device=device) < p_none
    a[none, 1:] = 0.0
    return a


def stage1_schedule(B, T, opt, device):
    return sample_deploy_schedule(B, T, opt.mask_rate_min, opt.mask_run_max, opt.mask_p_none, device)


def sample_visual_schedule(B, T, p_drop, mean_skip, device):
    """B0 only (train.py, train_b0_mask.py import it from here): Markov chain with contiguous masked runs."""
    a = torch.ones(B, T, device=device)
    leave_skip = 1.0 / max(mean_skip, 1.0)
    for t in range(1, T):
        u = torch.rand(B, device=device)
        a[:, t] = torch.where(a[:, t - 1] > 0.5, (u >= p_drop).float(), (u < leave_skip).float())
    return a
