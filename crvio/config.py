"""CLI options of PCM v3 training (B0 stays in the root train.py).

PCM v3 (2026-11-11, crvio/pcm.py): learned gain correction k = sigmoid(alpha (lp - lv) + beta), gate
pi = sigmoid(a - c NIS), both trained only by the fusion loss (--lambda_fuse) instead of the hand-made frozen-frame gate
label (--lambda_pi, removed); prior variance growing linearly in the intervals since the last call; the stillness
probability scales the translation output; tau = time since the last visual call. One preset, PCM3 (= PCM+ with the
gated complement). --pcm_version 3 is written into every checkpoint's opt; crvio.vio.build_model refuses anything else.

Older designs are not served by this code: B1-B7, B1P, B7R, BM, BMC are in the frozen snapshot
/root/VIO/legacy_crvio_20261111, PCM v1 / v2 (presets PCM, PCMC) in /root/VIO/snapshot_pcm12_20261111
(crvio.vio.build_model points there for their checkpoints). Options removed with v3: --pcm_phi_var, --pcm_detach_gain,
--lambda_pi.

The training recipe is the parser default: perturbation training incl. blank frames, stage-1 vision masked at
deployment-like rates, stillness windows, speed-balanced sampling, AdamW + an EMA of the weights, the gyro shortcut, an
anchored stage 2 (both candidates supervised at every interval) with a frozen visual encoder and a call-usage target,
optional bf16 visual encoder.
"""
import argparse
import math

from .perturb import ALL_VISUAL_TYPES, IMU_TYPES

PRESETS = {
    # PCM v3 (2026-11-11): the innovation over the IMU + state prediction of a common code (rotation / translation /
    # stillness blocks), with a learned gain correction and gate trained by the fusion loss, plus the image feature
    # scaled by the detached translation gain (the gated complement of PCM+). See crvio/pcm.py.
    'PCM3': dict(pcm_complement=True),
}

# Values for option keys that a checkpoint's opt may lack: test_crvio.py and tools/pcm_diag.py build the model from
# ckpt['opt']. Every v3 checkpoint carries the full v3 option set, so this is empty; an option added later goes here
# with the value that reproduces the earlier behaviour. Never add pcm_version: checkpoints without it are PCM v1 / v2.
CKPT_DEFAULTS = dict()


def str2bool(v):
    return str(v).lower() in ('1', 'true', 'yes', 'y')


def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--preset', default='PCM3', choices=sorted(PRESETS))
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--experiment_name', default='experiment')
    p.add_argument('--workers', type=int, default=4)

    # Splits. The validation sequence is for monitoring only; the last epoch is the result.
    p.add_argument('--train_seq', nargs='+', default=['00', '01', '02', '04', '06', '08'])
    p.add_argument('--val_seq', nargs='+', default=['09'])
    p.add_argument('--test_seq', nargs='+', default=['05', '07', '10'])

    # Baseline network settings (unchanged defaults).
    p.add_argument('--img_w', type=int, default=512)
    p.add_argument('--img_h', type=int, default=256)
    p.add_argument('--v_f_len', type=int, default=512)
    p.add_argument('--i_f_len', type=int, default=256)
    p.add_argument('--imu_dropout', type=float, default=0)
    p.add_argument('--rnn_hidden_size', type=int, default=1024)
    p.add_argument('--rnn_dropout_out', type=float, default=0.2)
    p.add_argument('--rnn_dropout_between', type=float, default=0.2)
    p.add_argument('--seq_len', type=int, default=11)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--weight_decay', type=float, default=5e-6, help='coupled L2 of --optimizer adam')
    p.add_argument('--dt', type=float, default=0.1, help='image interval in seconds')

    # Sizes
    p.add_argument('--x_dim', type=int, default=256, help='width of the modality projections x_v, x_i')
    p.add_argument('--h_proj_dim', type=int, default=128, help='width of the state projection read by A_i')
    p.add_argument('--direct_hidden', type=int, default=1244,
                   help='hidden width of Phi. 1244 = the capacity match of the old common-branch fusion at the '
                        'default sizes, used by every PCM run so far; the key name is kept for checkpoint compatibility')

    # Recipe: perturbation, masking, sampling
    p.add_argument('--perturb', default='mixed', choices=['none', 'visual', 'imu', 'mixed'])
    p.add_argument('--visual_types', nargs='+', default=list(ALL_VISUAL_TYPES), choices=ALL_VISUAL_TYPES,
                   help='visual perturbation kinds; blank (covered / dark / frozen frames) is where the fusion loss '
                        'teaches the gate and the gain to close')
    p.add_argument('--imu_types', nargs='+', default=list(IMU_TYPES), choices=IMU_TYPES)
    p.add_argument('--mask_rate_min', type=float, default=0.1,
                   help='stage-1 vision schedule: per window a visual rate in [mask_rate_min, 1] ...')
    p.add_argument('--mask_run_max', type=float, default=10.0, help='... and a mean masked run in [1, mask_run_max]')
    p.add_argument('--mask_p_none', type=float, default=0.1, help='share of windows visual at step 0 only')
    p.add_argument('--still_aug', type=float, default=0.05,
                   help='share of training windows replaced by a stationary window (repeated frame + sensor noise, '
                        'IMU = mean specific force + noise, zero gyro, identity labels)')
    p.add_argument('--speed_balance', type=str2bool, default=True,
                   help='sample training windows with weight count(speed bin)^-0.5 (slow windows are rare)')

    # Recipe: model and optimization
    p.add_argument('--gyro_shortcut', type=str2bool, default=True,
                   help='add W * (integrated gyro of the interval) to the predicted rotation of both candidates; '
                        'W is a learned 3x3 map (IMU to camera axes), least-squares initialized on training windows')
    p.add_argument('--optimizer', default='adamw', choices=['adam', 'adamw'],
                   help='adam = coupled L2 (weight_decay), as B0; adamw = decoupled decay adamw_wd, none on 1-d params')
    p.add_argument('--adamw_wd', type=float, default=0.01)
    p.add_argument('--ema_decay', type=float, default=0.9995,
                   help='>0: keep an EMA of the weights; validation and the saved "model" use it ("model_raw" '
                        'holds the trained weights for --resume)')
    p.add_argument('--amp', default='none', choices=['none', 'bf16'],
                   help='bf16 autocast for the visual encoder only (FlowNet + visual head); the rest stays float32')

    # Visual-call policy
    p.add_argument('--use_policy', type=str2bool, default=True, help='false: stage 1 only, tested at full vision')
    p.add_argument('--policy_signal', default='full', choices=['full', 'zero'],
                   help='full: the policy reads the prior log variances; zero: ablation without them')
    p.add_argument('--policy_detach_features', type=str2bool, default=True,
                   help='true matches the baseline, which detaches every policy input, so the cost term '
                        'cannot train the estimator')
    p.add_argument('--max_skip', type=int, default=0, help='safety fallback, report separately; 0 = off')

    # PCM v3 (crvio/pcm.py, crvio/pcm_losses.py)
    p.add_argument('--pcm_version', type=int, default=3,
                   help='written into the checkpoint opt; crvio.vio.build_model builds version 3 only')
    p.add_argument('--pcm_complement', type=str2bool, default=None,
                   help='PCM+: Phi also reads sg(pi * k_trans) * x_v (gated image feature); None keeps the preset')
    p.add_argument('--pcm_blocks', default='16,32,8,0',
                   help='code blocks rotation,translation,stillness,free (free has no grounding; 0 since review 2)')
    p.add_argument('--pcm_gate', type=str2bool, default=True,
                   help='common-cause gate pi = sigmoid(a - c * NIS) per readout block; false = pi = 1 (ablation)')
    p.add_argument('--pcm_tau_max', type=float, default=9.0,
                   help='cap (in intervals) of the tau feature read by the prior and the policy: the longest skip of a '
                        'training window. The linear prior-variance growth q * steps is not capped')
    p.add_argument('--lambda_ground', type=float, default=0.05,
                   help='readout grounding in pose-loss units (100 b_R^2 rot + b_d^2 trans MSE), image side on q_v, '
                        'prior side from t0_from_step; dimensionless, same meaning on KITTI and ground data')
    p.add_argument('--lambda_var', type=float, default=1e-3,
                   help='variance calibration NLL on detached readouts; trains only the V heads (detached inputs), so '
                        'under Adam the value only matters for the global gradient norm: at 1.0 the V heads made ~90%% '
                        'of it and the norm clipping at 1.0 shrank every pose gradient ~50x (first pcm_g run)')
    p.add_argument('--kappa_x', type=float, default=0.2,
                   help='L_x weight relative to the pose loss: lambda_x = kappa_x * sg(pose) / sg(L_x)')
    p.add_argument('--kappa_still', type=float, default=0.1,
                   help='stillness BCE weight relative to the pose loss, as kappa_x')
    p.add_argument('--lambda_fuse', type=float, default=0.05,
                   help='fusion loss: readout of sg(z_hat) + pi k sg(eps) (detached heads) against the motion, in the '
                        'units of --lambda_ground, on every interval incl. perturbed ones; the only loss that trains the '
                        'gain correction alpha, beta and the gate a, c')
    p.add_argument('--pcm_still_thresh', type=float, default=0.05,
                   help='an interval is still when |translation| < this * b_d')

    # Fixed residual scales; 0 = estimate from training poses once and store in the checkpoint.
    p.add_argument('--b_R', type=float, default=0.0)
    p.add_argument('--b_d', type=float, default=0.0)

    # Objective (crvio/losses.py)
    p.add_argument('--lambda_0', type=float, default=1.0, help='weight of the pose loss on the no-visual candidate T^0')
    p.add_argument('--t0_from_step', type=int, default=1,
                   help='first interval of a window where T^0 is supervised. Step 0 starts from the zero state, where '
                        'speed is unobservable without vision; at test time step 0 is always visual')
    p.add_argument('--lambda_c', type=float, default=3e-4,
                   help='initial weight of the expected call cost (per interval); steered by --target_usage')
    p.add_argument('--c_extra', type=float, default=1.0, help='C_extra / (C_base + C_extra), from tools/profile_cost.py')
    p.add_argument('--eta_r', type=float, default=0.1, help='stage-2 weight of the PCM terms')
    p.add_argument('--target_usage', type=float, default=0.25,
                   help='>0: stage 2 steers lambda_c each iteration so that the mean call probability reaches this')
    p.add_argument('--lambda_c_lr', type=float, default=0.01, help='step of log(lambda_c) per unit usage error')

    # Stages. Every stage: linear warmup, then cosine to min_lr_ratio * peak, set at every iteration from the
    # global step (resume needs no scheduler state).
    p.add_argument('--stage', default='both', choices=['1', '2', 'both'])
    p.add_argument('--epochs_stage1', type=int, default=60)
    p.add_argument('--lr_stage1', type=float, default=2.5e-4, help='peak lr')
    p.add_argument('--epochs_stage2', type=int, default=60)
    p.add_argument('--lr_policy', type=float, default=1e-4)
    p.add_argument('--lr_est', type=float, default=1e-5)
    p.add_argument('--stage2_freeze_visual', type=str2bool, default=True,
                   help='stage 2 keeps the visual encoder (FlowNet + visual_head) fixed, in eval mode')
    p.add_argument('--freeze_est_epochs', type=int, default=0, help='optional stabilization at stage 2 start')
    p.add_argument('--temp_init', type=float, default=5.0)
    p.add_argument('--eta', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=float, default=1.0)
    p.add_argument('--min_lr_ratio', type=float, default=0.01)
    p.add_argument('--grad_clip', type=float, default=1.0, help='global norm; 0 = off')
    p.add_argument('--bn_recalib_batches', type=int, default=0,
                   help='after the last epoch of a stage with --perturb, recompute BatchNorm statistics on this '
                        'many clean training batches (off: the running averages already match the mixed batches)')
    p.add_argument('--pretrain_flownet', default='./pretrain_models/flownets_bn_EPE2.459.pth.tar')
    p.add_argument('--pretrain', default=None, help='checkpoint to continue from in a new run (policy may be new)')
    p.add_argument('--resume', default=None,
                   help='stage*_last.pth of an interrupted run: restores model, optimizer, epoch and RNG. The '
                        'cosine horizon is part of the run, so keep the epoch counts of the original command')
    p.add_argument('--img_cache', default='', help='folder from tools/cache_images.py (frames already resized)')
    p.add_argument('--print_frequency', type=int, default=20)
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--eval_every', type=int, default=5)
    p.add_argument('--limit_batches', type=int, default=0, help='debug only')
    p.add_argument('--stop_after_epoch', type=int, default=-1, help='debug only: simulate an interruption')
    return p


def apply_preset(opt):
    """Preset values fill what the command line left open."""
    for key, value in PRESETS[opt.preset].items():
        if getattr(opt, key, None) is None:
            setattr(opt, key, value)
    opt.fusion = 'pcm'      # saved with the checkpoint: crvio.vio.build_model dispatches on it
    from .pcm import parse_blocks
    parse_blocks(opt.pcm_blocks)
    return opt


def load_opt(ckpt_opt):
    """Namespace of a checkpoint's opt dict; keys that older PCM checkpoints lack take CKPT_DEFAULTS."""
    from types import SimpleNamespace
    return SimpleNamespace(**{**CKPT_DEFAULTS, **ckpt_opt})


def cosine_lr(step, total, warmup, peak, min_ratio):
    """Linear warmup over `warmup` steps, then cosine from peak to min_ratio * peak at the last step.
    Also used by B0 (train.py, train_b0_mask.py)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    span = max(total - warmup - 1, 1)
    progress = min((step - warmup) / span, 1.0)
    return peak * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
