"""Two-stage training of PCM v3 (preset PCM3, crvio/vio.py). B0 stays in train.py.

  Stage 1: encoders, PCM fusion, LSTM, regressor; vision follows a deployment-rate masking schedule, both candidates
           (T1 with vision, T0 without) are supervised.
  Stage 2: policy at its own lr, estimator at a smaller lr (visual encoder frozen), state follows hard decisions,
           lambda_c steered to --target_usage.
Every stage: fresh optimizer, linear warmup then cosine (crvio.config.cosine_lr) set per iteration, gradient-norm
clipping, RNG re-seeded at the stage start, and the last epoch is the result. Recipe helpers: crvio/recipe.py.
"""
import json
import logging
import math
import os
import random

import numpy as np
import torch
from path import Path

from crvio import losses as L
from crvio.config import apply_preset, build_parser, cosine_lr
from crvio.evaluation import Evaluator, make_policy_fn
from crvio.geometry import euler_to_matrix, so3_log
from crvio.perturb import PerturbConfig, apply_perturbation
from crvio.recipe import (EMA, apply_stillness, fit_gyro_map, make_optimizer, param_groups, recalibrate_batchnorm,
                          speed_balanced_weights, visual_encoder_modules)
from crvio.vio import build_model
from dataset.KITTI_dataset import KITTI
from utils import custom_transform
from utils.utils import read_pose_from_text


def estimate_scales(data_dir, sequences):
    """Fixed b_R, b_d from training poses only: per-axis RMS of the relative motion."""
    rel = np.vstack([read_pose_from_text(Path(data_dir) / f'poses/{s}.txt')[1] for s in sequences])
    rot = so3_log(euler_to_matrix(torch.as_tensor(rel[:, :3]))).numpy()
    return float(np.sqrt((rot ** 2).mean())), float(np.sqrt((rel[:, 3:] ** 2).mean()))


def setup_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(message)s')
    for h in (logging.FileHandler(path), logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def load_flownet(model, path, logger):
    if not path or not os.path.exists(path):
        logger.info(f'FlowNet weights not found at {path}; visual encoder starts from scratch')
        return
    weights = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    target = model.Feature_net.state_dict()
    match = {k: v for k, v in weights.items() if k in target and target[k].shape == v.shape}
    target.update(match)
    model.Feature_net.load_state_dict(target)
    logger.info(f'loaded {len(match)} FlowNet tensors from {path}')


def rng_state(device, loader_gen):
    s = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state(), 'python': random.getstate(),
         'loader': loader_gen.get_state()}
    if device.type == 'cuda':
        s['cuda'] = torch.cuda.get_rng_state(device)
    return s


def set_rng_state(s, device, loader_gen):
    torch.set_rng_state(s['torch'])
    np.random.set_state(s['numpy'])
    random.setstate(s['python'])
    loader_gen.set_state(s['loader'])
    if 'cuda' in s and device.type == 'cuda':
        torch.cuda.set_rng_state(s['cuda'], device)


def seed_all(seed, loader_gen):
    """Called at every stage start, after the model is built: every run with the same seed then sees the same
    batches, masking schedules, perturbations and stage-2 Gumbel noise."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader_gen.manual_seed(seed)


def make_train_set(opt):
    transform = [custom_transform.ToTensor()]
    if not opt.img_cache:     # cached frames are already resized
        transform.append(custom_transform.Resize((opt.img_h, opt.img_w)))
    train_set = KITTI(opt.data_dir, sequence_length=opt.seq_len, train_seqs=opt.train_seq,
                      transform=custom_transform.Compose(transform), img_cache=opt.img_cache or None)
    if opt.img_cache and train_set.cache_size != [opt.img_h, opt.img_w]:
        raise RuntimeError(f'image cache holds {train_set.cache_size}, expected {[opt.img_h, opt.img_w]}')
    return train_set


def clean_batch_runner(model, stage, temp, device, opt):
    """run_batch for recalibrate_batchnorm: the stage's training forward on unperturbed windows."""
    clean = PerturbConfig()

    def run(batch):
        imgs, imus = batch[0].to(device).float(), batch[1].to(device).float()
        imgs, imus, imu_mask, _, _ = apply_perturbation(imgs, imus, clean)
        if stage == 1:
            schedule = L.stage1_schedule(imgs.shape[0], imgs.shape[1] - 1, opt, device)
            model.forward_train(imgs, imus, imu_mask, stage=1, schedule=schedule)
        else:
            model.forward_train(imgs, imus, imu_mask, stage=2, temp=temp)
    return run


def resumed_lambda_c(opt, resume):
    """lambda_c is the dual variable of the usage target and changes every stage-2 iteration; a run resumed inside
    stage 2 continues from the value saved with that checkpoint instead of restarting from the command line."""
    if resume is not None and resume['stage'] == 2 and opt.target_usage > 0 and 'lambda_c' in resume['opt']:
        return float(resume['opt']['lambda_c'])
    return opt.lambda_c


def train_epoch(model, optimizer, loader, stage, temp, perturb_cfg, device, opt, logger, ep, sched, ema=None):
    """sched = (first global step of this epoch, total steps of the stage, warmup steps)."""
    sums, counts, count = {}, {}, 0
    step0, total, warmup = sched
    for i, (imgs, imus, gts, _, _) in enumerate(loader):
        if opt.limit_batches and i >= opt.limit_batches:
            break
        for g in optimizer.param_groups:
            g['lr'] = cosine_lr(step0 + i, total, warmup, g['peak'], opt.min_lr_ratio)
        imgs, imus, gts = imgs.to(device).float(), imus.to(device).float(), gts.to(device).float()
        imgs, imus, gts = apply_stillness(imgs, imus, gts, opt.still_aug)
        imgs, imus, imu_mask, q_v, q_i, blank, frozen = apply_perturbation(imgs, imus, perturb_cfg, return_frozen=True)

        optimizer.zero_grad()
        if stage == 1:
            schedule = L.stage1_schedule(imgs.shape[0], imgs.shape[1] - 1, opt, device)
            out = model.forward_train(imgs, imus, imu_mask, stage=1, schedule=schedule)
            terms = L.stage1_loss(out, gts, q_v, opt, blank=blank, frozen=frozen)
        else:
            out = model.forward_train(imgs, imus, imu_mask, stage=2, temp=temp)
            terms = L.stage2_loss(out, gts, q_v, opt, blank=blank, frozen=frozen)

        loss = terms['total']
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite loss at epoch {ep}, batch {i}: '
                               + str({k: float(v) for k, v in terms.items()}))
        loss.backward()
        values = {k: float(v) for k, v in terms.items()}
        if opt.grad_clip > 0:   # the norm before clipping
            values['gnorm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip))
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if stage == 2 and opt.target_usage > 0 and 'p' in out:
            # Dual ascent on the call cost: log lambda_c follows the gap between the mean call probability and
            # the target (lambda_c from KITTI left the ground policies at 0.5-8% vision).
            gap = float(out['p'].mean()) - opt.target_usage
            opt.lambda_c = float(min(max(opt.lambda_c * math.exp(opt.lambda_c_lr * gap), 1e-7), 1e-1))
            values['lambda_c'] = opt.lambda_c

        values['usage'] = float(out['a'].mean())
        for k, v in values.items():
            if math.isfinite(v):    # NaN marks a monitor whose class is absent from this batch
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
        count += 1
        if i % opt.print_frequency == 0:
            values.update(L.diagnostics(out))
            logger.info(f'ep {ep} iter {i}/{len(loader)} ' + ' '.join(f'{k}={v:.5g}' for k, v in values.items()))
    return {k: v / max(counts[k], 1) for k, v in sums.items()}


def main():
    opt = apply_preset(build_parser().parse_args())
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    device = torch.device(f'cuda:{opt.gpu}' if torch.cuda.is_available() and opt.gpu >= 0 else 'cpu')

    run_dir = Path('./results') / opt.experiment_name
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.makedirs_p()
    logger = setup_logger(opt.experiment_name, str(run_dir / 'train.log'))

    if opt.resume and opt.pretrain:
        raise ValueError('use --resume (same run) or --pretrain (new run), not both')
    resume = torch.load(opt.resume, map_location='cpu', weights_only=False) if opt.resume else None
    pretrain = torch.load(opt.pretrain, map_location='cpu', weights_only=False) if opt.pretrain else None
    source = resume or pretrain
    if source is not None and opt.b_R == 0:   # scales stay fixed across stages
        opt.b_R, opt.b_d = source['opt']['b_R'], source['opt']['b_d']
    if opt.b_R == 0 or opt.b_d == 0:
        opt.b_R, opt.b_d = estimate_scales(opt.data_dir, opt.train_seq)
    opt.lambda_c = resumed_lambda_c(opt, resume)
    logger.info(json.dumps(vars(opt), indent=1))

    train_set = make_train_set(opt)
    loader_gen = torch.Generator()
    if opt.speed_balance:
        weights, speeds = speed_balanced_weights(train_set)
        slow = speeds < np.quantile(speeds, 0.1)
        logger.info(f'speed-balanced windows: slowest 10% (< {np.quantile(speeds, 0.1):.3f} m/frame) drawn with '
                    f'probability {float(weights[torch.as_tensor(slow)].sum() / weights.sum()):.3f}')
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=opt.batch_size, drop_last=True, num_workers=opt.workers, pin_memory=True,
            sampler=torch.utils.data.WeightedRandomSampler(weights, len(train_set), replacement=True,
                                                           generator=loader_gen), persistent_workers=opt.workers > 0)
    else:
        loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                             num_workers=opt.workers, pin_memory=True, generator=loader_gen,
                                             persistent_workers=opt.workers > 0)
    iters = min(len(loader), opt.limit_batches) if opt.limit_batches else len(loader)
    warmup = round(opt.warmup_epochs * iters)
    evaluator = Evaluator(opt, opt.val_seq)

    model = build_model(opt)
    if resume is not None:
        model.load_state_dict(resume.get('model_raw', resume['model']))
        logger.info(f'resuming {opt.resume}: stage {resume["stage"]} after epoch {resume["epoch"]}')
    elif pretrain is not None:
        # Only the policy may stay at its init (a run without a policy continued into stage 2).
        missing, unexpected = model.load_state_dict(pretrain['model'], strict=False)
        bad = unexpected + [k for k in missing if not k.startswith('Policy_net.')]
        if bad:
            raise RuntimeError(f'checkpoint does not match this preset: {bad[:8]}')
        logger.info(f'continuing from {opt.pretrain}; newly initialized: {sorted({k.split(".")[0] for k in missing})}')
    else:
        load_flownet(model, opt.pretrain_flownet, logger)
        if model.gyro_W is not None:
            M, r2 = fit_gyro_map(train_set, opt.dt)
            with torch.no_grad():
                model.gyro_W.weight.copy_(torch.as_tensor(M, dtype=model.gyro_W.weight.dtype))
            logger.info(f'gyro shortcut initialized by least squares on training windows: M={np.round(M, 3).tolist()} '
                        f'R2={np.round(r2, 4).tolist()}')
    model.to(device)
    logger.info('trainable parameters: %d' % sum(p.numel() for p in model.parameters()))

    if opt.perturb == 'none':
        perturb_cfg = PerturbConfig()
    else:
        perturb_cfg = PerturbConfig(mode=opt.perturb, visual_types=tuple(opt.visual_types), imu_types=tuple(opt.imu_types))

    stages = {'1': [1], '2': [2], 'both': [1, 2]}[opt.stage]
    if not opt.use_policy:
        stages = [s for s in stages if s == 1]    # no policy to learn

    ema = None

    def save(name, stage, ep, optimizer=None):
        ck = {'model': (ema.model if ema is not None else model).state_dict(), 'opt': vars(opt), 'stage': stage,
              'epoch': ep}
        if ema is not None:     # 'model' is the EMA (tested, and stage 2 starts from it); this resumes training
            ck['model_raw'], ck['ema_n'] = model.state_dict(), ema.n
        if optimizer is not None:   # enough to resume the run
            ck['optimizer'] = optimizer.state_dict()
            ck['rng'] = rng_state(device, loader_gen)
        torch.save(ck, ckpt_dir / name)

    for stage in stages:
        if resume is not None and stage < resume['stage']:
            continue
        epochs = opt.epochs_stage1 if stage == 1 else opt.epochs_stage2
        first_ep = 0
        if stage == 2 and 1 in stages and (resume is None or resume['stage'] == 1):
            model.load_state_dict(torch.load(ckpt_dir / 'stage1_last.pth', map_location='cpu', weights_only=False)['model'])
            logger.info('stage 2 starts from stage1_last.pth')
        optimizer = make_optimizer(param_groups(model, stage, opt), opt)
        ema = None
        if opt.ema_decay > 0:
            ema = EMA(model, opt.ema_decay)
            if resume is not None and stage == resume['stage'] and 'model_raw' in resume:
                ema.model.load_state_dict(resume['model'])
                ema.n = resume.get('ema_n', 0)
        best_val = float('inf')
        seed_all(opt.seed + 1000 * stage, loader_gen)
        if resume is not None and stage == resume['stage']:
            first_ep = resume['epoch'] + 1
            optimizer.load_state_dict(resume['optimizer'])
            set_rng_state(resume['rng'], device, loader_gen)
        total = epochs * iters
        for ep in range(first_ep, epochs):
            temp = 1.0
            if stage == 2:
                temp = opt.temp_init * math.exp(-opt.eta * ep)
                if opt.freeze_est_epochs > 0:
                    freeze = ep < opt.freeze_est_epochs
                    for g in optimizer.param_groups:
                        if g['name'].startswith('est'):
                            for p in g['params']:
                                p.requires_grad = not freeze

            model.train()
            if stage == 2 and opt.stage2_freeze_visual:
                for m in visual_encoder_modules(model):    # fixed features: no dropout, no BatchNorm updates
                    m.eval()
            stats = train_epoch(model, optimizer, loader, stage, temp, perturb_cfg, device, opt,
                                logger, ep, (ep * iters, total, warmup), ema=ema)
            lrs = ' '.join(f"lr_{g['name']}={g['lr']:.2e}" for g in optimizer.param_groups)
            logger.info(f'stage {stage} epoch {ep} temp={temp:.3f} {lrs} ' + ' '.join(f'{k}={v:.5g}' for k, v in stats.items()))
            if ep == epochs - 1 and opt.perturb != 'none' and opt.bn_recalib_batches > 0:
                n = recalibrate_batchnorm(model, loader, clean_batch_runner(model, stage, temp, device, opt),
                                          min(opt.bn_recalib_batches, iters))
                logger.info(f'stage {stage}: BatchNorm statistics recomputed on {n} clean batches')

            if (ep + 1) % opt.save_every == 0:
                save(f'stage{stage}_{ep:03d}.pth', stage, ep)

            if (ep + 1) % opt.eval_every == 0 or ep == epochs - 1:
                # Monitoring only: the last epoch is the result. The policy is untrained in stage 1, so stage 1
                # always validates at full vision.
                policy_fn = make_policy_fn('all') if stage == 1 and opt.use_policy else None
                errors = evaluator.evaluate(ema.model if ema is not None else model, device, policy_fn=policy_fn)
                m = errors['mean']
                if m['t_rel'] < best_val:      # kept for reference; the result is still the last epoch
                    best_val = m['t_rel']
                    save(f'stage{stage}_best.pth', stage, ep)
                logger.info(f'stage {stage} epoch {ep} VAL ' + ' '.join(f'{k}={v:.4f}' for k, v in m.items()))
            save(f'stage{stage}_last.pth', stage, ep, optimizer)
            if ep == opt.stop_after_epoch:
                logger.info(f'stopping after epoch {ep} (--stop_after_epoch)')
                return
        logger.info(f'stage {stage} finished')


if __name__ == '__main__':
    main()
