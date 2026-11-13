import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from model import DeepVIO
from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import math
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crvio.config import cosine_lr  # noqa: E402  the schedule shared with train_crvio.py

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='/nfs/turbo/coe-hunseok/mingyuy/KITTI_odometry', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

parser.add_argument('--train_seq', nargs='+', default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', nargs='+', default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
# Unified ladder protocol (2026-11-04), the same as train_crvio.py. The original 40 / 40 / 20 epochs at
# 5e-4 / 5e-5 / 1e-6 map to: stage 1 = the policy-free warmup with random vision, stage 2 = joint + fine
# with the Gumbel policy. Each stage: fresh Adam, linear warmup then cosine, set per iteration.
parser.add_argument('--stage', type=str, default='both', choices=['1', '2', 'both'])
parser.add_argument('--epochs_stage1', type=int, default=60, help='random vision, policy frozen')
parser.add_argument('--epochs_stage2', type=int, default=60, help='Gumbel policy')
parser.add_argument('--lr_stage1', type=float, default=2.5e-4, help='stage 1 peak lr')
parser.add_argument('--lr_policy', type=float, default=1e-4, help='stage 2 peak lr of the policy')
parser.add_argument('--lr_est', type=float, default=1e-5, help='stage 2 peak lr of everything else')
parser.add_argument('--warmup_epochs', type=float, default=1.0)
parser.add_argument('--min_lr_ratio', type=float, default=0.01)
parser.add_argument('--grad_clip', type=float, default=1.0, help='global norm; 0 = off')
parser.add_argument('--p_vision', type=float, default=0.5, help='stage 1 probability of the visual encoder')
parser.add_argument('--eval_every', type=int, default=5)
parser.add_argument('--save_every', type=int, default=10)
parser.add_argument('--resume', type=str, default=None, help='stage*_last.pth of an interrupted run (same epoch counts)')
parser.add_argument('--limit_batches', type=int, default=0, help='debug only')
parser.add_argument('--stop_after_epoch', type=int, default=-1, help='debug only: simulate an interruption')
parser.add_argument('--eta', type=float, default=0.05, help='exponential decay factor for temperature')
parser.add_argument('--temp_init', type=float, default=5, help='initial temperature for gumbel-softmax')
parser.add_argument('--Lambda', type=float, default=3e-5, help='penalty factor for the visual encoder usage')

parser.add_argument('--experiment_name', type=str, default='experiment', help='experiment name')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')

parser.add_argument('--pretrain_flownet',type=str, default='./pretrain_models/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain', type=str, default=None, help='weights to start from (state dict or stage*_last.pth), e.g. for --stage 2')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--img_cache', type=str, default='', help='folder from tools/cache_images.py (frames already resized)')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')

parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')

args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if os.environ.get('VIO_DETERMINISTIC'):   # tests only: makes the GPU resume check bitwise
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_all(seed, loader_gen):
    """At every stage start, after the model is built (the same rule as train_crvio.py)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader_gen.manual_seed(seed)


def rng_state(loader_gen):
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(), 'numpy': np.random.get_state(),
            'python': random.getstate(), 'loader': loader_gen.get_state()}


def set_rng_state(s, loader_gen):
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state(s['cuda'])
    np.random.set_state(s['numpy'])
    random.setstate(s['python'])
    loader_gen.set_state(s['loader'])


def make_optimizer(model, stage, args):
    """Stage 1: policy frozen and outside the optimizer. Stage 2: policy and estimator groups."""
    policy = list(model.module.Policy_net.parameters())
    ids = {id(p) for p in policy}
    est = [p for p in model.parameters() if id(p) not in ids]
    for p in policy:
        p.requires_grad = stage == 2
    for p in est:
        p.requires_grad = True
    if stage == 1:
        groups = [{'params': est, 'lr': args.lr_stage1, 'peak': args.lr_stage1, 'name': 'est'}]
    else:
        groups = [{'params': policy, 'lr': args.lr_policy, 'peak': args.lr_policy, 'name': 'policy'},
                  {'params': est, 'lr': args.lr_est, 'peak': args.lr_est, 'name': 'est'}]
    if args.optimizer == 'SGD':
        return torch.optim.SGD(groups, momentum=0.9)
    return torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)


def load_weights(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck['model'] if isinstance(ck, dict) and 'model' in ck else ck


def train(model, optimizer, train_loader, selection, temp, logger, ep, sched, p=0.5, weighted=False):
    """sched = (first global step of this epoch, total steps of the stage, warmup steps)."""
    step0, total, warmup = sched
    
    mse_losses = []
    penalties = []
    data_len = len(train_loader)

    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):
        if args.limit_batches and i >= args.limit_batches:
            break
        for g in optimizer.param_groups:
            g['lr'] = cosine_lr(step0 + i, total, warmup, g['peak'], args.min_lr_ratio)

        imgs = imgs.cuda().float()
        imus = imus.cuda().float()
        gts = gts.cuda().float() 
        weight = weight.cuda().float()

        optimizer.zero_grad()
                
        poses, decisions, probs, _ = model(imgs, imus, is_first=True, hc=None, temp=temp, selection=selection, p=p)
        
        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight/weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        
        pose_loss = 100 * angle_loss + translation_loss        
        penalty = (decisions[:,:,0].float()).sum(-1).mean()
        loss = pose_loss + args.Lambda * penalty 
        
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        if i % args.print_frequency == 0: 
            message = f'Epoch: {ep}, iters: {i}/{data_len}, pose loss: {pose_loss.item():.6f}, penalty: {penalty.item():.6f}, loss: {loss.item():.6f}'
            print(message)
            logger.info(message)

        mse_losses.append(pose_loss.item())
        penalties.append(penalty.item())

    return np.mean(mse_losses), np.mean(penalties)


def main():

    # Create Dir
    experiment_dir = Path('./results')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir_p()
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir_p()
    
    # Create logs
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt'%args.experiment_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    
    # Load the dataset
    transform_train = [custom_transform.ToTensor()]
    if not args.img_cache:
        transform_train += [custom_transform.Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [custom_transform.RandomHorizontalFlip()]
    if args.color:
        transform_train += [custom_transform.RandomColorAug()]
    transform_train = custom_transform.Compose(transform_train)

    train_dataset = KITTI(args.data_dir,
                        sequence_length=args.seq_len,
                        train_seqs=args.train_seq,
                        transform=transform_train,
                        img_cache=args.img_cache or None
                        )
    if args.img_cache and train_dataset.cache_size != [args.img_h, args.img_w]:
        raise RuntimeError('image cache holds {}, expected {}'.format(train_dataset.cache_size, [args.img_h, args.img_w]))
    logger.info('train_dataset: ' + str(train_dataset))
    
    loader_gen = torch.Generator()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
        generator=loader_gen,
    )
    iters = min(len(train_loader), args.limit_batches) if args.limit_batches else len(train_loader)
    warmup = round(args.warmup_epochs * iters)

    # GPU selections
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    # Initialize the tester
    tester = KITTI_tester(args)

    # Model initialization
    model = DeepVIO(args)
    resume = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if resume is not None:
        model.load_state_dict(resume['model'])
        logger.info(f'resuming {args.resume}: stage {resume["stage"]} after epoch {resume["epoch"]}')
    elif args.pretrain is not None:
        model.load_state_dict(load_weights(args.pretrain))
        logger.info('load model %s' % args.pretrain)
    else:
        logger.info('Training from scratch')
        if args.pretrain_flownet:   # the pre-trained flownet
            pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
            model_dict = model.Feature_net.state_dict()
            update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
            model_dict.update(update_dict)
            model.Feature_net.load_state_dict(model_dict)

    # Feed model to GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    def save(name, stage, ep, optimizer=None):
        if optimizer is None:   # plain state dict, loadable by test.py
            torch.save(model.module.state_dict(), f'{checkpoints_dir}/{name}')
            return
        torch.save({'model': model.module.state_dict(), 'stage': stage, 'epoch': ep, 'args': vars(args),
                    'optimizer': optimizer.state_dict(), 'rng': rng_state(loader_gen)}, f'{checkpoints_dir}/{name}')

    stages = {'1': [1], '2': [2], 'both': [1, 2]}[args.stage]
    for stage in stages:
        if resume is not None and stage < resume['stage']:
            continue
        epochs = args.epochs_stage1 if stage == 1 else args.epochs_stage2
        optimizer = make_optimizer(model, stage, args)
        seed_all(args.seed + 1000 * stage, loader_gen)
        first_ep = 0
        if resume is not None and stage == resume['stage']:
            first_ep = resume['epoch'] + 1
            optimizer.load_state_dict(resume['optimizer'])
            set_rng_state(resume['rng'], loader_gen)
        total = epochs * iters
        for ep in range(first_ep, epochs):
            if stage == 1:
                selection, temp = 'random', args.temp_init
            else:
                selection, temp = 'gumbel-softmax', args.temp_init * math.exp(-args.eta * ep)
            model.train()
            avg_pose_loss, avg_penalty_loss = train(model, optimizer, train_loader, selection, temp, logger, ep,
                                                    (ep * iters, total, warmup), p=args.p_vision)
            lrs = ' '.join(f"lr_{g['name']}={g['lr']:.2e}" for g in optimizer.param_groups)
            message = (f'stage {stage} epoch {ep} selection={selection} temp={temp:.3f} {lrs} '
                       f'pose={avg_pose_loss:.6g} penalty={avg_penalty_loss:.6g}')
            print(message)
            logger.info(message)

            if (ep + 1) % args.save_every == 0:
                save(f'stage{stage}_{ep:03d}.pth', stage, ep)
            if (ep + 1) % args.eval_every == 0 or ep == epochs - 1:
                # Monitoring only, the last epoch is the result. Stage 1 at full vision (the policy is
                # untrained), stage 2 with the Gumbel policy as in test.py.
                with torch.no_grad():
                    model.eval()
                    if stage == 1:
                        errors = tester.eval(model, 'random', num_gpu=len(gpu_ids), p=1.0)
                    else:
                        errors = tester.eval(model, 'gumbel-softmax', num_gpu=len(gpu_ids))
                m = {k: np.mean([e[k] for e in errors]) for k in ('t_rel', 'r_rel', 't_rmse', 'r_rmse', 'usage')}
                message = f'stage {stage} epoch {ep} VAL ' + ' '.join(f'{k}={v:.4f}' for k, v in m.items())
                print(message)
                logger.info(message)
            save(f'stage{stage}_last.pth', stage, ep, optimizer)
            if ep == args.stop_after_epoch:
                logger.info(f'stopping after epoch {ep} (--stop_after_epoch)')
                return
        logger.info(f'stage {stage} finished')


if __name__ == "__main__":
    main()




