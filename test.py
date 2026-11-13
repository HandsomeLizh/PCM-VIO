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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='/nfs/turbo/coe-hunseok/mingyuy/KITTI_odometry', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')

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

parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--experiment_name', type=str, default='test', help='experiment name')
parser.add_argument('--model', type=str, default='./pretrain_models/vf_512_if_256_3e-05.model', help='state dict or a train.py stage*_last.pth')
parser.add_argument('--selection', type=str, default='policy', choices=['policy', 'all'], help='all = visual encoder at every step')

args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def main():

    # Create Dir
    experiment_dir = Path('./results')
    experiment_dir.mkdir_p()
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir_p()
    result_dir = file_dir.joinpath('files/')
    result_dir.mkdir_p()
    
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

    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck)
    print('load model %s'%args.model)
        
    # Feed model to GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids = gpu_ids)
    model.eval()

    if args.selection == 'all':   # p = 1 makes the 'random' rule pick the visual encoder at every step
        errors = tester.eval(model, 'random', num_gpu=len(gpu_ids), p=1.0)
    else:
        errors = tester.eval(model, 'gumbel-softmax', num_gpu=len(gpu_ids))
    tester.generate_plots(result_dir, 30)
    tester.save_text(result_dir)
    
    for i, seq in enumerate(args.val_seq):
        message = f"Seq: {seq}, t_rel: {tester.errors[i]['t_rel']:.4f}, r_rel: {tester.errors[i]['r_rel']:.4f}, "
        message += f"t_rmse: {tester.errors[i]['t_rmse']:.4f}, r_rmse: {tester.errors[i]['r_rmse']:.4f}, "
        message += f"usage: {tester.errors[i]['usage']:.4f}"
        print(message)
    mean = {k: np.mean([e[k] for e in tester.errors]) for k in ('t_rel', 'r_rel', 't_rmse', 'r_rmse', 'usage')}
    print('mean: ' + ' '.join(f'{k}={v:.4f}' for k, v in mean.items()))
    
    

if __name__ == "__main__":
    main()




