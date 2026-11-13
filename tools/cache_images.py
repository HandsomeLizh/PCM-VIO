"""Resize KITTI frames once and store them as one uint8 array per sequence.

Training otherwise decodes 11 full-size PNGs per sample and resizes them on the CPU.
The cached frame is the training transform's result, ToTensor (-0.5) then tensor Resize,
rounded to uint8, so a loaded frame differs from the on-the-fly one by at most 0.5/255.

Usage: python tools/cache_images.py --data_dir data --out /root/VIO/cache/kitti_256x512
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.custom_transform import Resize, ToTensor  # noqa: E402

FORMAT = 'v1: uint8 round(255*(Resize(ToTensor(png)) + 0.5)), torchvision tensor resize'


def frame_paths(data_dir, seq):
    folder = os.path.join(data_dir, 'sequences', seq, 'image_2')
    # Same order as dataset/KITTI_dataset.py, which sorts the full paths.
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png'))


def process(args):
    paths, size = args
    torch.set_num_threads(1)
    imgs = [np.asarray(Image.open(p)) for p in paths]
    resized, _, _ = Resize(size)(*ToTensor()(imgs, None, None))      # the training transform
    q = torch.round((resized + 0.5) * 255).clamp_(0, 255).to(torch.uint8)
    return q.permute(0, 2, 3, 1).contiguous().numpy()                 # N, H, W, 3


def cache_sequence(data_dir, out_dir, seq, size, pool, chunk=32):
    final = os.path.join(out_dir, f'{seq}.npy')
    meta_path = os.path.join(out_dir, f'{seq}.json')
    paths = frame_paths(data_dir, seq)
    names = [os.path.basename(p) for p in paths]
    if os.path.exists(final) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta['files'] == names and meta['size'] == list(size) and meta['format'] == FORMAT:
            print(f'{seq}: up to date, {len(names)} frames')
            return
    tmp = final + '.tmp'
    arr = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(paths), size[0], size[1], 3))
    jobs = [(paths[i:i + chunk], size) for i in range(0, len(paths), chunk)]
    t0 = time.time()
    for k, block in enumerate(pool.imap(process, jobs)):
        arr[k * chunk:k * chunk + len(block)] = block
    arr.flush()
    del arr
    os.replace(tmp, final)
    with open(meta_path, 'w') as f:
        json.dump({'format': FORMAT, 'size': list(size), 'source': os.path.realpath(os.path.join(data_dir, 'sequences', seq)),
                   'files': names}, f)
    print(f'{seq}: {len(names)} frames in {time.time() - t0:.0f}s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--out', required=True)
    p.add_argument('--seqs', nargs='+', default=[f'{i:02d}' for i in range(11)])
    p.add_argument('--img_h', type=int, default=256)
    p.add_argument('--img_w', type=int, default=512)
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        for seq in a.seqs:
            cache_sequence(a.data_dir, a.out, seq, (a.img_h, a.img_w), pool)


if __name__ == '__main__':
    main()
