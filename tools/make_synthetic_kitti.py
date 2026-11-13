"""Builds a tiny KITTI-shaped tree to check the full pipeline without the real dataset.
Results on it say nothing about accuracy.

  python tools/make_synthetic_kitti.py /path/to/out
"""
import os
import sys

import numpy as np
import scipy.io as sio
from PIL import Image

LENGTHS = {'00': 80, '01': 60, '09': 45, '05': 45}


def main(root):
    rng = np.random.default_rng(0)
    for seq, n in LENGTHS.items():
        img_dir = f'{root}/sequences/{seq}/image_2'
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(f'{root}/poses', exist_ok=True)
        os.makedirs(f'{root}/imus', exist_ok=True)
        texture = rng.integers(0, 255, (96, 512, 3), dtype=np.uint8)
        yaw, pos, lines = 0.0, np.zeros(3), []
        rates = rng.normal(0, 0.02, n)
        for i in range(n):
            yaw += rates[i]
            pos = pos + 3.0 * np.array([np.sin(yaw), 0.0, np.cos(yaw)])
            c, s = np.cos(yaw), np.sin(yaw)
            mat = np.array([[c, 0, s, pos[0]], [0, 1, 0, 0.0], [-s, 0, c, pos[2]]])
            lines.append(' '.join(f'{v:.9e}' for v in mat.flatten()))
            frame = np.roll(texture, -int(40 * yaw) - 3 * i, axis=1)[:, :192]
            Image.fromarray(frame).save(f'{img_dir}/{i:06d}.png')
        with open(f'{root}/poses/{seq}.txt', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        imu = rng.normal(0, 0.05, (10 * (n - 1) + 1, 6))
        imu[:, 2] += 9.81
        imu[:, 5] += np.repeat(rates, 10)[:imu.shape[0]] / 0.1
        sio.savemat(f'{root}/imus/{seq}.mat', {'imu_data_interp': imu})


if __name__ == '__main__':
    main(sys.argv[1])
