"""Measures the real inference cost of each preset: C_base, C_extra, parameters, latency.

C_base is one interval with a_t = 0, and C_base + C_extra is one interval with a_t = 1,
both through the actual inference_step, so new modules are counted (section 5.7).

  python tools/profile_cost.py --presets PCM3
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crvio.config import apply_preset, build_parser  # noqa: E402
from crvio.vio import build_model  # noqa: E402


def attach_mac_counter(model):
    count = {'macs': 0}

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            count['macs'] += m.weight.numel() * out.numel() // (out.shape[0] * out.shape[1]) * out.shape[0]
        elif isinstance(m, nn.Linear):
            count['macs'] += m.weight.numel() * out.numel() // out.shape[-1]
        elif isinstance(m, nn.LSTM):
            # Four gates of input + hidden weights per layer and per step.
            T = inp[0].shape[1]
            B = inp[0].shape[0]
            count['macs'] += sum(p.numel() for n, p in m.named_parameters() if 'weight' in n) * T * B
    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.LSTM))]
    return count, handles


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--presets', nargs='+', default=['PCM3'])
    cli.add_argument('--gpu', type=int, default=0)
    cli.add_argument('--repeats', type=int, default=200)
    args = cli.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    print(f'{"":5}{"params(M)":>11}{"infer(M)":>10}{"C_base(MMAC)":>14}{"C_full(MMAC)":>14}{"c_extra":>9}{"t_base(ms)":>12}{"t_full(ms)":>12}')
    for name in args.presets:
        opt = apply_preset(build_parser().parse_args(['--preset', name]))
        model = build_model(opt).to(device).eval()
        total = sum(p.numel() for p in model.parameters())

        img = torch.zeros(1, 3, opt.img_h, opt.img_w, device=device)
        imu = torch.zeros(1, 11, 7, device=device)
        state = model.init_state(1, device)
        zero, one = torch.zeros(1, 1, device=device), torch.ones(1, 1, device=device)

        count, handles = attach_mac_counter(model)
        with_skip = opt.use_policy
        macs = {}
        for key, force in (('base', zero), ('full', one)):
            if key == 'base' and not with_skip:
                macs[key] = 0       # without a policy: full vision only
                continue
            count['macs'] = 0
            model.inference_step(img, img, imu, state, force_visual=force)
            macs[key] = count['macs']
        for h in handles:
            h.remove()

        def latency(force):
            with torch.no_grad():
                for _ in range(20):
                    model.inference_step(img, img, imu, state, force_visual=force)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                for _ in range(args.repeats):
                    model.inference_step(img, img, imu, state, force_visual=force)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
            return (time.perf_counter() - t0) / args.repeats * 1e3

        t_base = latency(zero) if with_skip else float('nan')
        t_full = latency(one)
        c_extra = (macs['full'] - macs['base']) / macs['full'] if with_skip else float('nan')
        print(f'{name:5}{total / 1e6:11.2f}{total / 1e6:10.2f}{macs["base"] / 1e6:14.1f}'
              f'{macs["full"] / 1e6:14.1f}{c_extra:9.3f}{t_base:12.2f}{t_full:12.2f}')


if __name__ == '__main__':
    main()
