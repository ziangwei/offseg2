#!/usr/bin/env python3
"""Time one decode head's forward+backward in isolation.

Usage:  python tools/bench_head.py <config> [<config> ...]
Builds only the decode head, feeds it random features of the shape the ADE
config produces, and reports ms/iteration.  Use it to find which head is slow
without waiting for a training job to start.
"""
import argparse
import time

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmseg.registry import MODELS


def bench(cfg_path, iters=30, warmup=10, batch=4, size=128, device='cuda'):
    cfg = Config.fromfile(cfg_path)
    init_default_scope('mmseg')
    for mod in (cfg.get('custom_imports', {}) or {}).get('imports', []):
        __import__(mod)
    head = MODELS.build(cfg.model.decode_head).to(device).train()

    channels = cfg.model.decode_head.get('in_channels', [32, 64, 144, 288])
    strides = [4, 8, 16, 32]
    inputs = [torch.randn(batch, c, 512 // s, 512 // s, device=device,
                          requires_grad=True)
              for c, s in zip(channels, strides)]

    def step():
        out = head(inputs)
        loss = (out['final_logits'] if isinstance(out, dict) else out).float()
        loss.square().mean().backward()
        head.zero_grad(set_to_none=True)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iters * 1000
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    params = sum(p.numel() for p in head.parameters())
    print('%-62s %8.1f ms/iter   peak %7.0f MB   %9d params'
          % (cfg_path.split('/')[-1][:62], elapsed, peak, params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--iters', type=int, default=30)
    args = parser.parse_args()
    for cfg in args.configs:
        torch.cuda.reset_peak_memory_stats()
        bench(cfg, iters=args.iters)
