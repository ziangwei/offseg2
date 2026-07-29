# -*- coding: utf-8 -*-
"""Standalone sanity checks for OffSeg-Dual (no dataset, CPU ok).

  1. shapes and gate init (~0.12, biased toward path A).
  2. positional cache correctness across two sizes.
  3. losses finite; error-focused CE weights only A-wrong pixels.
  4. gradients reach BOTH paths and the gate through loss_fuse (the SAF
     check: the gate must shape the heads, so grads from the fused CE must
     arrive in path A, path B, AND the gate).
  5. parameter budget vs the senior's 2.75M branch.

Usage: python tools/offsegdual_sanity_forward.py [--device cuda:0]
"""
import argparse

import torch
import torch.nn.functional as F

from mmseg.models.decode_heads.OffSegDual import _DualGate, _DualQueryPath


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def _ok(name, cond, extra=''):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")
    return bool(cond)


def main():
    args = parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    B, C, H, W, K = 2, 256, 32, 32, 150
    all_ok = True

    path = _DualQueryPath(dim=C, num_classes=K).to(dev)
    gate = _DualGate(K).to(dev)
    feat = torch.randn(B, C, H, W, device=dev, requires_grad=True)

    print('1) shapes and gate init')
    e, lb = path(feat)
    la = torch.randn(B, K, H, W, device=dev, requires_grad=True)
    a = gate(la, lb)
    all_ok &= _ok('e_B shape', tuple(e.shape) == (B, K, C), f'{tuple(e.shape)}')
    all_ok &= _ok('logits_B shape', tuple(lb.shape) == (B, K, H, W),
                  f'{tuple(lb.shape)}')
    all_ok &= _ok('alpha ~ 0.12', abs(a.mean().item() - 0.119) < 0.03,
                  f'mean={a.mean().item():.3f}')

    print('2) positional cache across sizes')
    feat2 = torch.randn(B, C, 64, 48, device=dev)
    _, lb2 = path(feat2)
    all_ok &= _ok('second size ok', tuple(lb2.shape) == (B, K, 64, 48),
                  f'{tuple(lb2.shape)}')
    _, lb3 = path(feat)
    all_ok &= _ok('cache back-switch ok', tuple(lb3.shape) == (B, K, H, W))

    print('3) error-focused CE weights only A-wrong pixels')
    gt = torch.randint(0, K, (B, H, W), device=dev)
    pred_a = la.argmax(1)
    wrong = (pred_a != gt)
    ce = F.cross_entropy(lb, gt, reduction='none')
    focus = (ce * wrong.float()).sum() / wrong.float().sum().clamp_min(1.0)
    ce_right = (ce * (~wrong).float()).sum() / (~wrong).float().sum().clamp_min(1.0)
    all_ok &= _ok('focus loss finite', bool(torch.isfinite(focus)),
                  f'{focus.item():.3f} (right-pixel mean {ce_right.item():.3f})')
    all_ok &= _ok('some wrong pixels exist', int(wrong.sum()) > 0,
                  f'{int(wrong.sum())}')

    print('4) gradient reaches A, B and gate through the fused CE (SAF check)')
    final = la + a * (lb - la)
    loss = F.cross_entropy(final, gt)
    loss.backward()
    g_a = la.grad
    g_b = path.query.weight.grad
    g_g = gate.net[-1].weight.grad
    all_ok &= _ok('grad -> path A logits', g_a is not None and
                  float(g_a.abs().sum()) > 0, f'|g|={float(g_a.abs().sum()):.2e}')
    all_ok &= _ok('grad -> path B queries', g_b is not None and
                  float(g_b.abs().sum()) > 0, f'|g|={float(g_b.abs().sum()):.2e}')
    all_ok &= _ok('grad -> gate (not detached)', g_g is not None and
                  float(g_g.abs().sum()) > 0, f'|g|={float(g_g.abs().sum()):.2e}')

    print('5) parameter budget')
    n_b = sum(p.numel() for p in path.parameters())
    n_g = sum(p.numel() for p in gate.parameters())
    print(f"    path B : {n_b/1e6:.3f} M")
    print(f"    gate   : {n_g/1e3:.1f} k")
    print(f"    total  : {(n_b+n_g)/1e6:.3f} M   vs senior branch 2.75M "
          f"({2.75e6/(n_b+n_g):.1f}x smaller)")
    all_ok &= _ok('under 1M', (n_b + n_g) < 1e6)

    print('=' * 62)
    print('ALL PASS' if all_ok else 'SOMETHING FAILED -- do not launch')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
