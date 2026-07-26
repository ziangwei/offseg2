# -*- coding: utf-8 -*-
"""Standalone sanity checks for PARSeg-HRA2 (no dataset, no checkpoint, 1 GPU/CPU).

Checks, in order:
  1. identity at init -- the field must leave the fused logits untouched at
     step 0 (magnitude biased onto the zero bin), so the model IS PARSeg3.
  2. shapes / field stride -- the field comes out at image_size/field_stride.
  3. relocation target -- on a synthetic label map with a straight boundary,
     the target direction must point AWAY from the boundary into the pixel's
     own class, and the target magnitude must be the shortest one that lands
     in the interior.
  4. losses finite and gradients reaching the field head.

Usage:
  python tools/hra2_sanity_forward.py            # CPU is fine
  python tools/hra2_sanity_forward.py --device cuda:0
"""
import argparse
import math

import torch
import torch.nn.functional as F

from mmseg.models.decode_heads.PARSegHRA2 import (
    _HRA2Field, _HRA2ImageStem, _warp)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--field-stride', type=int, default=2)
    return p.parse_args()


def _ok(name, cond, extra=''):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")
    return bool(cond)


def main():
    args = parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)

    B, C, H, W = 2, 64, 32, 32                 # stride-4 grid
    NCLS = 150
    img_h, img_w = H * 4, W * 4
    fs = args.field_stride
    up = 4 // fs
    out_hw = (img_h // fs, img_w // fs)
    mag_px = [0.0, 4.0, 8.0, 16.0]
    mag_field = [m / fs for m in mag_px]

    stem = _HRA2ImageStem(64).to(dev).eval()
    field = _HRA2Field(feat_dim=C, guide_dim=64, hidden=64, up=up,
                       num_dir=16, mag_centers=mag_field).to(dev).eval()

    image = torch.randn(B, 3, img_h, img_w, device=dev)
    feat = torch.randn(B, C, H, W, device=dev)
    logits = torch.randn(B, NCLS, H, W, device=dev)

    all_ok = True
    print('0) bias layout (PixelShuffle reads channel = c_out*up^2 + sub)')
    with torch.no_grad():
        bias = field.hra2_mlp[-1].bias.view(field.per_pos, up * up)
    zero_bin = bias[field.num_dir]
    gate_row = bias[-1]
    dir_rows = bias[:field.num_dir]
    all_ok &= _ok('zero-magnitude bin biased on every sub-position',
                  bool((zero_bin > 1.0).all()),
                  f'{[round(v, 2) for v in zero_bin.tolist()]}')
    all_ok &= _ok('gate biased on every sub-position',
                  bool((gate_row < -1.0).all()),
                  f'{[round(v, 2) for v in gate_row.tolist()]}')
    all_ok &= _ok('direction bins left at zero (uniform = no boundary)',
                  float(dir_rows.abs().max()) < 1e-6,
                  f'max|b|={float(dir_rows.abs().max()):.1e}')

    print('1) identity at init')
    with torch.no_grad():
        guide = stem(image)
        flow, gate, dlog, mlog, rho = field(feat, guide, out_hw)
        up_logits = F.interpolate(logits, size=out_hw, mode='bilinear',
                                  align_corners=False)
        out = _warp(up_logits, flow, gate)
    # With mag_bias=8 and 3 non-zero bins: p_nonzero = 3/(3+e^8) = 1.0e-3, so
    # rho ~ 1.0e-3 * mean(non-zero centers) ~ 5e-3 field px. Thresholds keep
    # ~4x headroom rather than being tightened onto the expected value.
    all_ok &= _ok('rho ~ 0', rho.abs().max().item() < 2e-2,
                  f'max|rho|={rho.abs().max().item():.2e} field px '
                  f'(expect ~5e-3)')
    all_ok &= _ok('warp == logits', (out - up_logits).abs().max().item() < 5e-3,
                  f'max|d|={(out - up_logits).abs().max().item():.2e}')
    all_ok &= _ok('gate ~ 0.12', abs(gate.mean().item() - 0.119) < 0.02,
                  f'mean={gate.mean().item():.3f}')

    print('2) shapes')
    all_ok &= _ok('field stride', tuple(flow.shape) == (B, 2, *out_hw),
                  f'flow={tuple(flow.shape)} expected={(B, 2, *out_hw)}')
    all_ok &= _ok('dir bins', dlog.shape[1] == 16, f'{dlog.shape[1]}')
    all_ok &= _ok('mag bins', mlog.shape[1] == 4, f'{mlog.shape[1]}')

    print('3) relocation target on a synthetic boundary')
    # class 0 on the left, class 1 on the right, vertical boundary at x=Xb.
    fh, fw = out_hw
    Xb = fw // 2
    lab = torch.zeros(1, fh, fw, dtype=torch.long, device=dev)
    lab[:, :, Xb:] = 1

    class _Stub:
        pass
    stub = _Stub()
    stub.ignore_index = 255
    stub.band = 2
    stub.num_dir = 16
    stub.hra2_field = field
    from mmseg.models.decode_heads.PARSegHRA2 import PARSegHRA2
    tgt = PARSegHRA2._relocation_target(stub, lab)
    dir_idx, mag_idx, has_target, band, interior = tgt

    # a band pixel just LEFT of the boundary belongs to class 0 -> must be sent
    # further left (angle ~ pi); one just right of it -> right (angle ~ 0).
    y = fh // 2
    left_x, right_x = Xb - 1, Xb
    a_left = float(dir_idx[0, y, left_x]) * 2 * math.pi / 16
    a_right = float(dir_idx[0, y, right_x]) * 2 * math.pi / 16
    all_ok &= _ok('band covers the boundary', bool(band[0, y, left_x]),
                  f'band={bool(band[0, y, left_x])}')
    all_ok &= _ok('left pixel sent left', abs(math.cos(a_left) + 1) < 0.35,
                  f'cos={math.cos(a_left):+.2f} (want -1)')
    all_ok &= _ok('right pixel sent right', abs(math.cos(a_right) - 1) < 0.35,
                  f'cos={math.cos(a_right):+.2f} (want +1)')
    all_ok &= _ok('target exists on the band',
                  float(has_target[band].float().mean()) > 0.9,
                  f'{100 * float(has_target[band].float().mean()):.0f}% of band')
    mg = mag_idx[band & has_target]
    all_ok &= _ok('shortest magnitude picked', int(mg.min()) == 1,
                  f'min bin={int(mg.min())} (centers {mag_px} image px)')
    all_ok &= _ok('interior is not the band',
                  not bool((band & interior).any()))

    print('4) gradients')
    field.train()
    guide = stem(image)
    flow, gate, dlog, mlog, rho = field(feat, guide, out_hw)
    up_logits = F.interpolate(logits, size=out_hw, mode='bilinear',
                              align_corners=False)
    out = _warp(up_logits, flow, gate)
    loss = out.square().mean() + dlog.square().mean() + mlog.square().mean()
    loss.backward()
    head_grad = field.hra2_mlp[-1].weight.grad
    all_ok &= _ok('loss finite', torch.isfinite(loss).item(),
                  f'{loss.item():.4f}')
    all_ok &= _ok('field head has grad',
                  head_grad is not None and float(head_grad.abs().sum()) > 0,
                  f'|g|={float(head_grad.abs().sum()):.3e}')

    print('=' * 60)
    print('ALL PASS' if all_ok else 'SOMETHING FAILED -- do not launch')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
