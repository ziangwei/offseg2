# -*- coding: utf-8 -*-
"""Standalone sanity checks for OffSeg-CCM (no dataset, no checkpoint, CPU ok).

  1. identity at init -- gain == 0 so M == I and stage-2 logits are EXACTLY
     the OffSeg masks. The model must BE OffSeg at step 0.
  2. stage-1 mirror -- our mirrored Offset_Learning forward must reproduce the
     upstream module bit-for-bit (it is mirrored, not modified, so a silent
     divergence here would poison everything).
  3. context behaves as claimed -- peaked posterior => z ~ that class's own
     vector (metric degenerates to per-class, the TAM regime); flat posterior
     => z ~ mixture of the rivals. This is the mechanism's core assumption and
     also its main failure mode.
  4. gradients + parameter budget vs the PARSeg3 branch it replaces.

Usage:
  python tools/offsegccm_sanity_forward.py
  python tools/offsegccm_sanity_forward.py --device cuda:0 --rank 64
"""
import argparse

import torch

from mmseg.models.decode_heads.OffSegCCM import ContextConditionedMetric
from mmseg.models.decode_heads.offset_learning import Offset_Learning


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--rank', type=int, default=64)
    p.add_argument('--hidden', type=int, default=128)
    return p.parse_args()


def _ok(name, cond, extra=''):
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")
    return bool(cond)


def _mirror(ol, x):
    """Same mirror as OffSegCCM._offset_learning_parts."""
    b, c, h, w = x.shape
    cls_repr = ol.cls_repr.expand(b, -1, -1)
    img_feat = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
    coupled = (img_feat @ cls_repr.transpose(1, 2)).permute(0, 2, 1)
    aligned_cls = cls_repr + ol.cls_offset_proj(coupled.softmax(dim=2) @ img_feat)
    aligned_feat = img_feat + ol.feat_offset_proj(
        coupled.softmax(dim=1).transpose(1, 2) @ cls_repr)
    masks = ol.mask_norm(aligned_feat @ aligned_cls.transpose(1, 2))
    return masks.permute(0, 2, 1).contiguous(), aligned_cls, aligned_feat


def main():
    args = parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)

    B, C, H, W, K = 2, 256, 16, 16, 150
    ol = Offset_Learning(num_classes=K, embed_dims=C).to(dev).eval()
    ccm = ContextConditionedMetric(embed_dims=C, rank=args.rank,
                                   hidden=args.hidden).to(dev).eval()
    x = torch.randn(B, C, H, W, device=dev)
    all_ok = True

    print('1) stage-1 mirror reproduces upstream Offset_Learning')
    with torch.no_grad():
        ref = ol(x)                                   # [B,K,H,W]
        masks, e, f = _mirror(ol, x)
        mine = masks.view(B, K, H, W)
    d = (ref - mine).abs().max().item()
    all_ok &= _ok('mirror == upstream', d < 1e-4, f'max|d|={d:.2e}')

    print('2) identity at init')
    with torch.no_grad():
        f_m, gain = ccm(f, e, masks)
        logits2 = ol.mask_norm(f_m @ e.transpose(1, 2)).permute(0, 2, 1)
    all_ok &= _ok('gain == 0', gain.abs().max().item() == 0.0,
                  f'max|gain|={gain.abs().max().item():.1e}')
    all_ok &= _ok('M == I (feat unchanged)',
                  (f_m - f).abs().max().item() < 1e-6,
                  f'max|d|={(f_m - f).abs().max().item():.1e}')
    dd = (logits2 - masks).abs().max().item()
    all_ok &= _ok('stage-2 logits == OffSeg masks', dd < 1e-4, f'max|d|={dd:.2e}')

    print('3) context behaves as claimed')
    with torch.no_grad():
        # peaked posterior -> z should be (almost) class 7's own vector
        peaked = torch.full((B, K, H * W), -20.0, device=dev)
        peaked[:, 7] = 20.0
        p = torch.softmax(peaked.transpose(1, 2), -1)
        keep = ccm._nucleus(p); p = p * keep; p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)
        z_peak = torch.bmm(p, e)
        cos_peak = torch.nn.functional.cosine_similarity(
            z_peak, e[:, 7:8].expand_as(z_peak), dim=-1).mean().item()

        # two-way tie between 3 and 9 -> z should sit between the two
        tie = torch.full((B, K, H * W), -20.0, device=dev)
        tie[:, 3] = 5.0; tie[:, 9] = 5.0
        p2 = torch.softmax(tie.transpose(1, 2), -1)
        keep2 = ccm._nucleus(p2); p2 = p2 * keep2
        p2 = p2 / p2.sum(-1, keepdim=True).clamp_min(1e-6)
        z_tie = torch.bmm(p2, e)
        mid = 0.5 * (e[:, 3:4] + e[:, 9:10]).expand_as(z_tie)
        cos_tie = torch.nn.functional.cosine_similarity(z_tie, mid, dim=-1).mean().item()
        n_kept = keep2.sum(-1).float().mean().item()
    all_ok &= _ok('peaked -> z ~ own class vector', cos_peak > 0.99,
                  f'cos={cos_peak:.4f}')
    all_ok &= _ok('tie -> z ~ midpoint of the two rivals', cos_tie > 0.99,
                  f'cos={cos_tie:.4f}')
    all_ok &= _ok('nucleus keeps only the rivals', 1.5 < n_kept < 3.5,
                  f'kept={n_kept:.2f} of {K}')

    print('4) gradients and parameter budget')
    ccm.train()
    f_m, gain = ccm(f, e, masks)
    loss = (ol.mask_norm(f_m @ e.transpose(1, 2))).square().mean()
    loss.backward()
    g_head = ccm.ccm_g[-1].weight.grad
    all_ok &= _ok('gain generator has grad',
                  g_head is not None and float(g_head.abs().sum()) > 0,
                  f'|g|={float(g_head.abs().sum()):.3e}')
    all_ok &= _ok('loss finite', bool(torch.isfinite(loss)), f'{loss.item():.4f}')

    n_ccm = sum(p.numel() for p in ccm.parameters())
    n_ol = sum(p.numel() for p in ol.parameters())
    # PARSeg3 branch this replaces: 2 x 1800 x 256 queries + FFN 256<->2048
    # + 4 attn projections + input_proj/FC/FC/attr/feat_proj + proto/route/AGCF
    n_parseg3 = (2 * 1800 * 256 + 2 * 256 * 2048 + 4 * 256 * 256
                 + 5 * 256 * 256 + 256 * 256 + 768 * 64 + 256 * 64 + 8000)
    print(f"    CCM head        : {n_ccm/1e6:.3f} M  (rank {args.rank})")
    print(f"    OffSeg offset_l : {n_ol/1e6:.3f} M  (inherited)")
    print(f"    PARSeg3 branch  : {n_parseg3/1e6:.3f} M  (REPLACED, estimate)")
    print(f"    ratio           : {n_parseg3/max(n_ccm,1):.1f}x smaller")

    print('=' * 62)
    print('ALL PASS' if all_ok else 'SOMETHING FAILED -- do not launch')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
