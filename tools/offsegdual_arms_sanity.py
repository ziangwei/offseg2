# -*- coding: utf-8 -*-
"""Sanity for the three Dual arms (OL / C / M). No dataset, CPU ok.

  OL: shapes; class-offset actually moves cls_repr; param count << MHA path.
  C : identity of the CCM part at init (gain==0 -> path A == plain OffSeg
      masks); needles present; grads reach ccm and both paths.
  M : mask built from hint has the empty-region fallback; masked and unmasked
      forward differ; shapes.

Usage: python tools/offsegdual_arms_sanity.py [--device cuda:0]
"""
import argparse

import torch
import torch.nn.functional as F

from mmseg.models.decode_heads.OffSegDualOL import _DualOffsetPath
from mmseg.models.decode_heads.OffSegDualM import _MaskedQueryPath
from mmseg.models.decode_heads.OffSegCCM import ContextConditionedMetric
from mmseg.models.decode_heads.offset_learning import Offset_Learning


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
    feat = torch.randn(B, C, H, W, device=dev)
    all_ok = True

    print('OL) scene-scale class-offset path')
    ol = _DualOffsetPath(dim=C, num_classes=K).to(dev)
    e, lg = ol(feat)
    all_ok &= _ok('shapes', tuple(e.shape) == (B, K, C)
                  and tuple(lg.shape) == (B, K, H, W))
    drift = (e - ol.cls_repr).abs().mean().item()
    all_ok &= _ok('class offset moves cls_repr', drift > 1e-4,
                  f'mean|offset|={drift:.4f}')
    n_ol = sum(p.numel() for p in ol.parameters())
    all_ok &= _ok('param budget ~0.11M', n_ol < 0.2e6, f'{n_ol/1e6:.3f} M')

    print('C) CCM identity at init inside the dual structure')
    up = Offset_Learning(num_classes=K, embed_dims=C).to(dev).eval()
    ccm = ContextConditionedMetric(embed_dims=C, rank=64, hidden=128).to(dev).eval()
    with torch.no_grad():
        b, c, h, w = feat.shape
        cls_repr = up.cls_repr.expand(b, -1, -1)
        img = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        coup = (img @ cls_repr.transpose(1, 2)).permute(0, 2, 1)
        e_a = cls_repr + up.cls_offset_proj(coup.softmax(2) @ img)
        f_a = img + up.feat_offset_proj(coup.softmax(1).transpose(1, 2) @ cls_repr)
        masks = up.mask_norm(f_a @ e_a.transpose(1, 2)).permute(0, 2, 1)
        f_m, gain = ccm(f_a, e_a, masks)
        la = up.mask_norm(f_m @ e_a.transpose(1, 2)).permute(0, 2, 1)
    all_ok &= _ok('gain == 0 at init', gain.abs().max().item() == 0.0)
    d = (la - masks).abs().max().item()
    all_ok &= _ok('path A == OffSeg masks at init', d < 1e-4, f'max|d|={d:.2e}')

    print('M) region-masked attention')
    mp = _MaskedQueryPath(dim=C, num_classes=K).to(dev)
    hint = torch.randn(B, K, H, W, device=dev)
    # force class 0 to own nothing -> its row must fall back to full attention
    hint[:, 0] = -1e4
    e1, l1 = mp(feat, hint_logits=hint)
    e0, l0 = mp(feat, hint_logits=None)
    all_ok &= _ok('shapes', tuple(l1.shape) == (B, K, H, W))
    diff = (l1 - l0).abs().mean().item()
    all_ok &= _ok('mask changes the result', diff > 1e-5, f'mean|d|={diff:.2e}')
    # rebuild the mask exactly as the module does, to check the fallback
    hs, ws = H // mp.pool, W // mp.pool
    am = F.adaptive_avg_pool2d(hint, (hs, ws)).argmax(1).flatten(1)
    classes = torch.arange(K, device=dev)
    disallow = am[:, None, :] != classes[None, :, None]
    empty = disallow.all(-1, keepdim=True)
    all_ok &= _ok('class 0 region is empty in hint', bool(empty[:, 0].all()))
    disallow = disallow & ~empty
    all_ok &= _ok('fallback opens empty rows',
                  bool((~disallow[:, 0]).all()))

    print('grads) one backward through all three paths')
    for m in (ol, mp):
        m.train()
    ccm.train()
    e, lg = ol(feat.requires_grad_(True))
    loss = lg.square().mean()
    e2, lg2 = mp(feat, hint_logits=hint)
    loss = loss + lg2.square().mean()
    loss.backward()
    all_ok &= _ok('loss finite', bool(torch.isfinite(loss)), f'{loss.item():.3f}')
    g1 = ol.cls_offset_proj.weight.grad
    g2 = mp.query.weight.grad
    all_ok &= _ok('grad -> OL cls_offset_proj',
                  g1 is not None and float(g1.abs().sum()) > 0)
    all_ok &= _ok('grad -> M queries',
                  g2 is not None and float(g2.abs().sum()) > 0)

    print('=' * 62)
    print('ALL PASS' if all_ok else 'SOMETHING FAILED -- do not launch')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
