# -*- coding: utf-8 -*-
"""Sanity check for PARSegLCR2. No dataset, no GPU, no carafe needed (~30 s).

Checks:
  1. IDENTITY AT INIT: score_mlp's last layer is zero-init, so
     base_head_logits == raw_base_head_logits at initialization
     (the model starts as plain PARSeg3 training).
  2. gate map is per-pixel, bounded in (0, gate_max), and starts uniform
     at gate_init.
  3. train forward + all losses (incl. lcr aux/rank) finite; backward runs;
     gate conv and window scorer receive gradients.
  4. eval forward runs without GT.

Run from repo root:  python tools/lcr2_sanity_forward.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from mmseg.models.decode_heads.PARSegLCR2 import PARSegLCR2
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


def build(seed=0):
    torch.manual_seed(seed)
    return PARSegLCR2(
        in_channels=[8, 16, 32, 64],
        new_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout_ratio=0.1,
        num_classes=13,
        cls_attributes=4,
        args=dict(
            basew=2.0, refinementw=1.5, fusionw=1.0,
            intra_div=0.1, tau=0.07, proto_topk_div=64,
            proto_residual_scale=1.0, refinement_focusw=0.75,
            fusion_mode='AGCF', use_class_prototypes=True,
            lcr_topk=5, lcr_dim=64, lcr_hidden=128,
            lcr_gate_max=0.35, lcr_gate_init=0.05,
            lcr_auxw=0.20, lcr_rankw=0.20,
            lcr_rank_margin=0.20, lcr_rank_hard_weight=2.0,
            lcr2_win_small=5, lcr2_win_large=13, lcr2_gate_hidden=16,
        ),
        norm_cfg=dict(type='GN', num_groups=4, requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0),
    )


def main():
    torch.manual_seed(42)
    B, Hf, Wf = 2, 32, 32
    feat_aligned = torch.randn(B, 64, Hf, Wf)

    # blocky GT with an ignore border, 6 of 13 classes present
    coarse = torch.randint(0, 6, (B, 1, 8, 8)).float()
    gt = torch.nn.functional.interpolate(coarse, size=(Hf * 4, Wf * 4),
                                         mode='nearest').long()
    gt[:, :, :8, :] = 255

    m = build(seed=0)
    n_new = sum(p.numel() for p in m.lcr.parameters()) + \
        sum(p.numel() for p in m.lcr_gate.parameters())
    print(f'  lcr2 module params: {n_new/1e3:.1f}k')

    # ---- 1+2: identity at init, gate bounded/uniform ----
    m.train()
    out = m._forward_from_aligned(feat_aligned)
    diff = (out['base_head_logits'] - out['raw_base_head_logits']).abs().max()
    print(f'  identity at init: max |base - raw| = {float(diff):.3e}')
    assert diff < 1e-6, 'model does not start as plain PARSeg3!'
    g = out['lcr_gate_map']
    assert g.shape == (B, 1, Hf, Wf), f'gate map shape {tuple(g.shape)}'
    assert 0.0 < float(g.min()) and float(g.max()) < 0.35, 'gate out of bounds'
    assert abs(float(g.mean()) - 0.05) < 1e-3, \
        f'gate should start uniform at 0.05, got {float(g.mean()):.4f}'
    print(f'  gate map OK: shape {tuple(g.shape)}, '
          f'init mean {float(g.mean()):.4f}, max bound 0.35')

    # ---- 3: losses + backward ----
    samples = []
    for b in range(B):
        s = SegDataSample()
        s.gt_sem_seg = PixelData(data=gt[b])
        samples.append(s)
    losses = m.loss_by_feat(out, samples)
    expect = ['loss_base', 'loss_refinement', 'loss_fusion',
              'loss_refinement_focus', 'loss_intra_div',
              'loss_lcr_aux', 'loss_lcr_rank']
    for k in expect:
        assert k in losses, f'missing loss {k}'
        assert torch.isfinite(losses[k]).all(), f'{k} not finite'
        print(f'  {k:24s} = {float(losses[k]):.4f}')
    total = sum(v for v in losses.values())
    total.backward()
    g1 = sum(p.grad.abs().sum() for p in m.lcr.score_mlp.parameters())
    g2 = sum(p.grad.abs().sum() for p in m.lcr_gate.parameters()
             if p.grad is not None)
    assert torch.isfinite(g1) and float(g1) > 0, 'scorer got no gradient'
    print(f'  backward OK: |grad| scorer={float(g1):.3f} gate={float(g2):.3f}')

    # ---- 4: eval forward ----
    m.eval()
    with torch.no_grad():
        out_eval = m._forward_from_aligned(feat_aligned)
    assert out_eval['final_logits'].shape == (B, 13, Hf, Wf)
    print('  eval forward OK')
    print('ALL LCR2 SANITY CHECKS PASSED')


if __name__ == '__main__':
    main()
