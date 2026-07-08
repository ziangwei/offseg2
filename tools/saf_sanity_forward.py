# -*- coding: utf-8 -*-
"""Sanity check for PARSegSAF. No dataset, no GPU, no carafe needed (~30 s).

Checks:
  1. alpha starts uniform at saf_alpha_init (blend begins AGCF-init-like).
  2. all losses (incl. loss_saf_bce on disagreement pixels) finite; backward.
  3. ISOLATION: the BCE meta-loss alone sends gradient to the arbiter ONLY
     -- zero gradient into either head or the trunk (detached inputs).
  4. eval forward needs no GT and produces the blended final logits.

Run from repo root:  python tools/saf_sanity_forward.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from mmseg.models.decode_heads.PARSegSAF import PARSegSAF
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


def build(seed=0):
    torch.manual_seed(seed)
    return PARSegSAF(
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
            saf_bcew=0.5, saf_warmup_iters=10,
            saf_hidden=32, saf_logit_ch=24, saf_alpha_init=0.12,
        ),
        norm_cfg=dict(type='GN', num_groups=4, requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0),
    )


def main():
    torch.manual_seed(42)
    B, Hf, Wf = 2, 32, 32
    feat = torch.randn(B, 64, Hf, Wf)
    coarse = torch.randint(0, 6, (B, 1, 8, 8)).float()
    gt = torch.nn.functional.interpolate(coarse, size=(Hf * 4, Wf * 4),
                                         mode='nearest').long()
    gt[:, :, :8, :] = 255

    m = build(seed=0)
    m.train()
    m._saf_step = 100  # ramp = 1

    out = m._forward_from_aligned(feat)
    a = torch.sigmoid(out['saf_alpha_logit'])
    assert a.shape == (B, 1, Hf, Wf)
    assert abs(float(a.mean()) - 0.12) < 5e-3, \
        f'alpha should start uniform at 0.12, got {float(a.mean()):.4f}'
    assert float(a.std()) < 1e-3, 'alpha should start (near) uniform'
    print(f'  alpha init OK: mean={float(a.mean()):.4f}, std={float(a.std()):.2e}')

    samples = []
    for b in range(B):
        s = SegDataSample()
        s.gt_sem_seg = PixelData(data=gt[b])
        samples.append(s)
    losses = m.loss_by_feat(out, samples)
    expect = ['loss_base', 'loss_refinement', 'loss_fusion',
              'loss_refinement_focus', 'loss_intra_div', 'loss_saf_bce']
    for k in expect:
        assert k in losses, f'missing {k}'
        assert torch.isfinite(losses[k]).all(), f'{k} not finite'
        print(f'  {k:24s} = {float(losses[k]):.4f}')
    assert 'acc_saf_alpha' in losses
    total = sum(v for k, v in losses.items() if k.startswith('loss'))
    total.backward()
    print('  full backward OK')

    # ---- isolation: BCE alone must not touch heads/trunk ----
    m2 = build(seed=1)
    m2.train()
    m2._saf_step = 100
    out2 = m2._forward_from_aligned(feat)
    losses2 = m2.loss_by_feat(out2, samples)
    losses2['loss_saf_bce'].backward()
    g_arb = sum(float(p.grad.abs().sum()) for p in m2.arbiter.parameters()
                if p.grad is not None)
    g_heads = sum(float(p.grad.abs().sum()) for mod in
                  (m2.offset_learning, m2.prototype_attribute_refinement)
                  for p in mod.parameters() if p.grad is not None)
    assert g_arb > 0, 'arbiter got no gradient from BCE'
    assert g_heads == 0, f'BCE leaked gradient into heads: {g_heads}'
    print(f'  isolation OK: |grad| arbiter={g_arb:.3f}, heads={g_heads:.1f}')

    # ---- eval: no GT anywhere ----
    m.eval()
    with torch.no_grad():
        oute = m._forward_from_aligned(feat)
    assert oute['final_logits'].shape == (B, 13, Hf, Wf)
    print('  eval forward OK (no GT, arbiter reads only the two heads)')
    print('ALL SAF SANITY CHECKS PASSED')


if __name__ == '__main__':
    main()
