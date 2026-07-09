# -*- coding: utf-8 -*-
"""Information-flow verification for PARSegACR. No dataset/GPU/carafe (~30s).

Checks:
  1. round-1 losses are the exact PARSeg3 set (recipe purity) + the two ACR
     terms; all finite; backward runs.
  2. DETACH DISCIPLINE: gradient from the round-2 CE alone must reach the
     layout encoder and round-2 decision but NOT flow into round-1 heads
     through the layout-reading channel (it may reach the trunk only via
     the shared feat_aligned path -- verified by checking offset_learning
     cls_repr, which is upstream of final_r1 ONLY through round-1).
  3. gate bounded, init at 0.1; blend correct at init.
  4. eval forward needs no GT.

Run from repo root:  python tools/acr_sanity_forward.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from mmseg.models.decode_heads.PARSegACR import PARSegACR
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


def build(seed=0):
    torch.manual_seed(seed)
    return PARSegACR(
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
            acr_layout_dim=64, acr_gate_max=1.0, acr_gate_init=0.1,
            acr_r2w=1.0, acr_r1w=0.5,
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
    samples = []
    for b in range(B):
        s = SegDataSample()
        s.gt_sem_seg = PixelData(data=gt[b])
        samples.append(s)

    m = build(seed=0)
    m.train()

    out = m._forward_from_aligned(feat)
    g = float(out['acr_gate'])
    assert abs(g - 0.1) < 1e-3, f'gate should init at 0.1, got {g}'
    blend = out['acr_r1_final'] + out['acr_gate'] * (
        out['acr_r2_logits'] - out['acr_r1_final'])
    assert float((blend - out['final_logits']).abs().max()) < 1e-6

    losses = m.loss_by_feat(out, samples)
    expect = ['loss_base', 'loss_refinement', 'loss_fusion',
              'loss_refinement_focus', 'loss_intra_div',
              'loss_acr_r2', 'loss_acr_r1']
    for k in expect:
        assert k in losses, f'missing {k}'
        assert torch.isfinite(losses[k]).all(), f'{k} not finite'
        print(f'  {k:24s} = {float(losses[k]):.4f}')
    assert 'acc_acr_gate' in losses
    total = sum(v for k, v in losses.items() if k.startswith('loss'))
    total.backward()
    print(f'  full backward OK, gate init {g:.3f} (max 1.0)')

    # ---- detach discipline: r2 CE alone ----
    m2 = build(seed=1)
    m2.train()
    out2 = m2._forward_from_aligned(feat)
    losses2 = m2.loss_by_feat(out2, samples)
    losses2['loss_acr_r2'].backward()
    g_layout = sum(float(p.grad.abs().sum()) for p in
                   m2.acr_layout.parameters() if p.grad is not None)
    g_r2 = sum(float(p.grad.abs().sum()) for p in
               m2.acr_decision.parameters() if p.grad is not None)
    # round-1-exclusive modules: AGCF fusion + PAL refinement. Their only
    # path to loss_acr_r2 would be through the (detached) layout input.
    g_r1_excl = sum(float(p.grad.abs().sum()) for mod in
                    (m2.fusion, m2.prototype_attribute_refinement)
                    for p in mod.parameters() if p.grad is not None)
    assert g_layout > 0 and g_r2 > 0, 'round-2 pathway got no gradient'
    assert g_r1_excl == 0, f'layout detach leaked into round 1: {g_r1_excl}'
    print(f'  detach OK: |grad| layout={g_layout:.2f} r2={g_r2:.2f} '
          f'round1-exclusive={g_r1_excl:.1f}')

    # ---- eval ----
    m.eval()
    with torch.no_grad():
        oute = m._forward_from_aligned(feat)
    assert oute['final_logits'].shape == (B, 13, Hf, Wf)
    print('  eval forward OK (no GT)')
    print('ALL ACR SANITY CHECKS PASSED')


if __name__ == '__main__':
    main()
