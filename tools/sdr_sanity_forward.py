# -*- coding: utf-8 -*-
"""Sanity check for PARSegSDR. No dataset, no GPU, no carafe needed (~30 s).

Checks:
  1. train forward from a synthetic `feat_aligned` produces the teacher
     branch outputs and all SDR losses are finite; backward runs.
  2. eval forward has NO teacher keys (no GT leak into inference).
  3. INFERENCE EQUIVALENCE: with identical weights, PARSegSDR's student
     path reproduces PARSeg3's final_logits exactly (max abs diff ~ 0)
     and the state_dicts have identical keys (checkpoint compatible).

Run from repo root:  python tools/sdr_sanity_forward.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from mmseg.models.decode_heads.PARSeg3 import PARSeg3
from mmseg.models.decode_heads.PARSegSDR import PARSegSDR
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


def build(cls, seed=0):
    torch.manual_seed(seed)
    # new_channels must be the real ones: FreqFusion's GroupNorm has
    # divisibility constraints checked at CONSTRUCTION time (the trunk is
    # never called in this test, but it must build).
    return cls(
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
            sdr_teacherw=1.0, sdr_kdw=0.5, sdr_rivalw=0.2,
            sdr_absentw=0.1, sdr_margin=0.5, sdr_purity=0.75,
            sdr_warmup_iters=10,
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

    # GT at 4x logit resolution: BLOCKY regions (so the purity mask has
    # clean interior cells, like real segmentation maps), only 6 of the 13
    # classes present (so the absent-suppression branch has work to do),
    # plus an ignore border (exercises validity masks).
    coarse = torch.randint(0, 6, (B, 1, 8, 8)).float()
    gt = torch.nn.functional.interpolate(coarse, size=(Hf * 4, Wf * 4),
                                         mode='nearest').long()
    gt[:, :, :8, :] = 255

    sdr = build(PARSegSDR, seed=0)

    # ---- 1. training forward + all losses + backward ----
    sdr.train()
    sdr._sdr_gt = gt
    out = sdr._forward_from_aligned(feat_aligned)
    for k in ('sdr_teacher_logits', 'sdr_y_feat', 'sdr_valid_feat',
              'sdr_present_feat'):
        assert k in out, f'missing teacher key {k}'
    assert out['sdr_teacher_logits'].shape == (B, 13, Hf, Wf)

    samples = []
    for b in range(B):
        s = SegDataSample()
        s.gt_sem_seg = PixelData(data=gt[b])
        samples.append(s)
    sdr._sdr_step = 100  # force ramp=1.0 so kd/margin branches are exercised
    losses = sdr.loss_by_feat(out, samples)
    sdr._sdr_gt = None

    expect = ['loss_base', 'loss_refinement', 'loss_fusion',
              'loss_refinement_focus', 'loss_intra_div',
              'loss_sdr_teacher', 'loss_sdr_kd', 'loss_sdr_rival',
              'loss_sdr_absent']
    for k in expect:
        assert k in losses, f'missing loss {k}'
        v = losses[k]
        assert torch.isfinite(v).all(), f'{k} not finite: {v}'
        print(f'  {k:26s} = {float(v):.4f}')
    total = sum(v for v in losses.values())
    total.backward()
    grad_norm = sum(p.grad.abs().sum() for p in sdr.parameters()
                    if p.grad is not None)
    assert torch.isfinite(grad_norm), 'non-finite grads'
    print(f'  backward OK, sum|grad| = {float(grad_norm):.2f}')

    # ---- 2. eval forward: no teacher keys, no GT needed ----
    sdr.eval()
    with torch.no_grad():
        out_eval = sdr._forward_from_aligned(feat_aligned)
    assert 'sdr_teacher_logits' not in out_eval, 'teacher leaked into eval'
    print('  eval forward has no teacher branch: OK')

    # ---- 3. inference equivalence vs PARSeg3 + ckpt compatibility ----
    p3 = build(PARSeg3, seed=1)
    k_sdr = set(sdr.state_dict().keys())
    k_p3 = set(p3.state_dict().keys())
    assert k_sdr == k_p3, (
        f'state_dict mismatch: {k_sdr ^ k_p3}')
    p3.load_state_dict(sdr.state_dict())
    p3.eval()
    # PARSeg3 has no _forward_from_aligned, so run its post-align tail
    # inline (identical to PARSeg3.forward after `align`).
    with torch.no_grad():
        base = p3.offset_learning(feat_aligned)
        ref, _ = p3.prototype_attribute_refinement(feat_aligned, base)
        final_p3 = p3.fusion(base, ref)
    diff = (out_eval['final_logits'] - final_p3).abs().max()
    print('  max final_logits diff (SDR eval vs PARSeg3): %.3e' % float(diff))
    assert diff < 1e-5, 'student path is NOT equivalent to PARSeg3!'
    print('ALL SDR SANITY CHECKS PASSED')


if __name__ == '__main__':
    main()
