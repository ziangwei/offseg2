# -*- coding: utf-8 -*-
"""Information-flow verification for PARSegACT and TextAnchoredAuxHead
(PARSeg-TAX). No dataset/GPU/carafe (~40 s).

Run from repo root:  python tools/act_tax_sanity_forward.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from mmseg.models.decode_heads.PARSegACT import PARSegACT
from mmseg.models.decode_heads.PARSegTAX import TextAnchoredAuxHead
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

NC, K = 13, 6


def make_fake_descs(path):
    torch.manual_seed(7)
    emb = F.normalize(torch.randn(NC, K, 512), dim=-1)
    torch.save({'embeddings': emb}, path)
    return path


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
    fake = make_fake_descs(os.path.join(tempfile.mkdtemp(), 'fake_desc.pt'))

    # ================= ACT =================
    torch.manual_seed(0)
    m = PARSegACT(
        in_channels=[8, 16, 32, 64], new_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3], channels=64, dropout_ratio=0.1,
        num_classes=NC, cls_attributes=4,
        args=dict(
            basew=2.0, refinementw=1.5, fusionw=1.0, intra_div=0.1,
            tau=0.07, proto_topk_div=64, proto_residual_scale=1.0,
            refinement_focusw=0.75, fusion_mode='AGCF',
            use_class_prototypes=True,
            acr_layout_dim=64, acr_gate_max=1.0, acr_gate_init=0.1,
            acr_r2w=1.0, acr_r1w=0.5, act_desc_path=fake),
        norm_cfg=dict(type='GN', num_groups=4, requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=1.0))
    m.train()
    assert 'acr_layout.desc_mean' in m.state_dict(), 'text buffer missing'
    r0 = m.acr_layout.embed_residual.abs().max()
    assert float(r0) == 0.0, 'layout residual must start at zero'
    out = m._forward_from_aligned(feat)
    losses = m.loss_by_feat(out, samples)
    for k in ('loss_base', 'loss_refinement', 'loss_fusion',
              'loss_refinement_focus', 'loss_intra_div',
              'loss_acr_r2', 'loss_acr_r1'):
        assert k in losses and torch.isfinite(losses[k]).all(), k
    total = sum(v for k, v in losses.items() if k.startswith('loss'))
    total.backward()
    gt_proj = float(m.acr_layout.text_proj.weight.grad.abs().sum())
    gt_res = float(m.acr_layout.embed_residual.grad.abs().sum())
    assert gt_proj > 0 and gt_res > 0, 'text layout pathway got no gradient'
    print(f'  ACT: text-tied layout mixing, residual zero-init, all losses '
          f'finite, text grads flow (proj {gt_proj:.2f}, res {gt_res:.2f}): OK')
    m.eval()
    with torch.no_grad():
        oute = m._forward_from_aligned(feat)
    assert oute['final_logits'].shape == (B, NC, Hf, Wf)
    print('  ACT: eval forward OK')

    # ================= TAX =================
    torch.manual_seed(1)
    aux = TextAnchoredAuxHead(
        desc_path=fake, tau=0.1, num_convs=1,
        in_channels=32, in_index=2, channels=64, dropout_ratio=0.1,
        num_classes=NC, input_transform=None,
        norm_cfg=dict(type='GN', num_groups=4, requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                         loss_weight=0.4))
    aux.train()
    assert 'desc_mean' in aux.state_dict()
    assert float(aux.cls_residual.abs().max()) == 0.0
    stage3 = [torch.randn(B, 8, 128, 128), torch.randn(B, 16, 64, 64),
              torch.randn(B, 32, 32, 32), torch.randn(B, 64, 16, 16)]
    logits = aux(stage3)
    assert logits.shape == (B, NC, 32, 32)
    # cosine/tau bounds: |logit| <= 1/tau
    assert float(logits.abs().max()) <= 1.0 / 0.1 + 1e-4
    loss = aux.loss_by_feat(logits, samples)
    key = [k for k in loss if k.startswith('loss')][0]
    assert torch.isfinite(loss[key]).all()
    loss[key].backward()
    gp = float(aux.text_proj.weight.grad.abs().sum())
    gr = float(aux.cls_residual.grad.abs().sum())
    gc = sum(float(p.grad.abs().sum()) for p in aux.convs.parameters()
             if p.grad is not None)
    assert gp > 0 and gr > 0 and gc > 0, 'aux pathways got no gradient'
    print(f'  TAX: cosine aux head on stage-3, logits bounded by 1/tau, '
          f'CE={float(loss[key]):.4f}, grads flow (proj {gp:.2f}, '
          f'res {gr:.2f}, convs {gc:.2f}): OK')
    print('  TAX: discarded at inference by framework design '
          '(auxiliary_head), deployed model = pure decode head')

    print('ALL ACT/TAX SANITY CHECKS PASSED')


if __name__ == '__main__':
    main()
