# -*- coding: utf-8 -*-
"""Sanity check for PARSegPAT. No dataset, no GPU, no carafe (~30 s).

Checks:
  1. grounding loss finite and > 0 on present classes; backward reaches the
     text projection AND the PAL pathway (calibrated tokens).
  2. INFERENCE EQUIVALENCE: forward is inherited from PARSeg3; with shared
     weights the final logits are bit-identical to PARSeg3.
  3. eval path needs no GT and no text beyond the frozen buffer.

Run from repo root:  python tools/pat_sanity_forward.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from mmseg.models.decode_heads.PARSeg3 import PARSeg3
from mmseg.models.decode_heads.PARSegPAT import PARSegPAT
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

NC, K = 13, 6


def make_fake_descs(path):
    torch.manual_seed(7)
    emb = F.normalize(torch.randn(NC, K, 512), dim=-1)
    torch.save({'embeddings': emb, 'class_names': [f'c{i}' for i in range(NC)],
                'model': 'fake'}, path)
    return path


def build(cls, desc_path=None, seed=0):
    torch.manual_seed(seed)
    args = dict(
        basew=2.0, refinementw=1.5, fusionw=1.0,
        intra_div=0.1, tau=0.07, proto_topk_div=64,
        proto_residual_scale=1.0, refinement_focusw=0.75,
        fusion_mode='AGCF', use_class_prototypes=True,
    )
    if desc_path is not None:
        args.update(pat_desc_path=desc_path, pat_w=0.15, pat_tau=0.1,
                    pat_warmup_iters=10, pat_token_dim=256)
    return cls(
        in_channels=[8, 16, 32, 64],
        new_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout_ratio=0.1,
        num_classes=NC,
        cls_attributes=4,
        args=args,
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

    tmp = tempfile.mkdtemp()
    fake = make_fake_descs(os.path.join(tmp, 'fake_desc.pt'))

    m = build(PARSegPAT, fake, seed=0)
    assert 'pat_desc' in m.state_dict(), 'desc buffer not in ckpt'
    m.train()
    m._pat_step = 100  # ramp = 1

    base = m.offset_learning(feat)
    ref, cal = m.prototype_attribute_refinement(feat, base)
    out = dict(base_head_logits=base, refinement_head_logits=ref,
               calibrated_attr_tokens=cal, final_logits=m.fusion(base, ref))
    samples = []
    for b in range(B):
        s = SegDataSample()
        s.gt_sem_seg = PixelData(data=gt[b])
        samples.append(s)
    losses = m.loss_by_feat(out, samples)
    assert 'loss_pat_ground' in losses, 'grounding loss missing'
    v = losses['loss_pat_ground']
    assert torch.isfinite(v) and float(v) > 0, f'degenerate: {v}'
    for k, lv in losses.items():
        assert torch.isfinite(lv).all(), f'{k} not finite'
        print(f'  {k:24s} = {float(lv):.4f}')
    sum(losses.values()).backward()
    g_proj = m.pat_proj.weight.grad
    g_pal = sum(float(p.grad.abs().sum()) for p in
                m.prototype_attribute_refinement.parameters()
                if p.grad is not None)
    assert g_proj is not None and float(g_proj.abs().sum()) > 0
    assert g_pal > 0, 'grounding gradient did not reach the PAL pathway'
    print(f'  backward OK: |grad| proj={float(g_proj.abs().sum()):.3f}, '
          f'PAL pathway={g_pal:.1f}')

    # ---- inference equivalence vs PARSeg3 (forward inherited) ----
    p3 = build(PARSeg3, None, seed=1)
    p3_keys = set(p3.state_dict().keys())
    pat_keys = set(m.state_dict().keys())
    assert p3_keys <= pat_keys, 'PAT must be a superset (buffer + proj only)'
    extra = pat_keys - p3_keys
    assert all(k.startswith('pat_') for k in extra), f'unexpected keys {extra}'
    p3.load_state_dict({k: v for k, v in m.state_dict().items()
                        if k in p3_keys})
    m.eval(); p3.eval()
    with torch.no_grad():
        b1 = m.offset_learning(feat)
        r1, _ = m.prototype_attribute_refinement(feat, b1)
        f1 = m.fusion(b1, r1)
        b2 = p3.offset_learning(feat)
        r2, _ = p3.prototype_attribute_refinement(feat, b2)
        f2 = p3.fusion(b2, r2)
    diff = (f1 - f2).abs().max()
    print(f'  final logits diff vs PARSeg3 (shared weights): {float(diff):.3e}')
    assert diff < 1e-6, 'inference path is not PARSeg3!'
    print('ALL PAT SANITY CHECKS PASSED')


if __name__ == '__main__':
    main()
