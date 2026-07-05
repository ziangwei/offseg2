# -*- coding: utf-8 -*-
"""Sanity checks for the three text-anchor models (LTA / LTC / PTA).

No dataset, no GPU, no carafe needed (~1 min). Uses a synthetic 13-class
anchor asset for the tiny models, plus a shape/normalization check of the
real 150-class asset if present.

Run from repo root:  python tools/text_anchor_sanity.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from mmseg.models.decode_heads.PARSegLTA import PARSegLTA, load_text_anchors
from mmseg.models.decode_heads.PARSegLTC import PARSegLTC
from mmseg.models.decode_heads.PARSegPTA import PARSegPTA
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

NC = 13


def make_fake_anchors(path, n=NC, e=512):
    torch.manual_seed(7)
    emb = F.normalize(torch.randn(n, e), dim=-1)
    torch.save({'embeddings': emb, 'class_names': [f'c{i}' for i in range(n)],
                'templates': ['fake'], 'model': 'fake'}, path)
    return path


def build(cls, anchor_path, seed=0, extra=None):
    torch.manual_seed(seed)
    args = dict(
        basew=2.0, refinementw=1.5, fusionw=1.0,
        intra_div=0.1, tau=0.07, proto_topk_div=64,
        proto_residual_scale=1.0, refinement_focusw=0.75,
        fusion_mode='AGCF', use_class_prototypes=True,
        lcr_topk=5, lcr_dim=64, lcr_hidden=128,
        lcr_gate_max=0.35, lcr_gate_init=0.05,
        lcr_auxw=0.20, lcr_rankw=0.20,
        lcr_rank_margin=0.20, lcr_rank_hard_weight=2.0,
        lta_anchor_path=anchor_path, lta_res_scale=0.1,
        pta_anchor_path=anchor_path, pta_res_scale=0.1,
        ltc_infoncew=0.15, ltc_tau=0.1, ltc_warmup_iters=10, ltc_min_pixels=1,
    )
    args.update(extra or {})
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


def blocky_gt(B, H, W):
    coarse = torch.randint(0, 6, (B, 1, 8, 8)).float()
    gt = torch.nn.functional.interpolate(
        coarse, size=(H, W), mode='nearest').long()
    gt[:, :, :8, :] = 255
    return gt


def samples_from(gt):
    out = []
    for b in range(gt.shape[0]):
        s = SegDataSample()
        s.gt_sem_seg = PixelData(data=gt[b])
        out.append(s)
    return out


def main():
    torch.manual_seed(42)
    B, Hf, Wf = 2, 32, 32
    feat = torch.randn(B, 64, Hf, Wf)
    gt = blocky_gt(B, Hf * 4, Wf * 4)
    tmp = tempfile.mkdtemp()
    fake = make_fake_anchors(os.path.join(tmp, 'fake_anchors.pt'))

    # ---------------- LTA ----------------
    m = build(PARSegLTA, fake, seed=0)
    assert 'lcr.class_embed.anchors' in m.state_dict(), 'anchors not in ckpt'
    r = m.lcr.class_embed
    d = (r.matrix() - r.proj(r.anchors)).abs().max()
    assert float(d) == 0.0, 'residual must start at zero'
    m.train()
    base = m.offset_learning(feat)
    rel = m.lcr(feat, base)
    assert rel['delta_logits'].shape == (B, NC, Hf, Wf)
    g = m._lcr_gate()
    bh = base + g * rel['delta_logits']
    ref, cal = m.prototype_attribute_refinement(feat, bh)
    out = dict(raw_base_head_logits=base, base_head_logits=bh,
               calibrated_attr_tokens=cal, refinement_head_logits=ref,
               final_logits=m.fusion(bh, ref),
               lcr_relation_logits=base.detach() + g * rel['delta_logits'],
               lcr_candidate_idx=rel['candidate_idx'])
    losses = m.loss_by_feat(out, samples_from(gt))
    for k, v in losses.items():
        assert torch.isfinite(v).all(), f'LTA {k} not finite'
    sum(losses.values()).backward()
    # NOTE: at exact init the scorer's last layer is zero, which blocks
    # gradient to everything upstream (same benign warm-start coupling as
    # v1's class_embed / LCR2's gate). Verify the PATHWAY by nudging the
    # last layer off zero and re-running.
    m.zero_grad()
    with torch.no_grad():
        m.lcr.score_mlp[-1].weight.normal_(0, 0.01)
    base_n = m.offset_learning(feat)
    rel_n = m.lcr(feat, base_n)
    (rel_n['delta_logits'].abs().sum()).backward()
    gproj = m.lcr.class_embed.proj.weight.grad
    assert gproj is not None and torch.isfinite(gproj).all() \
        and float(gproj.abs().sum()) > 0, 'anchor projection pathway dead'
    print('  LTA: anchors in ckpt, zero-init residual, losses finite, '
          'proj pathway flows once scorer unblocks: OK')

    # ---------------- LTC ----------------
    m2 = build(PARSegLTC, fake, seed=1)
    m2.train()
    m2._ltc_step = 100  # ramp = 1
    out2 = m2._forward_from_aligned(feat)
    assert 'ltc_feat_aligned' in out2
    losses2 = m2.loss_by_feat(out2, samples_from(gt))
    assert 'loss_ltc_infonce' in losses2, 'InfoNCE loss missing'
    v = losses2['loss_ltc_infonce']
    assert torch.isfinite(v) and float(v) > 0, f'InfoNCE degenerate: {v}'
    sum(losses2.values()).backward()
    print(f'  LTC: InfoNCE = {float(v):.4f} (>0, finite), backward OK')
    m2.eval()
    with torch.no_grad():
        oute = m2._forward_from_aligned(feat)
    assert oute['final_logits'].shape == (B, NC, Hf, Wf)
    print('  LTC: eval forward OK (no GT, no text model anywhere)')

    # ---------------- PTA ----------------
    m3 = build(PARSegPTA, fake, seed=2)
    assert 'offset_learning.anchors' in m3.state_dict()
    cr = m3.offset_learning._cls_repr()
    res0 = (cr - m3.offset_learning.anchor_proj(
        m3.offset_learning.anchors).unsqueeze(0)).abs().max()
    assert float(res0) == 0.0, 'PTA residual must start at zero'
    std = float(cr.std())
    assert 0.005 < std < 0.08, f'cls_repr init scale off: std={std:.4f}'
    m3.train()
    base3 = m3.offset_learning(feat)
    ref3, cal3 = m3.prototype_attribute_refinement(feat, base3)
    out3 = dict(base_head_logits=base3, refinement_head_logits=ref3,
                calibrated_attr_tokens=cal3,
                final_logits=m3.fusion(base3, ref3))
    losses3 = m3.loss_by_feat(out3, samples_from(gt))
    for k, v in losses3.items():
        assert torch.isfinite(v).all(), f'PTA {k} not finite'
    sum(losses3.values()).backward()
    ga = m3.offset_learning.anchor_proj.weight.grad
    assert ga is not None and float(ga.abs().sum()) > 0
    print(f'  PTA: anchors in ckpt, zero residual, cls_repr std={std:.4f} '
          '(~0.02 target), losses finite, proj grad flows: OK')

    # ---------------- real asset (if generated) ----------------
    real = os.path.join(os.path.dirname(__file__), '..',
                        'assets', 'text_anchors', 'ade20k_clip_vitb32.pt')
    if os.path.exists(real):
        emb, meta = load_text_anchors(real, 150)
        norms = emb.norm(dim=-1)
        assert emb.shape == (150, 512)
        assert torch.allclose(norms, torch.ones(150), atol=1e-4)
        print("  real asset OK: %s, normalized, model=%s"
              % (tuple(emb.shape), meta.get('model')))
    else:
        print('  real asset not found (generate with tools/gen_text_anchors.py)')

    print('ALL TEXT-ANCHOR SANITY CHECKS PASSED')


if __name__ == '__main__':
    main()
