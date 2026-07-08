# -*- coding: utf-8 -*-
"""Information-flow verification for the three text-semantic models
(LDR / TDL / TAM). No dataset, no GPU, no carafe (~1 min).

Per model: forward shapes, loss finiteness, backward reaching the text
pathway, and start-state discipline (identity or near-identity at init).

Run from repo root:  python tools/text3_sanity_forward.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F

from mmseg.models.decode_heads.PARSeg3 import PARSeg3
from mmseg.models.decode_heads.PARSegLDR import PARSegLDR
from mmseg.models.decode_heads.PARSegTDL import PARSegTDL
from mmseg.models.decode_heads.PARSegTAM import PARSegTAM
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData

NC, K = 13, 6


def make_fake_descs(path):
    torch.manual_seed(7)
    emb = F.normalize(torch.randn(NC, K, 512), dim=-1)
    torch.save({'embeddings': emb}, path)
    return path


def build(cls, desc_path, seed=0):
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
        ldr_desc_path=desc_path, tdl_desc_path=desc_path,
        tam_desc_path=desc_path,
        tdl_attn_dim=64, tdl_gate_max=0.5, tdl_gate_init=0.05,
        tam_scale=0.5,
    )
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
    coarse = torch.randint(0, 6, (B, 1, 8, 8)).float()
    gt = torch.nn.functional.interpolate(coarse, size=(Hf * 4, Wf * 4),
                                         mode='nearest').long()
    gt[:, :, :8, :] = 255
    fake = make_fake_descs(os.path.join(tempfile.mkdtemp(), 'fake_desc.pt'))
    samples = samples_from(gt)

    # ================= LDR =================
    m = build(PARSegLDR, fake, seed=0)
    m.train()
    out = m._forward_from_aligned(feat)
    d0 = (out['base_head_logits'] - out['raw_base_head_logits']).abs().max()
    assert float(d0) == 0.0, 'LDR must start as plain PARSeg3 (delta==0)'
    losses = m.loss_by_feat(out, samples)
    for k in ('loss_lcr_aux', 'loss_lcr_rank'):
        assert k in losses and torch.isfinite(losses[k]).all(), k
    sum(losses.values()).backward()
    # zero-init scorer blocks upstream grads at exact init (v1-identical
    # warm-start coupling); nudge and verify the desc pathway
    m.zero_grad()
    with torch.no_grad():
        m.lcr.score_mlp[-1].weight.normal_(0, 0.01)
    rel = m.lcr(feat, m.offset_learning(feat))
    rel['delta_logits'].abs().sum().backward()
    g = m.lcr.desc_proj.weight.grad
    assert g is not None and float(g.abs().sum()) > 0, 'desc pathway dead'
    print('  LDR: identity at init, aux/rank losses finite, description '
          'evidence pathway flows: OK')

    # ================= TDL =================
    m2 = build(PARSegTDL, fake, seed=1)
    m2.train()
    lk = m2.prototype_attribute_refinement.tdl_lookup
    g0 = float(lk.gate())
    assert abs(g0 - 0.05) < 1e-3, f'gate should init at 0.05, got {g0}'
    base2 = m2.offset_learning(feat)
    ref2, cal2 = m2.prototype_attribute_refinement(feat, base2)
    out2 = dict(base_head_logits=base2, refinement_head_logits=ref2,
                calibrated_attr_tokens=cal2,
                final_logits=m2.fusion(base2, ref2))
    losses2 = m2.loss_by_feat(out2, samples)
    for k, v in losses2.items():
        assert torch.isfinite(v).all(), f'TDL {k} not finite'
    sum(losses2.values()).backward()
    gq = sum(float(p.grad.abs().sum()) for p in
             (lk.q_proj.weight, lk.k_proj.weight, lk.v_proj.weight))
    ga = float(lk.gate_alpha.grad.abs())
    assert gq > 0 and ga >= 0, 'dictionary pathway got no gradient'
    print(f'  TDL: gate init {g0:.3f} (max 0.5), losses finite, dictionary '
          f'q/k/v grads flow ({gq:.2f}): OK')

    # ================= TAM =================
    m3 = build(PARSegTAM, fake, seed=2)
    h3 = m3.prototype_attribute_refinement
    w = h3._tam_weights()
    assert float((w - 1.0).abs().max()) == 0.0, 'w must be exactly 1 at init'
    # shared-weight identity vs plain PARSeg3
    p3 = build(PARSeg3, fake, seed=3)
    p3_keys = set(p3.state_dict().keys())
    m3_keys = set(m3.state_dict().keys())
    extra = m3_keys - p3_keys
    assert p3_keys <= m3_keys and all('tam_' in k for k in extra), extra
    p3.load_state_dict({k: v for k, v in m3.state_dict().items()
                        if k in p3_keys})
    m3.eval(); p3.eval()
    with torch.no_grad():
        b_a = m3.offset_learning(feat)
        r_a, _ = m3.prototype_attribute_refinement(feat, b_a)
        b_b = p3.offset_learning(feat)
        r_b, _ = p3.prototype_attribute_refinement(feat, b_b)
    diff = (r_a - r_b).abs().max()
    assert float(diff) < 1e-6, f'TAM init is not identity: {float(diff)}'
    m3.train()
    base3 = m3.offset_learning(feat)
    ref3, cal3 = m3.prototype_attribute_refinement(feat, base3)
    out3 = dict(base_head_logits=base3, refinement_head_logits=ref3,
                calibrated_attr_tokens=cal3,
                final_logits=m3.fusion(base3, ref3))
    losses3 = m3.loss_by_feat(out3, samples)
    for k, v in losses3.items():
        assert torch.isfinite(v).all(), f'TAM {k} not finite'
    sum(losses3.values()).backward()
    gp = float(h3.tam_proj.weight.grad.abs().sum())
    gr = float(h3.tam_residual.grad.abs().sum())
    assert gp > 0 and gr > 0, 'metric pathway got no gradient'
    print(f'  TAM: w==1 at init, refinement bit-identical to PARSeg3, '
          f'metric grads flow (proj {gp:.2f}, residual {gr:.2f}): OK')

    print('ALL TEXT-3 SANITY CHECKS PASSED')


if __name__ == '__main__':
    main()
