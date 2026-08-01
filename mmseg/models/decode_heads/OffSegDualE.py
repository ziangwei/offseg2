# -*- coding: utf-8 -*-
"""OffSeg-Dual-E: evidence-level decorrelation. Path B reads the PRE-FUSION
scene feature instead of the fused one.

Motivated by the wall that emerged after Dual-NF read out: three independent
mechanisms land within 0.2 of each other (CCM 46.80 / ccm2t1 46.88 / Dual-NF
46.69) -- "OffSeg + one extra mechanism" saturates at ~46.8 regardless of the
mechanism. All arms so far share ONE trait: both paths read the same
feat_aligned, so their errors inherit a common floor of correlation no matter
what the decision side does. The decorrelation principle has five levers --
pressure (focus loss), parameterization (OL), attention scope (M), metric
(C), and EVIDENCE SOURCE. This arm is the fifth, the only one untested, and
the prime suspect for the wall.

Single variable vs offsegdual: where B's tokens come from.

    Dual   B tokens = avgpool4(feat_aligned)      fused, stride 16
    THIS   B tokens = pre[3] output               pre-fusion, stride 32,
                                                  256 ch, 16x16 at 512 crop

A reads the fused fine evidence; B reads the scene context BEFORE FreqFusion
mixes detail into it. The two alignments now differ from the evidence level
up, not merely in how they decide. B's class vectors are built from coarse
evidence and still matched densely against the fused feature for output (the
class side is what changes, mirroring the OL arm's division of labor).

Pre-registered read-out: if evidence decorrelation is real, acc_dual_disagree
comes out clearly ABOVE Dual's and mIoU >= Dual; if stride-32 evidence is too
coarse to carry a path, alpha stays low and mIoU ~ Dual-NF, closing this axis
too. Kill: 96k-128k clearly below the ccm2t1 curve (46.88).

Everything else -- gate not detached (SAF), error-focused CE on B (LTM),
losses, init -- inherited from OffSegDual unchanged.
"""
import torch

from mmseg.registry import MODELS
from .OffSegDual import OffSegDual, _DualQueryPath


@MODELS.register_module()
class OffSegDualE(OffSegDual):
    """OffSegDual whose path B tokens come from the pre-fusion stride-32
    stage."""

    def __init__(self, in_channels, new_channels, num_classes, **kwargs):
        kwargs.pop('dual_pool', None)          # tokens are already stride 32
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, dual_pool=1, **kwargs)
        assert new_channels[-1] == self.channels, (
            'OffSegDualE feeds pre[3] output straight into path B and '
            f'requires new_channels[-1] == channels, got {new_channels[-1]} '
            f'vs {self.channels}.')

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        new_inputs = new_inputs[::-1]

        scene = new_inputs[0]                  # pre-fusion stride-32, 256 ch

        lowres_feat = new_inputs[0]
        for hires_feat, freqfusion in zip(new_inputs[1:], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat,
                                                    lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
        feat = self.align(lowres_feat)

        logits_a = self.offset_learning(feat)

        # Path B: class vectors from PRE-FUSION scene evidence, matched
        # densely against the fused feature for output.
        b_, c_, hs, ws = scene.shape
        tok = scene.flatten(2).transpose(1, 2)                    # [B, N, C]
        tok = self.dual_path.norm_kv(tok) + self.dual_path._pos(
            hs, ws, c_, tok.device, tok.dtype)
        q = self.dual_path.norm_q(self.dual_path.query.weight)[None].expand(
            b_, -1, -1)
        e, _ = self.dual_path.attn(q, tok, tok, need_weights=False)
        e = q + e
        e = e + self.dual_path.ffn(e)
        logits_b = torch.einsum('bchw,bkc->bkhw', feat, e)
        logits_b = self.dual_path.mask_norm(
            logits_b.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        alpha = self.dual_gate(logits_a, logits_b)
        final = logits_a + alpha * (logits_b - logits_a)
        return dict(a_logits=logits_a, b_logits=logits_b,
                    final_logits=final, dual_alpha=alpha)
