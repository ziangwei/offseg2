# -*- coding: utf-8 -*-
"""OffSeg-CCM2: the decision as a fixed point of context-conditioned rescoring.

Generation 2 of the own-decoder line. Gen 1 read out at 46.8 (OffSeg-B 45.9,
PARSeg3 48.17), with acc_ccm_gain converging to 0.20 against a bound of 1.0
and the tail flat (128k 46.65 / 144k 46.57 / 160k 46.79).

What those two numbers say
--------------------------
  * gain 0.20 with the tanh bound never approached: the conditional metric IS
    used (a zero-initialised gate that the model pushed to 0.2 and held), and
    the OUTPUT BOUND is not what limits it. Raising ccm_gain_scale would do
    nothing. The predicted failure mode -- context degenerating to a single
    class vector late in training, driving gain to 0 -- did not happen.
  * a flat tail rather than LCR's still-rising endpoint: the mechanism found
    what it can express and stopped. Two candidate limits remain, capacity and
    the fact that gen 1 rescores exactly ONCE.

Gen 2 addresses both, and the second one is the interesting half.

The idea
--------
Gen 1 is internally inconsistent: it claims the metric is determined by the
competition a pixel faces, then rescores under that metric ONCE -- even though
rescoring CHANGES the competition. If the claim is right, one step cannot be
the end of it. The decision should be the fixed point of

    p*  =  F(p*),    F(p) = softmax( < M(p) f , e > )

i.e. "re-decide under the metric implied by your current competition", applied
until it stops moving. Concretely, with WEIGHTS SHARED across steps:

    p^0 = softmax(OffSeg logits)
    for t in 0..T-1:
        z^t     = sum_c p^t_c e_c            (nucleus-truncated)
        gain^t  = g([z^t ; f])
        M^t     = I + U diag(gain^t) V
        p^(t+1) = softmax( < M^t f , e > )

Properties that matter here:
  * ZERO new parameters. The same metric generator is applied T times, so the
    head stays at ~0.19M (rank 192) against the ~2.7M PARSeg3 branch it
    replaces. T is not a capacity knob.
  * T=1 reproduces generation 1 EXACTLY, so the ablation over T is free and
    the previous 46.8 is a point on that curve. T is reported as a curve, not
    chosen as a constant.
  * The feature f is never rewritten -- only the metric is re-derived. The
    evidence is fixed; what changes is how it is measured. That is what the
    claim actually says, and it also removes any drift/explosion risk from
    iterating.
  * No second decision path and no fusion module, so the overlap with PARSeg3
    does not move.

Gradient truncation. The context is detached at every step, so the loss
reaches the parameters only through the final application (a 1-step / phantom
gradient, as used for deep equilibrium models). Earlier steps act as a free,
non-parametric refinement of the CONDITIONING VARIABLE: by step T the posterior
has sharpened, so "who am I actually competing with" is estimated better than
from OffSeg's raw posterior. Set ccm2_detach_steps=False for full BPTT through
the chain, at the usual stability cost.

Also in gen 2: rank 64 -> 192 (capacity was the other candidate limit; it is
bundled because slots are expensive, and it is separable after the fact since
the T curve is measured and the T=1/rank=64 point is already on file at 46.8).

Read-out vs gen 1's 46.8 and OffSeg-B 45.9. Kill: 96k-128k clearly below the
gen-1 curve. Needles: acc_ccm_gain (mean|gain| at the last step; ~0.2 = same
regime as gen 1) and acc_ccm_move (mean total-variation between p^0 and p^T --
0 means the iteration is a no-op and T buys nothing).
"""
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from ..utils import resize
from .OffSegCCM import OffSegCCM


@MODELS.register_module()
class OffSegCCM2(OffSegCCM):
    """OffSeg whose decision is a fixed point of context-conditioned rescoring.

    Args (on top of OffSegCCM):
        ccm2_steps (int): T, number of rescoring steps. T=1 == generation 1.
        ccm2_detach_steps (bool): detach the context between steps (default
            True: 1-step gradient, stable). False = full BPTT.
        ccm2_step_w (float): optional CE on every intermediate step's logits
            (default 0.0 = off; the final step alone is supervised).
    """

    def __init__(self, in_channels, new_channels, num_classes,
                 ccm2_steps=3, ccm2_detach_steps=True, ccm2_step_w=0.0,
                 **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.ccm2_steps = max(1, int(ccm2_steps))
        self.ccm2_detach_steps = bool(ccm2_detach_steps)
        self.ccm2_step_w = float(ccm2_step_w)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        new_inputs = new_inputs[::-1]
        lowres_feat = new_inputs[0]
        for hires_feat, freqfusion in zip(new_inputs[1:], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat,
                                                    lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)

        feat_aligned = self.align(lowres_feat)
        masks, e, f, (h, w) = self._offset_learning_parts(feat_aligned)
        b, k, _ = masks.shape
        ctx_e = e.detach() if self.ccm_detach_context else e

        # ---- fixed-point iteration; weights shared, f never rewritten ----
        logits = masks
        step_logits = []
        for t in range(self.ccm2_steps):
            ctx = logits.detach() if (self.ccm2_detach_steps
                                      or self.ccm_detach_context) else logits
            f_m, gain = self.ccm(f, ctx_e, ctx)
            logits = self.offset_learning.mask_norm(
                f_m @ e.transpose(1, 2)).permute(0, 2, 1).contiguous()
            step_logits.append(logits)

        with torch.no_grad():
            p0 = torch.softmax(masks.transpose(1, 2), dim=-1)
            pT = torch.softmax(logits.transpose(1, 2), dim=-1)
            move = 0.5 * (p0 - pT).abs().sum(-1).mean()

        out = dict(
            stage1_logits=masks.view(b, k, h, w),
            final_logits=logits.view(b, k, h, w),
            ccm_gain=gain,
            ccm_move=move,
        )
        if self.ccm2_step_w > 0 and self.ccm2_steps > 1:
            out['step_logits'] = [s.view(b, k, h, w) for s in step_logits[:-1]]
        return out

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_ccm_move'] = seg_logits['ccm_move'].detach()

        steps = seg_logits.get('step_logits', None)
        if steps:
            seg_label = self._stack_batch_gt(batch_data_samples)
            if seg_label.dim() == 4:
                seg_label = seg_label.squeeze(1)
            size = seg_label.shape[-2:]
            acc = 0.0
            for s in steps:
                s = resize(input=s, size=size, mode='bilinear',
                           align_corners=self.align_corners)
                acc = acc + self.loss_decode(
                    s, seg_label, ignore_index=self.ignore_index)
            losses['loss_ccm_steps'] = acc / len(steps) * self.ccm2_step_w
        return losses
