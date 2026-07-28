# -*- coding: utf-8 -*-
"""OffSeg-CCM-S: scene composition as part of the conditioning context.

Third single-variable arm of the CCM family. The family has three axes --
capacity (rank), depth (fixed-point steps T), and the CONDITIONING VARIABLE.
The first two are being measured by offsegccm2t1 and offsegccm2; this is the
third, and it is the only one of the three that adds INFORMATION rather than
more capacity or more compute. Thirty prior experiments in this repo say the
system does not respond to more of the same; the few that moved it added a
channel that was not there before.

    gen 1  T=1 rank= 64  context = pixel          46.8
    A      T=3 rank= 64  context = pixel          (running)
    B      T=1 rank=192  context = pixel          (running)
    C      T=1 rank= 64  context = pixel + scene  (this file)

What is missing from the gen-1 context
--------------------------------------
`z` is built from a single pixel's own posterior, so it can only say "this
pixel is torn between wall and window". It cannot say "this is an outdoor
scene and window should not be competing here at all". But the error
decomposition on file is:

    ABSENT-FP     42.6%   the predicted class is not in the image at all
    PRESENT-CONF  57.4%   confident confusion among co-present classes

and LCR's autopsy attributes 78% of its gain to suppressing absent-class
intruders. That mass is invisible to a per-pixel context by construction.

So the context gets a second, image-level term, which is free:

    z_pix = sum_c p_c(x)          e_c        which rivals is THIS pixel facing
    z_img = sum_c mean_x[p_c(x)]  e_c        what scene is this
    gain  = g([z_pix ; z_img ; f])

Both terms are mixtures over the SAME class vectors, so the metric generator
sees the local competition and the global composition in one representation.
No presence prediction, no candidate set is pruned, nothing is masked out --
the scene only conditions the metric.

Honest note on a neighbouring dead axis
---------------------------------------
The experiment log marks active-class prediction dead: a learnable presence
predictor realised only +0.03 against a +10.22 oracle. That result is about
using a PREDICTED CLASS SET as a hard prior. Here nothing is pruned and no
presence decision is made; scene composition enters as a conditioning
variable for a metric, in the same way the per-pixel posterior already does.
LCR separately demonstrated that the absent-FP mass IS attackable (it took
78% of its gain there) as long as it is not attacked with explicit presence
prediction. The distinction is real but it is adjacent to a dead axis, and
that is a genuine risk of this arm.

Scope of "image": the pooled term is computed over whatever tensor the head
receives, which is a 512x512 crop at training time and a 512x512 slide window
at inference -- identical statistics on both sides, no train/test mismatch.

Cost: g's first layer widens from 2C to 3C, i.e. +33k params (0.11M -> 0.14M)
and under +0.5 GFLOP. T and rank are UNCHANGED from generation 1, so this arm
differs from the 46.8 control in exactly one variable.

Read-out vs gen 1's 46.8. Kill: 96k-128k clearly below the gen-1 curve.
Needle: acc_ccm_gain -- if it lands far above gen-1's 0.20, the scene term is
carrying real signal; if it collapses, the pooled term is noise to the
generator.
"""
import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .OffSegCCM import OffSegCCM, ContextConditionedMetric


class SceneContextConditionedMetric(ContextConditionedMetric):
    """Conditional metric whose context is local competition + scene make-up."""

    def __init__(self, embed_dims, rank=64, hidden=128, top_p=0.9,
                 gain_scale=1.0):
        super().__init__(embed_dims=embed_dims, rank=rank, hidden=hidden,
                         top_p=top_p, gain_scale=gain_scale)
        # [z_pix ; z_img ; f] instead of [z_pix ; f].
        self.ccm_g = nn.Sequential(
            nn.Linear(3 * embed_dims, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, rank, bias=True),
        )
        nn.init.zeros_(self.ccm_g[-1].weight)
        nn.init.zeros_(self.ccm_g[-1].bias)

    def forward(self, feat, cls_repr, masks_logits):
        """feat [B,HW,C]; cls_repr [B,K,C]; masks_logits [B,K,HW]."""
        p_raw = torch.softmax(masks_logits.transpose(1, 2), dim=-1)   # [B,HW,K]

        # local competition: nucleus-truncated, as in the pixel-only version
        keep = self._nucleus(p_raw)
        p = p_raw * keep
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-6)
        z_pix = torch.bmm(p, cls_repr)                                # [B,HW,C]

        # scene composition: the crop's mean posterior over classes. NOT
        # truncated -- a class that is weakly present everywhere is exactly the
        # signal we want here, and truncating per pixel first would erase it.
        p_img = p_raw.mean(dim=1, keepdim=True)                       # [B,1,K]
        z_img = torch.bmm(p_img, cls_repr).expand(-1, feat.shape[1], -1)

        gain = self.ccm_g(torch.cat([z_pix, z_img, feat], dim=-1))
        gain = self.gain_scale * torch.tanh(gain)
        feat_m = feat + self.ccm_u(gain * self.ccm_v(feat))
        return feat_m, gain


@MODELS.register_module()
class OffSegCCMS(OffSegCCM):
    """OffSegCCM whose conditioning context also carries scene composition.

    Everything else -- forward, losses, identity at init, detach policy -- is
    inherited unchanged, so this differs from generation 1 in exactly one
    variable.
    """

    def __init__(self, in_channels, new_channels, num_classes, **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.ccm = SceneContextConditionedMetric(
            embed_dims=self.channels,
            rank=self.ccm.rank,
            hidden=self.ccm.ccm_g[0].out_features,
            top_p=self.ccm.top_p,
            gain_scale=self.ccm.gain_scale)
