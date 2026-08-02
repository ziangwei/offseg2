# OffSeg + Rectangular Self-Calibration Module (RCM) as Pyramid Context Extraction.
#
# The mechanism is transplanted verbatim from CGRSeg (ECCV 2024,
# "Context-Guided Spatial Feature Reconstruction for Efficient Semantic
# Segmentation", arXiv:2405.06228, Ni et al., Huawei Noah's Ark Lab).
# Nothing here is invented: every operator and every hyper-parameter below is
# the one their ablations selected.
#
# Why this module, on this decoder
# --------------------------------
# The OffSeg decoder is  pre(1x1) -> FreqFusion cascade -> align(1x1) -> offset
# classifier.  There is no context / large-receptive-field stage anywhere in it.
# Every competitor at this scale has one: SegNeXt has Hamburger, CGRSeg has
# pyramid context + RCM, VWFormer has varying-window attention, LRFormer has
# low-resolution self-attention.  The only thing we ever tested was a global
# average pool (offsegccms, -0.34), which is the weakest member of that family
# and does not license killing the family.
#
# CGRSeg's own ablation (their Table 4, CGRSeg-T on ADE20K):
#     base                     40.86   3.56G   6.08M
#     + RCM as PCE             42.09  (+1.23)  +0.19G
#     + RCM as SFR             +1.13            +0.17G
# and their Table 6 puts RCA above every alternative token mixer at equal FLOPs:
#     Self-Att 39.9 / ConvNeXt 41.6 / InceptionNeXt 41.6 / CoordAtt 41.5 /
#     GatherExcite 41.7 / RCA 43.6.
# Their backbone is EfficientFormerV2, the same family we use, so the transplant
# is onto native soil.
#
# The formulas (their Eq. 2-4):
#     shape self-calibration   xi_C(y)   = sigmoid( psi_{k x 1}( BN-ReLU( psi_{1 x k}(y) ) ) )
#     fusion                   xi_F(x,y) = psi_{3 x 3}(x)  *  y
#     RCM                      out       = rho( xi_F( x, xi_C( H_P(x) (+) V_P(x) ) ) ) + x
# with (+) broadcast addition, psi depthwise convolution, rho = BN + MLP
# (MetaNeXt block structure, which their Sec. 3.2 says they follow).
#
# Hyper-parameters, all taken from their ablations, none tuned by us:
#     strip kernel k = 11        (their Table 9: none 41.27 / 5 42.51 / 7 43.39
#                                 / 9 43.50 / 11 43.61)
#     fusion conv    = 3 x 3     (their Table 10: none 42.40 / 1x1 41.70 /
#                                 3x3 43.39 / 5x5 42.99)
#     key-area op    = addition  (their Table 8: none 42.22 / mul 43.08 /
#                                 add 43.61)
#     pyramid scale  = H/64      (their Eq. 1: AP(F2,8), AP(F3,4), AP(F4,2))
#     F1 (stride 4) is not part of the pyramid, exactly as in their Eq. 1.
#
# Two deliberate deviations, both declared:
#   (1) Identity start.  Each level's context injection carries a learnable
#       scalar gamma initialised to 0, so at step 0 this head is bit-identical
#       to OffSegHead.  This is our own rule (TAM law: every winner so far has
#       been "structural part + identity start"; SDR / LDR / focus are the
#       counter-examples), not CGRSeg's.  It costs 3 scalars.
#   (2) Norm inside RCM follows the head's norm_cfg (GN-32 here) instead of BN.
#       BN at an 8x8 pyramid would make the read-out depend on the per-GPU batch
#       size, and this repo has already been bitten once by a per-GPU BN
#       confound (2x8 vs 4x4).  GN is batch-independent, so 4x4 and 2x8 stay
#       comparable.  Set rcm_norm='bn' to reproduce CGRSeg exactly.
#
# What this file does NOT do: RCM as spatial feature reconstruction (their
# second use, +1.13 on its own).  One variable per generation.  If PCE reads out
# positive, SFR is generation 2 and the two are expected to be additive because
# they act at different resolutions.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .freqfusion import FreqFusion
from mmseg.models.decode_heads import Offset_Learning


class RectangularSelfCalibrationAttention(nn.Module):
    """RCA, CGRSeg Eq. 2-3.

    Horizontal and vertical pooling give two axis vectors; broadcast addition
    of the two models a rectangular region of interest; two large-kernel strip
    convolutions then bend that rectangle towards the foreground; the result
    gates a locally-refined copy of the input.
    """

    def __init__(self, dim, kernel_size=11, fusion_kernel=3, norm_cfg=None):
        super().__init__()
        pad = kernel_size // 2

        # xi_C : shape self-calibration
        self.conv_h = nn.Conv2d(dim, dim, (1, kernel_size),
                                padding=(0, pad), groups=dim, bias=False)
        self.norm = build_norm_layer(norm_cfg, dim)[1]
        self.act = nn.ReLU(inplace=True)
        self.conv_v = nn.Conv2d(dim, dim, (kernel_size, 1),
                                padding=(pad, 0), groups=dim, bias=False)

        # xi_F : local detail branch that the attention gates
        self.local = nn.Conv2d(dim, dim, fusion_kernel,
                               padding=fusion_kernel // 2, groups=dim, bias=False)

    def forward(self, x):
        # H_P(x) : (B, C, H, 1)   V_P(x) : (B, C, 1, W)   broadcast add -> (B,C,H,W)
        y = x.mean(dim=3, keepdim=True) + x.mean(dim=2, keepdim=True)
        y = self.conv_v(self.act(self.norm(self.conv_h(y))))
        y = torch.sigmoid(y)
        return self.local(x) * y


class RCM(nn.Module):
    """Rectangular Self-Calibration Module = RCA in a MetaNeXt block."""

    def __init__(self, dim, kernel_size=11, mlp_ratio=4, norm_cfg=None,
                 gamma_init=0.0):
        super().__init__()
        self.rca = RectangularSelfCalibrationAttention(
            dim, kernel_size=kernel_size, norm_cfg=norm_cfg)
        self.norm = build_norm_layer(norm_cfg, dim)[1]
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        # identity start (deviation 1)
        self.gamma = nn.Parameter(gamma_init * torch.ones(dim))

    def forward(self, x):
        y = self.mlp(self.norm(self.rca(x)))
        return x + self.gamma.view(1, -1, 1, 1) * y


@MODELS.register_module()
class OffSegRCM(BaseDecodeHead):
    """OffSeg decode head with CGRSeg pyramid context extraction.

    Args:
        rcm_depth (int): number of stacked RCMs in the pyramid. CGRSeg stacks
            several; 2 is the conservative starting point here.
        rcm_kernel (int): strip convolution kernel, 11 per their Table 9.
        rcm_mlp_ratio (int): MetaNeXt MLP expansion, 4.
        rcm_levels (tuple): which pyramid levels join the context stage.
            (1, 2, 3) = strides 8/16/32, i.e. F1 dropped, per their Eq. 1.
        rcm_pool_div (int): the pyramid resolution is the last level's size
            divided by this. 2 gives H/64, per their Eq. 1.
        rcm_norm (str): 'head' uses norm_cfg (GN, batch-independent, default);
            'bn' reproduces CGRSeg exactly.
    """

    def __init__(self,
                 in_channels,
                 new_channels,
                 num_classes,
                 rcm_depth=2,
                 rcm_kernel=11,
                 rcm_mlp_ratio=4,
                 rcm_levels=(1, 2, 3),
                 rcm_pool_div=2,
                 rcm_norm='head',
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         num_classes=num_classes,
                         input_transform='multiple_select',
                         **kwargs)
        self.new_channels = new_channels
        self.rcm_levels = tuple(rcm_levels)
        self.rcm_pool_div = rcm_pool_div

        # ---- everything below this line is OffSegHead, unchanged ----
        self.pre = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.pre.append(
                ConvModule(self.in_channels[i],
                           self.new_channels[i],
                           1,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg,
                           act_cfg=self.act_cfg))

        self.freqfusions = nn.ModuleList()
        rev_channels = new_channels[::-1]
        pre_c = rev_channels[0]
        for c in rev_channels[1:]:
            self.freqfusions.append(
                FreqFusion(hr_channels=c, lr_channels=pre_c,
                           compressed_channels=(pre_c + c) // 4))
            pre_c += c

        self.align = ConvModule(
            sum(self.new_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.offset_learning = Offset_Learning(self.num_classes, self.channels)
        # ---- end of unchanged OffSegHead ----

        # ---- new: pyramid context extraction ----
        rcm_norm_cfg = (dict(type='BN', requires_grad=True)
                        if rcm_norm == 'bn' else self.norm_cfg)
        self.pce_split = [self.new_channels[i] for i in self.rcm_levels]
        pce_dim = sum(self.pce_split)
        self.pce = nn.Sequential(*[
            RCM(pce_dim, kernel_size=rcm_kernel, mlp_ratio=rcm_mlp_ratio,
                norm_cfg=rcm_norm_cfg) for _ in range(rcm_depth)
        ])
        # one identity gate per level, zero-initialised
        self.pce_gamma = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in self.rcm_levels])

    def _pyramid_context(self, feats):
        """CGRSeg Eq. 1, then broadcast the context back to each level."""
        ref = feats[self.rcm_levels[-1]]
        th = max(1, ref.shape[-2] // self.rcm_pool_div)
        tw = max(1, ref.shape[-1] // self.rcm_pool_div)

        pooled = [F.adaptive_avg_pool2d(feats[i], (th, tw))
                  for i in self.rcm_levels]
        p = self.pce(torch.cat(pooled, dim=1))
        parts = torch.split(p, self.pce_split, dim=1)

        for k, i in enumerate(self.rcm_levels):
            guide = resize(parts[k], size=feats[i].shape[2:],
                           mode='bilinear', align_corners=self.align_corners)
            feats[i] = feats[i] + self.pce_gamma[k] * guide
        return feats

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        feats = [self.pre[i](inputs[i]) for i in range(len(inputs))]

        feats = self._pyramid_context(feats)

        feats = feats[::-1]
        lowres_feat = feats[0]
        for hires_feat, freqfusion in zip(feats[1:], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat,
                                                    lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)

        output = self.align(lowres_feat)
        output = self.offset_learning(output)
        return output
