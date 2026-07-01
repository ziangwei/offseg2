# -*- coding: utf-8 -*-
"""PARSeg-HCE: end-to-end latent confusion refinement.

HCE-v2 removes the hand-written ADE20K confusion sets from the first draft.
The model now learns a latent class-relation matrix from trainable class
tokens, uses the current base prediction to form a per-pixel candidate
subspace, and applies a bounded residual only inside that subspace.

This keeps the useful part of the original idea: dense prediction errors are
often local re-decision errors among semantically plausible classes, while a
free dense residual can damage classes that were already correct. The residual
is candidate-masked, but it is not entropy-gated: confident-wrong pixels are a
major failure mode, so low base entropy must not suppress the correction path.
The routing is learned jointly with ordinary segmentation losses and does not
require a pretrained PARSeg3 checkpoint, validation-set confusion statistics,
or dataset-specific class groups.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3


class LearnedConfusionRouter(nn.Module):
    """Learns a class-to-class candidate relation and routes each pixel.

    Rows of the relation matrix answer: if the base distribution assigns mass
    to class i, which classes should be considered plausible local
    alternatives? The current base probabilities are detached before routing,
    so the router consumes the present decision state without becoming a
    second dense classifier target by itself.
    """

    def __init__(
        self,
        num_classes,
        relation_dim=64,
        candidate_topk=8,
        relation_temperature=1.0,
        self_bias=2.0,
    ):
        super().__init__()
        if relation_dim <= 0:
            raise ValueError("hce_relation_dim must be positive")
        if relation_temperature <= 0:
            raise ValueError("hce_relation_temperature must be positive")

        self.num_classes = int(num_classes)
        self.candidate_topk = int(candidate_topk)
        self.relation_temperature = float(relation_temperature)
        self.self_bias = float(self_bias)
        self.class_tokens = nn.Parameter(torch.empty(self.num_classes, relation_dim))
        nn.init.normal_(self.class_tokens, mean=0.0, std=0.02)

    def relation_matrix(self):
        tokens = F.normalize(self.class_tokens, p=2, dim=-1, eps=1e-6)
        logits = tokens @ tokens.t()
        logits = logits / self.relation_temperature
        eye = torch.eye(self.num_classes, device=logits.device, dtype=logits.dtype)
        logits = logits + eye * self.self_bias
        return F.softmax(logits, dim=-1)

    def _keep_topk(self, candidate_weights):
        if self.candidate_topk <= 0:
            return candidate_weights.new_zeros(candidate_weights.shape)
        if self.candidate_topk >= candidate_weights.shape[1]:
            return candidate_weights

        values, indices = candidate_weights.topk(k=self.candidate_topk, dim=1)
        sparse = candidate_weights.new_zeros(candidate_weights.shape)
        sparse.scatter_(1, indices, values)
        return sparse

    def forward(self, base_head_logits):
        p_base = F.softmax(base_head_logits.detach(), dim=1)
        relation = self.relation_matrix().to(device=p_base.device, dtype=p_base.dtype)
        candidate_weights = torch.einsum('bihw,ij->bjhw', p_base, relation)
        return self._keep_topk(candidate_weights)


class CandidateMaskedExpert(nn.Module):
    """Predicts residual logits that will be masked by learned candidates."""

    def __init__(self, in_channels, num_classes, hidden=None,
                 conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        hidden = hidden or in_channels
        self.transform = ConvModule(
            in_channels, hidden, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.classifier = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def forward(self, feat):
        return self.classifier(self.transform(feat))


@MODELS.register_module()
class PARSegHCE(PARSeg3):
    """PARSeg3 with learned latent confusion routing before PAL refinement."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.args = args or {}

        hidden = int(self.args.get('hce_hidden', self.channels))
        self.confusion_router = LearnedConfusionRouter(
            num_classes=num_classes,
            relation_dim=int(self.args.get('hce_relation_dim', 64)),
            candidate_topk=int(self.args.get('hce_candidate_topk', 8)),
            relation_temperature=float(self.args.get('hce_relation_temperature', 1.0)),
            self_bias=float(self.args.get('hce_self_bias', 2.0)),
        )
        self.candidate_expert = CandidateMaskedExpert(
            self.channels, num_classes, hidden=hidden,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,
        )
        self.hce_stop_gradient = bool(self.args.get('hce_stop_gradient', True))

        gate_max = float(self.args.get('hce_gate_max', 0.30))
        init_gate = float(self.args.get('hce_gate_init', 0.05))
        if gate_max <= 0:
            raise ValueError("hce_gate_max must be positive")
        if not (0.0 < init_gate < gate_max):
            raise ValueError("hce_gate_init must satisfy 0 < init < hce_gate_max")
        ratio = init_gate / gate_max
        self.hce_gate_max = gate_max
        self.hce_alpha = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def _hce_gate(self):
        return self.hce_gate_max * torch.sigmoid(self.hce_alpha)

    def _hce_refine_base(self, feat_aligned, raw_base_head_logits):
        expert_feat = feat_aligned.detach() if self.hce_stop_gradient else feat_aligned
        delta_logits = self.candidate_expert(expert_feat)
        candidate_weights = self.confusion_router(raw_base_head_logits)
        masked_delta = delta_logits * candidate_weights
        base_head_logits = raw_base_head_logits + self._hce_gate() * masked_delta
        return base_head_logits, delta_logits, candidate_weights

    def forward(self, inputs, return_vis=False):
        inputs = self._transform_inputs(inputs)

        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        lowres_feat = new_inputs[-1]
        for hires_feat, freqfusion in zip(new_inputs[:-1][::-1], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [
                    hires_feat.reshape(b * 4, -1, h, w),
                    lowres_feat.reshape(b * 4, -1, h, w),
                ],
                dim=1,
            ).reshape(b, -1, h, w)
        feat_aligned = self.align(lowres_feat)

        raw_base_head_logits = self.offset_learning(feat_aligned)
        base_head_logits, delta_logits, candidate_weights = self._hce_refine_base(
            feat_aligned,
            raw_base_head_logits,
        )

        refinement_head_logits, calibrated_attr_tokens = self.prototype_attribute_refinement(
            feat_aligned,
            base_head_logits,
        )
        fusion_mode = self.args.get('fusion_mode', 'AGCF')
        if fusion_mode == 'AGCF':
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == 'avg':
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == 'catconv':
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f"Unsupported fusion_mode for PARSegHCE: {fusion_mode}")

        return dict(
            raw_base_head_logits=raw_base_head_logits,
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            hce_feat_aligned=feat_aligned,
            hce_delta_logits=delta_logits,
            hce_candidate_weights=candidate_weights,
        )

    def _candidate_sparsity_loss(self, candidate_weights):
        mass = candidate_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        prob = candidate_weights / mass
        entropy = -(prob * torch.log(prob.clamp_min(1e-6))).sum(dim=1)
        entropy = entropy / (math.log(candidate_weights.shape[1]) + 1e-6)
        return entropy.mean()

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        sparsityw = float(self.args.get('hce_sparsityw', 0.0))
        if sparsityw > 0:
            losses['loss_hce_sparsity'] = (
                self._candidate_sparsity_loss(seg_logits['hce_candidate_weights']) * sparsityw
            )
        return losses
