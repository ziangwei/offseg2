# -*- coding: utf-8 -*-
"""PARSeg-PLCR: PAL-guided local candidate relation.

PLCR keeps the useful LCR idea (dynamic per-pixel top-k candidate comparison)
but removes the independent class embedding. Candidate class representations
come from PAL class features, so local ranking is tied to PARSeg's attribute
primitive rather than becoming an external logits MLP.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .PARSegPALX import PARSegPALX


class PALCandidateRelation(nn.Module):
    """Local candidate relation using PAL-generated class features."""

    def __init__(
        self,
        feat_dim=256,
        relation_dim=64,
        topk=5,
        hidden=128,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.topk = topk
        self.pixel_proj = ConvModule(
            feat_dim,
            relation_dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.class_proj = nn.Linear(feat_dim, relation_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(relation_dim * 3 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

    def _gather_class_features(self, class_feats, topk_idx):
        b, _, d = class_feats.shape
        _, k, h, w = topk_idx.shape
        idx_flat = topk_idx.permute(0, 2, 3, 1).reshape(b, h * w * k)
        gathered = torch.gather(
            class_feats,
            dim=1,
            index=idx_flat.unsqueeze(-1).expand(-1, -1, d),
        )
        return gathered.reshape(b, h, w, k, d)

    def forward(self, refinement_feats, pal_class_feats, raw_base_logits):
        topk = min(max(int(self.topk), 1), raw_base_logits.shape[1])
        topk_idx = raw_base_logits.detach().topk(k=topk, dim=1).indices
        topk_vals = raw_base_logits.detach().gather(1, topk_idx)
        topk_prob = F.softmax(raw_base_logits.detach(), dim=1).gather(1, topk_idx)

        pixel_feat = self.pixel_proj(refinement_feats)
        pixel_feat = F.normalize(pixel_feat, p=2, dim=1, eps=1e-6)
        pixel_feat = pixel_feat.permute(0, 2, 3, 1).contiguous()

        class_feat = self.class_proj(pal_class_feats)
        class_feat = F.normalize(class_feat, p=2, dim=-1, eps=1e-6)
        candidate_class_feat = self._gather_class_features(class_feat, topk_idx)

        pixel_feat = pixel_feat.unsqueeze(3).expand_as(candidate_class_feat)
        prob_bhwk = topk_prob.permute(0, 2, 3, 1).contiguous()
        prob_gap = prob_bhwk[..., :1] - prob_bhwk

        relation_input = torch.cat(
            [
                pixel_feat,
                candidate_class_feat,
                torch.abs(pixel_feat - candidate_class_feat),
                prob_bhwk.unsqueeze(-1),
                prob_gap.unsqueeze(-1),
            ],
            dim=-1,
        )
        candidate_delta = self.score_mlp(relation_input).squeeze(-1)
        candidate_delta = candidate_delta.permute(0, 3, 1, 2).contiguous()

        delta_logits = raw_base_logits.new_zeros(raw_base_logits.shape)
        delta_logits.scatter_(1, topk_idx, candidate_delta)

        return dict(
            delta_logits=delta_logits,
            candidate_delta=candidate_delta,
            candidate_idx=topk_idx,
            candidate_raw_scores=topk_vals,
        )


@MODELS.register_module()
class PARSegPLCR(PARSegPALX):
    """PARSegPALX with PAL-guided local candidate relation."""

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
        self.plcr = PALCandidateRelation(
            feat_dim=self.channels,
            relation_dim=int(self.args.get("plcr_dim", 64)),
            topk=int(self.args.get("plcr_topk", 5)),
            hidden=int(self.args.get("plcr_hidden", 128)),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.plcr_gate_max = float(self.args.get("plcr_gate_max", 0.30))
        init_gate = float(self.args.get("plcr_gate_init", 0.04))
        init_gate = min(max(init_gate, 1e-4), self.plcr_gate_max - 1e-4)
        ratio = init_gate / self.plcr_gate_max
        self.plcr_alpha = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def _plcr_gate(self):
        return self.plcr_gate_max * torch.sigmoid(self.plcr_alpha)

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
        raw_refine = self.prototype_attribute_refinement(feat_aligned, raw_base_head_logits)
        relation = self.plcr(
            raw_refine["palx_refinement_feats"],
            raw_refine["palx_class_feats"],
            raw_base_head_logits,
        )

        gate = self._plcr_gate()
        base_head_logits = raw_base_head_logits + gate * relation["delta_logits"]
        relation_logits = raw_base_head_logits.detach() + gate * relation["delta_logits"]

        refine = self.prototype_attribute_refinement(feat_aligned, base_head_logits)
        refinement_head_logits = refine["refinement_head_logits"]

        fusion_mode = self.args.get("fusion_mode", "AGCF")
        if fusion_mode == "AGCF":
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == "avg":
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == "catconv":
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f"Unsupported fusion_mode for PARSegPLCR: {fusion_mode}")

        return dict(
            raw_base_head_logits=raw_base_head_logits,
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=refine["calibrated_attr_tokens"],
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            palx_class_cos=refine["palx_class_cos"],
            palx_class_feats=refine["palx_class_feats"],
            palx_refinement_feats=refine["palx_refinement_feats"],
            palx_route_prob=refine["palx_route_prob"],
            palx_class_proto=refine["palx_class_proto"],
            plcr_relation_logits=relation_logits,
            plcr_delta_logits=relation["delta_logits"],
            plcr_candidate_delta=relation["candidate_delta"],
            plcr_candidate_idx=relation["candidate_idx"],
            plcr_candidate_raw_scores=relation["candidate_raw_scores"],
            plcr_gate=gate,
        )

    def _plcr_rank_loss(self, raw_logits, corrected_logits, candidate_idx, seg_label):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return corrected_logits.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0
        candidate_scores = corrected_logits.gather(1, candidate_idx)
        gt_match = candidate_idx == safe_label.unsqueeze(1)
        gt_in_candidates = gt_match.any(dim=1) & valid
        if not bool(gt_in_candidates.any()):
            return corrected_logits.sum() * 0.0

        gt_score = (candidate_scores * gt_match.float()).sum(dim=1)
        neg_scores = candidate_scores.masked_fill(gt_match, -1e4)
        margin = float(self.args.get("plcr_rank_margin", 0.20))
        rank_loss = F.relu(margin + neg_scores - gt_score.unsqueeze(1))
        neg_valid = (~gt_match) & gt_in_candidates.unsqueeze(1)

        raw_pred = raw_logits.detach().argmax(dim=1)
        hard_pixel = gt_in_candidates & (raw_pred != safe_label)
        hard_weight = float(self.args.get("plcr_rank_hard_weight", 2.0))
        pixel_weight = gt_in_candidates.float() + (hard_weight - 1.0) * hard_pixel.float()
        weight = neg_valid.float() * pixel_weight.unsqueeze(1)
        return (rank_loss * weight).sum() / weight.sum().clamp_min(1.0)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_label(batch_data_samples)
        target_size = seg_label.shape[-2:]

        auxw = float(self.args.get("plcr_auxw", 0.15))
        if auxw > 0:
            relation_logits = resize(
                input=seg_logits["plcr_relation_logits"],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            losses["loss_plcr_aux"] = self.loss_decode(
                relation_logits,
                seg_label,
                ignore_index=self.ignore_index,
            ) * auxw

        rankw = float(self.args.get("plcr_rankw", 0.20))
        if rankw > 0:
            h, w = seg_logits["plcr_relation_logits"].shape[-2:]
            seg_small = self._small_label(seg_label, size=(h, w))
            losses["loss_plcr_rank"] = self._plcr_rank_loss(
                raw_logits=seg_logits["raw_base_head_logits"],
                corrected_logits=seg_logits["plcr_relation_logits"],
                candidate_idx=seg_logits["plcr_candidate_idx"],
                seg_label=seg_small,
            ) * rankw

        return losses
