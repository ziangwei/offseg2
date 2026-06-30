# -*- coding: utf-8 -*-
"""PARSeg-LCR: local candidate relation for PARSeg3.

Probe results say the correct class is often already in PARSeg3's top-k list,
but the first-place ranking is wrong. LCR avoids another global 150-way metric
head: it builds a per-pixel candidate set from raw base logits, scores only
those candidate/class relations against the aligned feature, and injects the
learned candidate delta before PAL refinement and AGCF.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


class LocalCandidateRelation(nn.Module):
    """Conditioned relation scorer over each pixel's local candidate set."""

    def __init__(
        self,
        in_channels,
        num_classes,
        relation_dim=64,
        topk=5,
        hidden=128,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.relation_dim = relation_dim
        self.topk = topk

        self.feat_proj = ConvModule(
            in_channels,
            relation_dim,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.class_embed = nn.Embedding(num_classes, relation_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(relation_dim * 3 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

    def forward(self, feat_aligned, raw_base_logits):
        topk = min(max(int(self.topk), 1), self.num_classes)
        topk_idx = raw_base_logits.detach().topk(k=topk, dim=1).indices
        topk_vals = raw_base_logits.detach().gather(1, topk_idx)
        topk_prob = F.softmax(raw_base_logits.detach(), dim=1).gather(1, topk_idx)

        feat = self.feat_proj(feat_aligned)
        feat = F.normalize(feat, p=2, dim=1, eps=1e-6)
        pixel_feat = feat.permute(0, 2, 3, 1).contiguous()

        idx_bhwk = topk_idx.permute(0, 2, 3, 1).contiguous()
        class_feat = self.class_embed(idx_bhwk)
        class_feat = F.normalize(class_feat, p=2, dim=-1, eps=1e-6)

        pixel_feat = pixel_feat.unsqueeze(3).expand_as(class_feat)
        prob_bhwk = topk_prob.permute(0, 2, 3, 1).contiguous()
        top1_prob = prob_bhwk[..., :1]
        prob_gap = top1_prob - prob_bhwk

        relation_input = torch.cat(
            [
                pixel_feat,
                class_feat,
                torch.abs(pixel_feat - class_feat),
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
class PARSegLCR(PARSeg3):
    """PARSeg3 with local candidate relation before PAL refinement."""

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
        self.lcr = LocalCandidateRelation(
            in_channels=self.channels,
            num_classes=num_classes,
            relation_dim=int(self.args.get("lcr_dim", 64)),
            topk=int(self.args.get("lcr_topk", 5)),
            hidden=int(self.args.get("lcr_hidden", 128)),
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.lcr_gate_max = float(self.args.get("lcr_gate_max", 0.35))
        init_gate = float(self.args.get("lcr_gate_init", 0.05))
        init_gate = min(max(init_gate, 1e-4), self.lcr_gate_max - 1e-4)
        ratio = init_gate / self.lcr_gate_max
        self.lcr_alpha = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )

    def _lcr_gate(self):
        return self.lcr_gate_max * torch.sigmoid(self.lcr_alpha)

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
        relation = self.lcr(feat_aligned, raw_base_head_logits)

        gate = self._lcr_gate()
        base_head_logits = raw_base_head_logits + gate * relation["delta_logits"]
        relation_logits = raw_base_head_logits.detach() + gate * relation["delta_logits"]

        refinement_head_logits, calibrated_attr_tokens = self.prototype_attribute_refinement(
            feat_aligned,
            base_head_logits,
        )

        fusion_mode = self.args.get("fusion_mode", "AGCF")
        if fusion_mode == "AGCF":
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == "avg":
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == "catconv":
            final_logits = self.fuse_catconv(torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f"Unsupported fusion_mode for PARSegLCR: {fusion_mode}")

        return dict(
            raw_base_head_logits=raw_base_head_logits,
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            lcr_relation_logits=relation_logits,
            lcr_delta_logits=relation["delta_logits"],
            lcr_candidate_delta=relation["candidate_delta"],
            lcr_candidate_idx=relation["candidate_idx"],
            lcr_candidate_raw_scores=relation["candidate_raw_scores"],
            lcr_gate=gate,
        )

    def _stack_label(self, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        return seg_label.squeeze(1) if seg_label.dim() == 4 else seg_label

    def _small_label(self, seg_label, size):
        return F.interpolate(
            seg_label.unsqueeze(1).float(),
            size=size,
            mode="nearest",
        ).squeeze(1).long()

    def _lcr_rank_loss(self, raw_logits, corrected_logits, candidate_idx, seg_label):
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

        margin = float(self.args.get("lcr_rank_margin", 0.20))
        rank_loss = F.relu(margin + neg_scores - gt_score.unsqueeze(1))
        neg_valid = (~gt_match) & gt_in_candidates.unsqueeze(1)

        raw_pred = raw_logits.detach().argmax(dim=1)
        hard_pixel = gt_in_candidates & (raw_pred != safe_label)
        hard_weight = float(self.args.get("lcr_rank_hard_weight", 2.0))
        pixel_weight = gt_in_candidates.float() + (hard_weight - 1.0) * hard_pixel.float()
        weight = neg_valid.float() * pixel_weight.unsqueeze(1)
        return (rank_loss * weight).sum() / weight.sum().clamp_min(1.0)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_label(batch_data_samples)
        target_size = seg_label.shape[-2:]

        auxw = float(self.args.get("lcr_auxw", 0.20))
        if auxw > 0:
            relation_logits = resize(
                input=seg_logits["lcr_relation_logits"],
                size=target_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            losses["loss_lcr_aux"] = self.loss_decode(
                relation_logits,
                seg_label,
                ignore_index=self.ignore_index,
            ) * auxw

        rankw = float(self.args.get("lcr_rankw", 0.20))
        if rankw > 0:
            h, w = seg_logits["lcr_relation_logits"].shape[-2:]
            seg_small = self._small_label(seg_label, size=(h, w))
            losses["loss_lcr_rank"] = self._lcr_rank_loss(
                raw_logits=seg_logits["raw_base_head_logits"],
                corrected_logits=seg_logits["lcr_relation_logits"],
                candidate_idx=seg_logits["lcr_candidate_idx"],
                seg_label=seg_small,
            ) * rankw

        return losses
