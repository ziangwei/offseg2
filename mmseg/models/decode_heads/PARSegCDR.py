# -*- coding: utf-8 -*-
"""PARSeg-CDR: training-only candidate decision ranking.

CDR keeps PARSeg3 inference unchanged. It only adds a local top-k ranking loss
during training, directly targeting the diagnosis that the GT class is often in
the candidate list but ranked below a nearby wrong class.
"""

import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


@MODELS.register_module()
class PARSegCDR(PARSeg3):
    """PARSeg3 with candidate decision ranking losses only."""

    def _stack_label(self, batch_data_samples):
        seg_label = self._stack_batch_gt(batch_data_samples)
        return seg_label.squeeze(1) if seg_label.dim() == 4 else seg_label

    def _candidate_rank_loss(self, logits, seg_label):
        valid = seg_label != self.ignore_index
        if not bool(valid.any()):
            return logits.sum() * 0.0

        safe_label = seg_label.clone()
        safe_label[~valid] = 0

        topk = min(max(int(self.args.get("cdr_topk", 5)), 1), self.num_classes)
        topk_idx = logits.detach().topk(k=topk, dim=1).indices
        candidate_scores = logits.gather(1, topk_idx)

        gt_match = topk_idx == safe_label.unsqueeze(1)
        gt_in_candidates = gt_match.any(dim=1) & valid
        if not bool(gt_in_candidates.any()):
            return logits.sum() * 0.0

        gt_score = (candidate_scores * gt_match.float()).sum(dim=1)
        neg_scores = candidate_scores.masked_fill(gt_match, -1e4)

        margin = float(self.args.get("cdr_rank_margin", 0.20))
        rank_loss = F.relu(margin + neg_scores - gt_score.unsqueeze(1))
        neg_valid = (~gt_match) & gt_in_candidates.unsqueeze(1)

        raw_pred = logits.detach().argmax(dim=1)
        hard_pixel = gt_in_candidates & (raw_pred != safe_label)
        hard_weight = float(self.args.get("cdr_hard_weight", 2.0))
        pixel_weight = gt_in_candidates.float() + (hard_weight - 1.0) * hard_pixel.float()
        weight = neg_valid.float() * pixel_weight.unsqueeze(1)
        return (rank_loss * weight).sum() / weight.sum().clamp_min(1.0)

    def _resized_rank_loss(self, logits, seg_label):
        logits = resize(
            input=logits,
            size=seg_label.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return self._candidate_rank_loss(logits, seg_label)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_label(batch_data_samples)

        basew = float(self.args.get("cdr_base_rankw", 0.05))
        if basew > 0:
            losses["loss_cdr_base_rank"] = self._resized_rank_loss(
                seg_logits["base_head_logits"],
                seg_label,
            ) * basew

        refinementw = float(self.args.get("cdr_refinement_rankw", 0.10))
        if refinementw > 0:
            losses["loss_cdr_refinement_rank"] = self._resized_rank_loss(
                seg_logits["refinement_head_logits"],
                seg_label,
            ) * refinementw

        finalw = float(self.args.get("cdr_final_rankw", 0.20))
        if finalw > 0:
            losses["loss_cdr_final_rank"] = self._resized_rank_loss(
                seg_logits["final_logits"],
                seg_label,
            ) * finalw

        return losses
