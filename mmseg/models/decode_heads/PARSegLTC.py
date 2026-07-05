# -*- coding: utf-8 -*-
"""PARSeg-LTC: LTA + GT-routed class-granular text-alignment loss (switch B).

On top of PARSegLTA (frozen language anchors in the candidate scorer), LTC
adds ONE training-only loss and nothing else:

  For each image, pool the scorer's own relation space
  (lcr.feat_proj(feat_aligned), the exact 64-d space where pixel-candidate
  relations are scored) over each GT-present class's mask -> one visual
  prototype per present class. InfoNCE-align each prototype to its class's
  anchored vector against all 150 anchored vectors.

This is the train-only translation of DTFormer's finding that GT-mapped
vocabulary guidance works best: the per-image GT decides the POSITIVE
routing (which prototype pulls to which anchor) -- information the CE loss
already possesses, so there is no leakage surface -- while the anchors
contribute the pretrained inter-class language geometry. Because confusable
classes sit close in text space (wall~door 0.86, sidewalk~road 0.90), the
InfoNCE softmax automatically concentrates gradient on exactly those hard
negatives.

Granularity is deliberately CLASS-level (<= ~8-15 present classes per
image, so tens of pairs per batch), NOT DenseCLIP-style dense pixel-text
contrast: dense auxiliary gradients on the shared trunk are the family that
kept failing here (SDR/APC).

args (new): ltc_infoncew=0.15, ltc_tau=0.1, ltc_warmup_iters=8000,
            ltc_min_pixels=4
"""
import torch
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSegLTA import PARSegLTA


@MODELS.register_module()
class PARSegLTC(PARSegLTA):
    """PARSegLTA + GT-routed prototype-to-anchor InfoNCE (training only)."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes,
                 args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.args = args or {}
        self.ltc_infoncew = float(self.args.get('ltc_infoncew', 0.15))
        self.ltc_tau = float(self.args.get('ltc_tau', 0.1))
        self.ltc_warmup_iters = int(self.args.get('ltc_warmup_iters', 8000))
        self.ltc_min_pixels = int(self.args.get('ltc_min_pixels', 4))
        if self.ltc_tau <= 0:
            raise ValueError('ltc_tau must be positive')
        self._ltc_step = 0

    def _ltc_iter(self):
        try:
            from mmengine.logging import MessageHub
            it = MessageHub.get_current_instance().get_info('iter')
            if it is not None:
                return int(it)
        except Exception:
            pass
        return self._ltc_step

    # ------------------------------------------------------------------
    # forward: identical to PARSegLCR, plus stashing feat_aligned so the
    # alignment loss can reuse the scorer's own projection space.
    # ------------------------------------------------------------------
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
        return self._forward_from_aligned(feat_aligned)

    def _forward_from_aligned(self, feat_aligned):
        raw_base_head_logits = self.offset_learning(feat_aligned)
        relation = self.lcr(feat_aligned, raw_base_head_logits)

        gate = self._lcr_gate()
        base_head_logits = raw_base_head_logits + gate * relation['delta_logits']
        relation_logits = raw_base_head_logits.detach() + gate * relation['delta_logits']

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
            raise ValueError(f'Unsupported fusion_mode for PARSegLTC: {fusion_mode}')

        return dict(
            raw_base_head_logits=raw_base_head_logits,
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            final_logits=final_logits,
            lcr_relation_logits=relation_logits,
            lcr_delta_logits=relation['delta_logits'],
            lcr_candidate_delta=relation['candidate_delta'],
            lcr_candidate_idx=relation['candidate_idx'],
            lcr_candidate_raw_scores=relation['candidate_raw_scores'],
            lcr_gate=gate,
            ltc_feat_aligned=feat_aligned,
        )

    # ------------------------------------------------------------------
    # losses: parent (LCR aux + rank + PARSeg3 recipe) + one InfoNCE
    # ------------------------------------------------------------------
    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        self._ltc_step += 1

        feat_aligned = seg_logits.get('ltc_feat_aligned')
        if feat_aligned is None or self.ltc_infoncew <= 0:
            return losses

        seg_label = self._stack_label(batch_data_samples)     # [B, H, W]
        Hf, Wf = feat_aligned.shape[-2:]
        y = self._small_label(seg_label, size=(Hf, Wf))       # [B, Hf, Wf]
        valid = (y != self.ignore_index) & (y < self.num_classes)

        # scorer's own relation space (shared projection with the candidate
        # scorer -- this is the space whose geometry we want to organize)
        feat = self.lcr.feat_proj(feat_aligned)               # [B, D, Hf, Wf]
        feat = F.normalize(feat, p=2, dim=1, eps=1e-6)

        B = y.shape[0]
        protos, targets = [], []
        for b in range(B):
            cls = torch.unique(y[b][valid[b]])
            for c in cls:
                mask = (y[b] == c) & valid[b]
                if int(mask.sum()) < self.ltc_min_pixels:
                    continue
                protos.append(feat[b, :, mask].mean(dim=1))
                targets.append(int(c))

        if len(protos) == 0:
            losses['loss_ltc_infonce'] = feat.sum() * 0.0
            return losses

        z = F.normalize(torch.stack(protos), p=2, dim=-1)     # [P, D]
        anchors = F.normalize(self.lcr.class_embed.matrix(), p=2, dim=-1)
        logits = z @ anchors.t() / self.ltc_tau               # [P, C]
        target = torch.tensor(targets, device=logits.device, dtype=torch.long)

        ramp = min(1.0, float(self._ltc_iter()) / max(1, self.ltc_warmup_iters))
        losses['loss_ltc_infonce'] = (
            F.cross_entropy(logits, target) * self.ltc_infoncew * ramp)
        return losses
