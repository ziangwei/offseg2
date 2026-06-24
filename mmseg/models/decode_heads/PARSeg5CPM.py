# -*- coding: utf-8 -*-
"""PARSeg5-CPM: cross-image prototype memory as an independent evidence source.

Why this exists (vs EAF / ICAR):
EAF and ICAR both add a context branch, but that branch reads the *same*
``feat_aligned`` the base head reads. So on the diagnosed failure mode
(all-heads-same-wrong confident interior errors) the context branch tends to
repeat the base error, because the misleading information is in the features
themselves.

CPM changes the *information source* instead of the receptive field. It keeps a
dataset-level class prototype bank, updated by EMA from GT-hit pixels across
images (no_grad, train-only). Each pixel's similarity to that bank gives
``global_logits`` -- evidence built from thousands of correct pixels of each
class across the whole dataset, which does NOT depend on this image's base
logits. The reference vectors are therefore independent of the per-image base
error; the pixel embedding still comes from shared features, so the
independence is in the *reference*, not the compared pixel.

Injection point matches EAF on purpose (tri-source residual fusion), so the
only variable between EAF and CPM is the evidence *source*. That makes the
three-model story a clean ablation:
  EAF  = inject at fusion,        source = same-image multi-dilation context
  ICAR = inject at image proto,   source = same-image multi-dilation context
  CPM  = inject at fusion,        source = cross-image GT prototype bank
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3

# NOTE: self-contained on purpose. This file only depends on the PARSeg3
# baseline (the permanent foundation), never on sibling PARSeg5* variant files,
# so it keeps working even if EAF/ICAR are deleted and only one variant is kept.


class CrossImagePrototypeMemory(nn.Module):
    """Dataset-level class prototype bank with EMA update from GT pixels.

    The bank is a registered buffer so it is saved in the checkpoint and frozen
    at test time. ``forward`` reads the bank (detached) and returns cosine
    similarity logits plus the pixel embeddings; ``update`` refreshes the bank
    from clean GT-hit pixels and is only called during training.
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        emb_dim=256,
        tau=0.1,
        momentum=0.999,
        ignore_index=255,
        min_count=1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.tau = tau
        self.momentum = momentum
        self.ignore_index = ignore_index
        self.min_count = min_count

        # Plain projection (no activation) so the embedding direction is free
        # before L2 normalization.
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=1, bias=True)

        # Zero-init bank => zero similarity => neutral global_logits at start,
        # so CPM begins close to PARSeg3 just like EAF's zero-init classifier.
        self.register_buffer("proto_bank", torch.zeros(num_classes, emb_dim))
        self.register_buffer("bank_update_count", torch.zeros(num_classes))

    def embed(self, feat):
        emb = self.proj(feat)
        return F.normalize(emb, p=2, dim=1, eps=1e-6)

    def forward(self, feat):
        emb = self.embed(feat)                       # [B, D, h, w]
        bank = self.proto_bank.detach()              # [Nc, D], no grad into bank
        global_logits = torch.einsum("bdhw,cd->bchw", emb, bank) / self.tau
        return global_logits, emb

    @torch.no_grad()
    def update(self, emb, seg_label):
        """EMA-update the bank from GT-hit pixels. emb is detached [B, D, h, w]."""
        b, d, h, w = emb.shape
        label = seg_label
        if label.dim() == 4:
            label = label.squeeze(1)

        label_low = F.interpolate(
            label.unsqueeze(1).float(), size=(h, w), mode="nearest"
        ).long().squeeze(1)                          # [B, h, w]

        emb_flat = emb.permute(0, 2, 3, 1).reshape(-1, d)
        lab_flat = label_low.reshape(-1)
        valid = lab_flat != self.ignore_index
        lab_flat = lab_flat[valid]
        emb_flat = emb_flat[valid]
        if lab_flat.numel() == 0:
            return

        sums = torch.zeros(self.num_classes, d, device=emb.device, dtype=emb.dtype)
        sums.index_add_(0, lab_flat, emb_flat)
        counts = torch.bincount(lab_flat, minlength=self.num_classes).to(emb.dtype)

        present = counts >= self.min_count
        if not bool(present.any()):
            return

        means = sums / counts.clamp_min(1.0).unsqueeze(1)
        means = F.normalize(means, p=2, dim=1, eps=1e-6)

        bank = self.proto_bank.clone()
        bank_norm = bank.norm(dim=1)
        uninit = present & (bank_norm < 1e-6)        # first time we see this class
        ema = present & (bank_norm >= 1e-6)

        if bool(uninit.any()):
            bank[uninit] = means[uninit]
        if bool(ema.any()):
            m = self.momentum
            bank[ema] = m * bank[ema] + (1.0 - m) * means[ema]

        # Renormalize (untouched rows stay normalized; empty rows stay ~0).
        bank = F.normalize(bank, p=2, dim=1, eps=1e-6)
        self.proto_bank.copy_(bank)
        self.bank_update_count += present.to(self.bank_update_count.dtype)


class EvidenceAwareCorrectionFusion(nn.Module):
    """Three-source residual fusion: base + refine correction + evidence correction.

    Inlined here (rather than imported from a sibling file) so PARSeg5CPM stays
    self-contained. Same form as EAF's fusion, so EAF vs CPM isolates only the
    evidence source.
    """

    def __init__(self, num_classes, hidden=32):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )

        mid_channels = max(num_classes // 8, 4)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_classes * 3, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes * 2, kernel_size=1, bias=True),
        )
        self.channel_floor_logit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer(
            "max_entropy",
            torch.tensor(math.log(num_classes), dtype=torch.float32),
            persistent=False,
        )

        nn.init.zeros_(self.spatial_attn[-1].weight)
        nn.init.constant_(self.spatial_attn[-1].bias[0], -2.0)
        nn.init.constant_(self.spatial_attn[-1].bias[1], -3.5)
        nn.init.zeros_(self.channel_attn[-1].weight)
        nn.init.constant_(self.channel_attn[-1].bias[:num_classes], -2.0)
        nn.init.constant_(self.channel_attn[-1].bias[num_classes:], -3.5)

    def _entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        log_probs = torch.log(probs.clamp_min(1e-6))
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        max_ent = self.max_entropy.to(device=logits.device, dtype=logits.dtype)
        return (entropy / (max_ent + 1e-6)).clamp(0.0, 1.0), probs

    def forward(self, base_logits, refine_logits, evidence_logits):
        ent_base, p_base = self._entropy(base_logits)
        ent_refine, p_refine = self._entropy(refine_logits)
        ent_evidence, p_evidence = self._entropy(evidence_logits)

        disagree_br = 0.5 * torch.sum(torch.abs(p_base - p_refine), dim=1, keepdim=True)
        disagree_be = 0.5 * torch.sum(torch.abs(p_base - p_evidence), dim=1, keepdim=True)
        disagree_re = 0.5 * torch.sum(torch.abs(p_refine - p_evidence), dim=1, keepdim=True)

        spatial = torch.sigmoid(
            self.spatial_attn(
                torch.cat([
                    ent_base,
                    ent_refine,
                    ent_evidence,
                    disagree_br,
                    disagree_be,
                    disagree_re,
                ], dim=1)
            )
        )

        channel = torch.sigmoid(
            self.channel_attn(torch.cat([base_logits, refine_logits, evidence_logits], dim=1))
        )
        b, _, _, _ = base_logits.shape
        channel = channel.view(b, 2, self.num_classes, 1, 1)

        floor = torch.sigmoid(self.channel_floor_logit)
        alpha_refine = spatial[:, 0:1] * (floor + (1.0 - floor) * channel[:, 0])
        alpha_evidence = spatial[:, 1:2] * (floor + (1.0 - floor) * channel[:, 1])

        return (
            base_logits
            + alpha_refine * (refine_logits - base_logits)
            + alpha_evidence * (evidence_logits - base_logits)
        )


@MODELS.register_module()
class PARSeg5CPM(PARSeg3):
    """PARSeg3 + cross-image prototype evidence injected via tri-source fusion."""

    def __init__(self, in_channels, new_channels, num_classes, cls_attributes, args=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        a = args or {}
        self.proto_memory = CrossImagePrototypeMemory(
            in_channels=self.channels,
            num_classes=num_classes,
            emb_dim=int(a.get("cpm_emb_dim", 256)),
            tau=float(a.get("cpm_tau", 0.1)),
            momentum=float(a.get("cpm_momentum", 0.999)),
            ignore_index=self.ignore_index,
            min_count=int(a.get("cpm_update_min_count", 1)),
        )
        # Tri-source residual fusion, defined in this same file (self-contained).
        # Same form as EAF, so EAF vs CPM isolates the evidence source only.
        # (Inherited self.fusion / self.fuse_catconv stay unused;
        # find_unused_parameters=True in the config tolerates that.)
        self.evidence_fusion = EvidenceAwareCorrectionFusion(num_classes=num_classes)

    def _forward_features(self, inputs):
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

        return self.align(lowres_feat)

    def forward(self, inputs, return_vis=False):
        feat_aligned = self._forward_features(inputs)

        base_head_logits = self.offset_learning(feat_aligned)
        refinement_head_logits, calibrated_attr_tokens = self.prototype_attribute_refinement(
            feat_aligned, base_head_logits
        )
        global_logits, pixel_emb = self.proto_memory(feat_aligned)
        final_logits = self.evidence_fusion(
            base_head_logits, refinement_head_logits, global_logits
        )

        return dict(
            base_head_logits=base_head_logits,
            calibrated_attr_tokens=calibrated_attr_tokens,
            refinement_head_logits=refinement_head_logits,
            global_logits=global_logits,
            pixel_emb=pixel_emb.detach(),
            final_logits=final_logits,
        )

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)

        globalw = self.args.get("globalw", 0.4)
        global_focusw = self.args.get("global_focusw", 0.2)
        global_pred = seg_logits.get("global_logits", None)
        base_pred = seg_logits.get("base_head_logits", None)
        pixel_emb = seg_logits.get("pixel_emb", None)

        if global_pred is not None and globalw > 0:
            global_pred_resized = resize(
                input=global_pred,
                size=seg_label.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            losses["loss_global"] = self.loss_decode(
                global_pred_resized,
                seg_label,
                ignore_index=self.ignore_index,
            ) * globalw

            if global_focusw > 0 and base_pred is not None:
                base_pred_resized = resize(
                    input=base_pred,
                    size=seg_label.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                losses["loss_global_focus"] = self._base_error_focused_ce(
                    logits=global_pred_resized,
                    seg_label=seg_label,
                    base_head_logits=base_pred_resized,
                    err_weight=self.args.get("global_focus_err_weight", 1.0),
                    unc_weight=self.args.get("global_focus_unc_weight", 0.5),
                    use_class_balance=self.args.get("global_focus_class_balance", True),
                ) * global_focusw

        # EMA bank update from clean GT (independent of base errors), train-only.
        # Done after computing the losses so loss_global uses the bank state that
        # forward actually read. Under DDP, broadcast_buffers syncs the bank from
        # rank 0 each step, so the EMA is effectively rank-0 driven (acceptable).
        if pixel_emb is not None:
            self.proto_memory.update(pixel_emb, seg_label)

        return losses
