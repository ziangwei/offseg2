# -*- coding: utf-8 -*-
"""PARSeg-SDR: scene-discriminative refinement with a GT-anchored teacher.

One-sentence claim
------------------
PARSeg3's scene adaptation is SELF-REFERENTIAL -- the PGAC module pools each
image's class prototypes with `p_base * confidence`, so the base head's
confident errors directly contaminate the prototypes, the PAL refinement head
inherits the base's wrong beliefs, and both heads end up confidently wrong
together. SDR breaks the self-reference during training only: a GT-anchored
teacher branch pools prototypes by ground-truth masks (clean by construction),
the inference branch (the "student" -- the UNCHANGED PARSeg3 graph) is aligned
to the teacher by self-distillation, and two in-scene discriminative losses
train exactly the quantity the probes showed is broken (present-rival margins
< 1 for 44% of errors; absent-FP = 42.6% of errors).

Why this is not another dead family member
-------------------------------------------
* Not post-hoc: nothing is corrected after logits exist; training dynamics of
  the whole model change (from-scratch 160k, NOT a warm-start FT probe --
  warm-start screens structurally cannot see training-time mechanisms).
* Not a new branch reading the same features at inference: the teacher exists
  only when GT is available. Inference graph and parameter set are EXACTLY
  PARSeg3 (zero new parameters, zero inference cost; the state_dict is
  checkpoint-compatible with PARSeg3).
* Grounded in this project's own diagnosis chain: confident errors (probe 2),
  answer already in top-2/3 (probe 3), features separable (probe 5), prototype
  contamination measured directly (CGR probe), and the only two positive
  spikes all year (PALX: GT center alignment, LCR: candidate margins) are
  degenerate versions of exactly these two mechanisms.
* Concurrent external validation: ECAC (arXiv 2510.25174) reaches SOTA on
  ADE20K with a teacher that corrects classifier context using GT and
  distills to a student. Same principle family, different mechanism and
  motivation: they enhance a generic classifier with a dataset-level memory
  bank; SDR repairs the specific self-referential pooling pathway that this
  project's probes identified inside PARSeg3, and adds in-scene margin
  objectives that ECAC does not have. Cite and differentiate.

Losses added on top of the unchanged PARSeg3 recipe (all at 1/4 logit
resolution -- cheap, and interior-focused via a purity mask that excludes
mixed boundary cells):
  1. loss_sdr_teacher : CE on the teacher's refinement logits (trains the
     shared calibration machinery -- proto_proj / gate_mlp / routing /
     decoder -- on clean prototypes).
  2. loss_sdr_kd      : KL(student refinement || teacher refinement.detach());
     pulls the inference path toward GT-anchored decisions without ever
     letting GT into the inference path itself (no train/test input gap).
  3. loss_sdr_rival   : softplus margin pushing the GT-class logit above every
     CO-PRESENT rival class logit on the fused final logits (targets the
     PRESENT-CONF 57.4% error mass).
  4. loss_sdr_absent  : softplus margin pushing ABSENT-class logits below the
     top present-class logit (targets the ABSENT-FP 42.6% error mass). This is
     a TRAINING-time shaping of the decision surface, not an inference-time
     presence prior (the post-hoc version of that idea measured +0.03 and is
     dead).

args (all optional, defaults in __init__):
  sdr_teacherw=1.0, sdr_kdw=0.5, sdr_kd_temp=1.0,
  sdr_rivalw=0.2, sdr_absentw=0.1, sdr_margin=0.5,
  sdr_purity=0.75, sdr_warmup_iters=8000
KD and margin losses ramp linearly from 0 to full weight over
`sdr_warmup_iters` so early CE dominates; teacher CE is on from step 0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .PARSeg3 import PARSeg3


@MODELS.register_module()
class PARSegSDR(PARSeg3):
    """PARSeg3 trained with a GT-anchored teacher branch and in-scene
    discriminative objectives. Inference path is byte-identical to PARSeg3."""

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
        self.sdr_teacherw = float(self.args.get('sdr_teacherw', 1.0))
        self.sdr_kdw = float(self.args.get('sdr_kdw', 0.5))
        self.sdr_kd_temp = float(self.args.get('sdr_kd_temp', 1.0))
        self.sdr_rivalw = float(self.args.get('sdr_rivalw', 0.2))
        self.sdr_absentw = float(self.args.get('sdr_absentw', 0.1))
        self.sdr_margin = float(self.args.get('sdr_margin', 0.5))
        self.sdr_purity = float(self.args.get('sdr_purity', 0.75))
        self.sdr_warmup_iters = int(self.args.get('sdr_warmup_iters', 8000))
        if self.sdr_kd_temp <= 0:
            raise ValueError('sdr_kd_temp must be positive')

        # GT stash for the training-time teacher branch. Set in loss(),
        # cleared in a finally-block; predict()/val never sets it, so the
        # teacher branch cannot leak into inference.
        self._sdr_gt = None
        # Fallback iteration counter (used only if the mmengine MessageHub
        # does not expose 'iter').
        self._sdr_step = 0

    # ------------------------------------------------------------------
    # training entry: stash GT so forward() can build the teacher branch
    # ------------------------------------------------------------------
    def loss(self, inputs, batch_data_samples, train_cfg):
        self._sdr_gt = self._stack_batch_gt(batch_data_samples)  # [B,1,H,W]
        try:
            seg_logits = self.forward(inputs)
            losses = self.loss_by_feat(seg_logits, batch_data_samples)
        finally:
            self._sdr_gt = None
        self._sdr_step += 1
        return losses

    def _sdr_iter(self):
        try:
            from mmengine.logging import MessageHub
            it = MessageHub.get_current_instance().get_info('iter')
            if it is not None:
                return int(it)
        except Exception:
            pass
        return self._sdr_step

    # ------------------------------------------------------------------
    # shared decision tail (routing + cosine logits), exact PARSeg3 replica
    # operating on the SHARED modules of prototype_attribute_refinement
    # ------------------------------------------------------------------
    def _sdr_route_and_logits(self, par, attr_tokens, calibrated_tokens,
                              proto_raw, seg_feats_normed):
        dynamic_route = par.route_mlp(proto_raw.detach())            # [B,Nc,A]
        class_bias = par.route_class_bias.weight.unsqueeze(0)        # [1,Nc,A]
        route_prob = F.softmax(dynamic_route + class_bias, dim=-1)
        if self.args['use_class_prototypes']:
            class_feats = torch.einsum('bcad,bca->bcd', calibrated_tokens, route_prob)
        else:
            class_feats = torch.einsum('bcad,bca->bcd', attr_tokens, route_prob)
        class_feats = F.normalize(class_feats, p=2, dim=-1, eps=1e-6)
        sim = torch.einsum('bhwd,bcd->bchw', seg_feats_normed, class_feats)
        return sim / self.args['tau']

    # ------------------------------------------------------------------
    # teacher branch: GT-routed prototype pooling through the SAME modules
    # ------------------------------------------------------------------
    def _sdr_teacher_branch(self, par, attr_tokens, refinement_feats,
                            seg_feats_normed, gt_seg):
        pgac = par.proto_refiner
        B, Nc, A, D = attr_tokens.shape
        Hf, Wf = refinement_feats.shape[-2:]

        with torch.no_grad():
            y = F.interpolate(gt_seg.float(), size=(Hf, Wf), mode='nearest')
            y = y.squeeze(1).long()                                  # [B,Hf,Wf]
            valid = (y != self.ignore_index) & (y < Nc)
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            onehot = F.one_hot(y_safe, num_classes=Nc)               # [B,Hf,Wf,Nc]
            onehot = onehot.permute(0, 3, 1, 2).to(refinement_feats.dtype)
            onehot = onehot * valid.unsqueeze(1).to(refinement_feats.dtype)
            w = onehot.flatten(2)                                    # [B,Nc,HW]
            area = w.sum(dim=-1, keepdim=True)                       # [B,Nc,1]
            present = area.squeeze(-1) > 0                           # [B,Nc]
            w_norm = w / area.clamp_min(1.0)

        feat_flat = refinement_feats.flatten(2).transpose(1, 2)      # [B,HW,D]
        proto_raw_t = torch.bmm(w_norm, feat_flat)                   # [B,Nc,D]

        proto_t = pgac.proto_proj(proto_raw_t)                       # [B,Nc,D]
        proto_t = proto_t.unsqueeze(2).expand(-1, -1, A, -1)         # [B,Nc,A,D]
        gate_in = torch.cat(
            [attr_tokens, proto_t, torch.abs(attr_tokens - proto_t)], dim=-1)
        gate = torch.sigmoid(pgac.gate_mlp(gate_in))                 # [B,Nc,A,1]
        presence_t = present.to(attr_tokens.dtype)[:, :, None, None]
        calibrated_t = pgac.norm(
            attr_tokens
            + pgac.residual_scale * presence_t * gate * (proto_t - attr_tokens))

        teacher_logits = self._sdr_route_and_logits(
            par, attr_tokens, calibrated_t, proto_raw_t, seg_feats_normed)
        return teacher_logits, y, valid, present

    # ------------------------------------------------------------------
    # forward: PARSeg3 trunk -> decision; teacher branch only in training
    # ------------------------------------------------------------------
    def forward(self, inputs, return_vis=False):
        inputs = self._transform_inputs(inputs)

        new_inputs = [self.pre[i](inputs[i]) for i in range(len(inputs))]
        lowres_feat = new_inputs[-1]
        for hires_feat, freqfusion in zip(new_inputs[:-1][::-1], self.freqfusions):
            _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat)
            b, _, h, w = hires_feat.shape
            lowres_feat = torch.cat(
                [hires_feat.reshape(b * 4, -1, h, w),
                 lowres_feat.reshape(b * 4, -1, h, w)], dim=1).reshape(b, -1, h, w)
        feat_aligned = self.align(lowres_feat)
        return self._forward_from_aligned(feat_aligned)

    def _forward_from_aligned(self, feat_aligned):
        """Everything after `align`. Split out so it can be sanity-tested
        without the carafe/FreqFusion trunk (see tools/sdr_sanity_forward.py)."""
        base_head_logits = self.offset_learning(feat_aligned)

        par = self.prototype_attribute_refinement
        refinement_feats = par.refinement_feat_proj(feat_aligned)    # [B,D,H,W]
        attr_tokens = par.spatial_attribute_decoder(
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits)                       # [B,Nc,A,D]

        seg_feats = refinement_feats.permute(0, 2, 3, 1)             # [B,H,W,D]
        seg_feats_normed = F.normalize(seg_feats, p=2, dim=-1, eps=1e-6)

        # ---- student = native PARSeg3 refinement (model-pooled PGAC) ----
        calibrated_attr_tokens, class_proto = par.proto_refiner(
            attr_tokens=attr_tokens,
            refinement_feats=refinement_feats,
            base_head_logits=base_head_logits)
        refinement_head_logits = self._sdr_route_and_logits(
            par, attr_tokens, calibrated_attr_tokens, class_proto,
            seg_feats_normed)

        fusion_mode = self.args.get('fusion_mode', 'AGCF')
        if fusion_mode == 'AGCF':
            final_logits = self.fusion(base_head_logits, refinement_head_logits)
        elif fusion_mode == 'avg':
            final_logits = 0.5 * (base_head_logits + refinement_head_logits)
        elif fusion_mode == 'catconv':
            final_logits = self.fuse_catconv(
                torch.cat([base_head_logits, refinement_head_logits], dim=1))
        else:
            raise ValueError(f'Unsupported fusion_mode for PARSegSDR: {fusion_mode}')

        returndict = {
            'base_head_logits': base_head_logits,
            'calibrated_attr_tokens': calibrated_attr_tokens,
            'refinement_head_logits': refinement_head_logits,
            'final_logits': final_logits,
        }

        # ---- teacher branch: training only, GT-routed prototypes ----
        if self.training and self._sdr_gt is not None:
            teacher_logits, y_feat, valid_feat, present = self._sdr_teacher_branch(
                par, attr_tokens, refinement_feats, seg_feats_normed,
                self._sdr_gt)
            returndict['sdr_teacher_logits'] = teacher_logits
            returndict['sdr_y_feat'] = y_feat
            returndict['sdr_valid_feat'] = valid_feat
            returndict['sdr_present_feat'] = present
        return returndict

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------
    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)

        teacher_logits = seg_logits.get('sdr_teacher_logits')
        if teacher_logits is None:
            return losses

        y_feat = seg_logits['sdr_y_feat']                            # [B,Hf,Wf]
        valid_feat = seg_logits['sdr_valid_feat']                    # [B,Hf,Wf] bool
        z_s = seg_logits['refinement_head_logits']                   # [B,Nc,Hf,Wf]
        z_f = seg_logits['final_logits']                             # [B,Nc,Hf,Wf]
        B, Nc = teacher_logits.shape[:2]
        device = teacher_logits.device

        if not bool(valid_feat.any()):
            # degenerate crop with no valid GT pixels: keep the graph alive
            # with zero-valued losses instead of producing 0/0 NaNs.
            zero = (teacher_logits.sum() + z_s.sum() + z_f.sum()) * 0.0
            losses['loss_sdr_teacher'] = zero
            losses['loss_sdr_kd'] = zero
            losses['loss_sdr_rival'] = zero
            losses['loss_sdr_absent'] = zero
            return losses

        it = self._sdr_iter()
        ramp = min(1.0, float(it) / max(1, self.sdr_warmup_iters))

        # -- 1. teacher CE (trains the shared calibration machinery on
        #       clean prototypes; active from step 0) --
        y_ce = torch.where(valid_feat, y_feat,
                           torch.full_like(y_feat, self.ignore_index))
        losses['loss_sdr_teacher'] = F.cross_entropy(
            teacher_logits, y_ce,
            ignore_index=self.ignore_index) * self.sdr_teacherw

        # -- 2. self-distillation: student refinement -> teacher refinement --
        t = self.sdr_kd_temp
        p_t = F.softmax(teacher_logits.detach() / t, dim=1)
        logp_s = F.log_softmax(z_s / t, dim=1)
        kl_map = F.kl_div(logp_s, p_t, reduction='none').sum(dim=1)  # [B,Hf,Wf]
        vmask = valid_feat.to(kl_map.dtype)
        loss_kd = (kl_map * vmask).sum() / vmask.sum().clamp_min(1.0)
        losses['loss_sdr_kd'] = loss_kd * (t * t) * self.sdr_kdw * ramp

        # -- purity mask: only clean interior cells receive margin pressure
        #    (avoids pushing margins on genuinely mixed boundary cells) --
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)                          # [B,H,W]
        Hf, Wf = y_feat.shape[-2:]
        Hg, Wg = seg_label.shape[-2:]
        with torch.no_grad():
            if Hg % Hf == 0 and Wg % Wf == 0 and Hg // Hf == Wg // Wf:
                r = Hg // Hf
                rep = y_feat.repeat_interleave(r, dim=1).repeat_interleave(r, dim=2)
                agree = ((seg_label == rep) &
                         (seg_label != self.ignore_index)).to(z_f.dtype)
                purity = F.avg_pool2d(agree.unsqueeze(1), kernel_size=r).squeeze(1)
            else:
                purity = torch.ones_like(y_feat, dtype=z_f.dtype)
            clean = valid_feat & (purity >= self.sdr_purity)          # [B,Hf,Wf]

            # per-image present set from FULL-resolution GT (exact)
            present512 = torch.zeros(B, Nc, dtype=torch.bool, device=device)
            for b in range(B):
                cls = torch.unique(seg_label[b])
                cls = cls[(cls != self.ignore_index) & (cls < Nc)]
                if cls.numel() > 0:
                    present512[b, cls.long()] = True

        y_safe = torch.where(valid_feat, y_feat, torch.zeros_like(y_feat))
        z_y = z_f.gather(1, y_safe.unsqueeze(1))                      # [B,1,Hf,Wf]
        cls_idx = torch.arange(Nc, device=device)[None, :, None, None]
        not_gt = cls_idx != y_safe.unsqueeze(1)                       # [B,Nc,Hf,Wf]
        pres_map = present512[:, :, None, None]
        clean_map = clean.unsqueeze(1)

        # -- 3. in-scene rival margin on the fused decision surface --
        rival_mask = (pres_map & not_gt & clean_map).to(z_f.dtype)
        rival_pen = F.softplus(z_f - z_y + self.sdr_margin)
        losses['loss_sdr_rival'] = (
            (rival_pen * rival_mask).sum() / rival_mask.sum().clamp_min(1.0)
        ) * self.sdr_rivalw * ramp

        # -- 4. absent-class suppression below the top present logit --
        # neg_fill is intentionally moderate (-1e4, not dtype-min): images
        # with an empty present set then yield large-but-FINITE penalties
        # that are zeroed by the mask, instead of inf * 0 = NaN.
        neg_fill = -1e4
        top_present = z_f.masked_fill(~pres_map, neg_fill).amax(dim=1, keepdim=True)
        absent_mask = ((~pres_map) & clean_map).to(z_f.dtype)
        absent_pen = F.softplus((z_f - top_present + self.sdr_margin).clamp(max=5e4))
        losses['loss_sdr_absent'] = (
            (absent_pen * absent_mask).sum() / absent_mask.sum().clamp_min(1.0)
        ) * self.sdr_absentw * ramp

        return losses
