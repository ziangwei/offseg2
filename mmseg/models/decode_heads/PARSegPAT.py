# -*- coding: utf-8 -*-
"""PARSeg-PAT: grounding PAL attribute tokens in language-described
attributes. Training-only; forward and inference are EXACTLY PARSeg3.

Why this is the one text hypothesis still standing (see experiment log):
  * name-level text failed with a measured mechanism (CLIP name cone: class
    vectors collapse, LTA 46.95 / PTA 46.97) and name similarity is only
    weakly related to visual confusability (confusion pairs sit at the
    ~82nd percentile of text similarity, 8/25 in text top-5);
  * so names are the wrong granularity. DESCRIPTIONS carry what names lack:
    discriminative attribute content ("hinged panel with a handle" is what
    separates door from wall, not the word "door");
  * PAL's cls_attributes tokens are the natural landing site: they are the
    thesis's own probabilistic attributes, currently anonymous free
    parameters. PAT gives them language-described attribute targets at
    training time. Precedent: PALX's GT-anchoring of PAL geometry produced
    the year's rare positive spikes -- the PAL pathway tolerates gentle
    anchoring.

Mechanism (ONE loss, nothing else):
  * asset: [150, K, 512] frozen CLIP embeddings of K=6 attribute
    descriptions per class (tools/gen_text_descriptions.py; per-description
    embeddings, never averaged -- averaging would re-create the cone);
  * per image, for GT-PRESENT classes only: the class's A calibrated
    attribute tokens are scored against every class's description SET by
      score(c') = mean_a max_k cos(token_a, P(desc_{c',k}))
    (each token matches its best-fitting description -- soft assignment,
    tokens are free to specialize on different attributes);
  * InfoNCE over classes: the present class's own description set must win.
    The discriminative form is deliberate -- LTC showed pure geometry
    adoption is harmful while discriminative alignment self-protects
    against cone collapse. Negatives are implicit (all other classes'
    descriptions); no explicit contrast is encoded in the text itself.
  * P is a learnable 512->token_dim projection; text stays frozen. No text
    model in the training graph, none at inference. Zero forward changes.

args (new): pat_desc_path='assets/text_anchors/ade20k_clip_vitb32_desc6.pt',
            pat_w=0.15, pat_tau=0.1, pat_warmup_iters=8000
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSeg3 import PARSeg3
from .PARSegLTA import resolve_asset_path


@MODELS.register_module()
class PARSegPAT(PARSeg3):
    """PARSeg3 + description-grounded attribute tokens (training only)."""

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
        self.pat_w = float(self.args.get('pat_w', 0.15))
        self.pat_tau = float(self.args.get('pat_tau', 0.1))
        self.pat_warmup_iters = int(self.args.get('pat_warmup_iters', 8000))
        if self.pat_tau <= 0:
            raise ValueError('pat_tau must be positive')

        path = resolve_asset_path(self.args.get(
            'pat_desc_path',
            os.path.join('assets', 'text_anchors',
                         'ade20k_clip_vitb32_desc6.pt')))
        d = torch.load(path, map_location='cpu')
        emb = d['embeddings'].float()                     # [C, K, E]
        if emb.dim() != 3 or emb.shape[0] != num_classes:
            raise ValueError(
                f'description asset shape {tuple(emb.shape)} does not match '
                f'num_classes={num_classes} (need [C, K, E])')
        emb = F.normalize(emb, p=2, dim=-1)
        self.register_buffer('pat_desc', emb)             # frozen text

        # token dim = PAL mask_dim (256 in this codebase)
        token_dim = int(self.args.get('pat_token_dim', 256))
        self.pat_proj = nn.Linear(emb.shape[-1], token_dim, bias=False)
        nn.init.normal_(self.pat_proj.weight, std=0.02)

        self._pat_step = 0

    def _pat_iter(self):
        try:
            from mmengine.logging import MessageHub
            it = MessageHub.get_current_instance().get_info('iter')
            if it is not None:
                return int(it)
        except Exception:
            pass
        return self._pat_step

    # forward: fully inherited from PARSeg3 (zero changes).

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        self._pat_step += 1

        if self.pat_w <= 0:
            return losses
        tokens = seg_logits.get('calibrated_attr_tokens')
        if tokens is None:
            return losses

        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)              # [B, H, W]

        B, C, A, D = tokens.shape
        device = tokens.device

        with torch.no_grad():
            present = torch.zeros(B, C, dtype=torch.bool, device=device)
            for b in range(B):
                cls = torch.unique(seg_label[b])
                cls = cls[(cls != self.ignore_index) & (cls < C)]
                if cls.numel() > 0:
                    present[b, cls.long()] = True
        rows = present.view(-1).nonzero(as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            losses['loss_pat_ground'] = tokens.sum() * 0.0
            return losses

        # present-class token sets  [P, A, D]
        z = tokens.reshape(B * C, A, D).index_select(0, rows)
        z = F.normalize(z, p=2, dim=-1, eps=1e-6)
        target = (rows % C).long()                        # class id per row

        # projected description sets  [C, K, D]
        t = F.normalize(self.pat_proj(self.pat_desc), p=2, dim=-1, eps=1e-6)

        # score(p, c') = mean_a max_k cos(z[p,a], t[c',k])
        sim = torch.einsum('pad,ckd->pack', z, t)         # [P, A, C, K]
        score = sim.max(dim=-1).values.mean(dim=1)        # [P, C]

        ramp = min(1.0, float(self._pat_iter()) / max(1, self.pat_warmup_iters))
        losses['loss_pat_ground'] = (
            F.cross_entropy(score / self.pat_tau, target)
            * self.pat_w * ramp)
        return losses
