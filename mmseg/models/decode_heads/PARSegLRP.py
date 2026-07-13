# -*- coding: utf-8 -*-
"""PARSeg-LRP: language-reparameterized prototype retrieval.

This is the integrated PARSeg3 candidate rather than a replacement decoder:

* FreqFusion, Offset_Learning, PAL attributes, soft routing, AGCF and all
  original PARSeg3 losses are kept.
* The proven LCR branch is kept unchanged.
* Only PGAC's image prototype source is upgraded.  In addition to the
  base-logit-pooled prototype, a class query derived from frozen language
  descriptions retrieves an image-conditioned visual prototype directly from
  the refinement feature map.  It does not pass through another 150-way
  classifier and therefore does not inherit the base prediction mask.

Language is used as retrieval-query geometry, never as the final classifier.
The retrieved value is purely visual and is mixed into the original PGAC
prototype through a bounded, near-baseline gate.  At inference the frozen
description features are constants in the checkpoint; no text encoder and no
image-specific text input are required.  ``fold_language_queries`` can further
materialize the language branch into a plain query buffer before export.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSeg3 import PrototypeAttributeRefinementHead
from .PARSegLCR import PARSegLCR
from .PARSegLTA import resolve_asset_path


def _load_description_set(path, num_classes):
    """Load and normalize a [classes, descriptions, text_dim] asset."""
    resolved = resolve_asset_path(path)
    data = torch.load(resolved, map_location="cpu")
    descriptions = data["embeddings"].float()
    if descriptions.dim() != 3 or descriptions.shape[0] != num_classes:
        raise ValueError(
            "LRP requires description embeddings shaped [C, K, E], but got "
            f"{tuple(descriptions.shape)} for num_classes={num_classes}; "
            f"path={resolved}"
        )
    return F.normalize(descriptions, p=2, dim=-1)


class LanguageReparameterizedPrototypeCalibration(nn.Module):
    """PGAC with an independent language-query visual prototype source.

    The original base-guided prototype remains the identity path.  A compact
    cross-attention block retrieves one additional visual prototype per class
    from a pooled feature grid.  Class descriptions parameterize the queries;
    an unconstrained residual lets them leave the CLIP geometry when visual
    discrimination requires it.
    """

    def __init__(
        self,
        dim,
        num_classes,
        cls_attributes,
        descriptions,
        residual_scale=1.0,
        topk_div=64,
        num_heads=8,
        grid_size=16,
        proto_gate_init=0.10,
        proto_gate_max=0.50,
        center_text=True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        if not (0.0 < proto_gate_init < proto_gate_max <= 1.0):
            raise ValueError(
                "expected 0 < proto_gate_init < proto_gate_max <= 1, got "
                f"{proto_gate_init} and {proto_gate_max}"
            )

        self.dim = dim
        self.num_classes = num_classes
        self.cls_attributes = cls_attributes
        self.residual_scale = residual_scale
        self.topk_div = topk_div
        self.grid_size = int(grid_size)
        self.proto_gate_max = float(proto_gate_max)
        self.center_text = bool(center_text)

        # Frozen offline text features are model constants, not runtime input.
        self.register_buffer("descriptions", descriptions)

        text_dim = descriptions.shape[-1]
        num_descriptions = descriptions.shape[1]
        self.text_proj = nn.Linear(text_dim, dim, bias=False)
        nn.init.orthogonal_(self.text_proj.weight)

        # Preserve the description set instead of committing to a fixed mean.
        # Uniform initialization is neutral; each class may learn which visual
        # attributes in its description set are useful for evidence retrieval.
        self.description_logits = nn.Parameter(
            torch.zeros(num_classes, num_descriptions)
        )
        self.query_residual = nn.Parameter(torch.zeros(num_classes, dim))
        self.query_norm = nn.LayerNorm(dim)

        # This buffer is populated only for deployment/export.  It makes the
        # training-time language reparameterization exactly foldable into an
        # ordinary class-query tensor.
        self.register_buffer("folded_query", torch.zeros(num_classes, dim))
        self.register_buffer(
            "query_is_folded", torch.tensor(False, dtype=torch.bool)
        )

        self.visual_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.visual_norm = nn.LayerNorm(dim)
        self.prototype_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.prototype_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.independent_norm = nn.LayerNorm(dim)

        gate_hidden = max(dim // 4, 16)
        self.prototype_gate = nn.Sequential(
            nn.Linear(dim * 3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.prototype_gate[-1].weight)
        gate_ratio = proto_gate_init / proto_gate_max
        nn.init.constant_(
            self.prototype_gate[-1].bias,
            math.log(gate_ratio / (1.0 - gate_ratio)),
        )

        # Original PARSeg3 PGAC calibration path, kept structurally identical.
        self.proto_proj = nn.Linear(dim, dim)
        hidden = dim // 4
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.uniform_(self.gate_mlp[-1].weight, -0.01, 0.01)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer(
            "max_entropy",
            torch.tensor(math.log(num_classes), dtype=torch.float32),
            persistent=False,
        )

    def _language_query(self):
        descriptions = self.descriptions
        if self.center_text:
            # Remove CLIP's shared language direction.  The common component
            # is unhelpful for class-conditioned retrieval; class differences
            # and the full within-class description set are retained.
            descriptions = descriptions - descriptions.mean(dim=(0, 1), keepdim=True)
            descriptions = F.normalize(descriptions, p=2, dim=-1, eps=1e-6)

        projected = self.text_proj(descriptions)  # [C, K, D]
        weights = F.softmax(self.description_logits, dim=-1).unsqueeze(-1)
        language_anchor = (weights * projected).sum(dim=1)
        return self.query_norm(language_anchor + self.query_residual)

    def materialized_query(self):
        if bool(self.query_is_folded.item()):
            return self.folded_query
        return self._language_query()

    @torch.no_grad()
    def fold_language_queries(self):
        """Fold text projection and residual into ordinary class queries."""
        self.folded_query.copy_(self._language_query())
        self.query_is_folded.fill_(True)
        return self

    @torch.no_grad()
    def unfold_language_queries(self):
        """Return to the trainable language parameterization."""
        self.query_is_folded.fill_(False)
        return self

    def _base_prototype(self, refinement_feats, base_head_logits):
        logits = base_head_logits.detach()
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-6))).sum(
            dim=1, keepdim=True
        )
        max_entropy = self.max_entropy.to(device=logits.device, dtype=logits.dtype)
        confidence = 1.0 - (entropy / (max_entropy + 1e-6)).clamp(0.0, 1.0)
        class_mask = probs * confidence

        mask_flat = class_mask.flatten(2)
        k = max(1, mask_flat.shape[-1] // self.topk_div)
        topk_vals, topk_idx = torch.topk(mask_flat, k=k, dim=-1)
        sparse_mask = torch.zeros_like(mask_flat)
        sparse_mask.scatter_(-1, topk_idx, topk_vals)
        proto_weight = sparse_mask / (sparse_mask.sum(dim=-1, keepdim=True) + 1e-6)

        feat_flat = refinement_feats.flatten(2).transpose(1, 2)
        class_proto = torch.bmm(proto_weight, feat_flat)
        presence = topk_vals.mean(dim=-1, keepdim=True)
        return class_proto, presence

    def _independent_prototype(self, refinement_feats):
        batch_size = refinement_feats.shape[0]
        pooled = F.adaptive_avg_pool2d(
            self.visual_proj(refinement_feats),
            output_size=(self.grid_size, self.grid_size),
        )
        visual_tokens = pooled.flatten(2).transpose(1, 2)
        visual_tokens = self.visual_norm(visual_tokens)

        query = self.materialized_query().unsqueeze(0).expand(batch_size, -1, -1)
        retrieved, _ = self.prototype_attention(
            query=query,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=False,
        )
        # No query residual is added here: the prototype passed to PGAC is a
        # weighted visual value, while language only controls where to look.
        return self.independent_norm(retrieved + self.prototype_ffn(retrieved))

    def forward(self, attr_tokens, refinement_feats, base_head_logits):
        _, _, num_attributes, _ = attr_tokens.shape
        base_proto, presence = self._base_prototype(
            refinement_feats, base_head_logits
        )
        independent_proto = self._independent_prototype(refinement_feats)

        base_unit = F.normalize(base_proto, p=2, dim=-1, eps=1e-6)
        independent_unit = F.normalize(
            independent_proto, p=2, dim=-1, eps=1e-6
        )
        gate_input = torch.cat(
            [base_unit, independent_unit, torch.abs(base_unit - independent_unit)],
            dim=-1,
        )
        proto_gate = self.proto_gate_max * torch.sigmoid(
            self.prototype_gate(gate_input)
        )
        class_proto = base_proto + proto_gate * (
            independent_proto - base_proto
        )

        proto_projected = self.proto_proj(class_proto)
        proto_projected = proto_projected.unsqueeze(2).expand(
            -1, -1, num_attributes, -1
        )
        calibration_input = torch.cat(
            [
                attr_tokens,
                proto_projected,
                torch.abs(attr_tokens - proto_projected),
            ],
            dim=-1,
        )
        calibration_gate = torch.sigmoid(self.gate_mlp(calibration_input))
        calibrated = self.norm(
            attr_tokens
            + self.residual_scale
            * presence.unsqueeze(2)
            * calibration_gate
            * (proto_projected - attr_tokens)
        )
        return calibrated, class_proto


class LanguagePrototypeAttributeRefinementHead(PrototypeAttributeRefinementHead):
    """PARSeg3 refinement head with only its PGAC source replaced."""

    def __init__(
        self,
        in_channels,
        num_classes,
        cls_attributes,
        descriptions,
        mask_dim=256,
        args=None,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            mask_dim=mask_dim,
            args=args,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        a = args or {}
        self.proto_refiner = LanguageReparameterizedPrototypeCalibration(
            dim=mask_dim,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            descriptions=descriptions,
            residual_scale=a["proto_residual_scale"],
            topk_div=a["proto_topk_div"],
            num_heads=int(a.get("lrp_num_heads", 8)),
            grid_size=int(a.get("lrp_grid_size", 16)),
            proto_gate_init=float(a.get("lrp_gate_init", 0.10)),
            proto_gate_max=float(a.get("lrp_gate_max", 0.50)),
            center_text=bool(a.get("lrp_center_text", True)),
        )


@MODELS.register_module()
class PARSegLRP(PARSegLCR):
    """PARSeg3 + LCR + language-reparameterized prototype retrieval."""

    def __init__(
        self,
        in_channels,
        new_channels,
        num_classes,
        cls_attributes,
        args=None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            new_channels=new_channels,
            num_classes=num_classes,
            cls_attributes=cls_attributes,
            args=args,
            **kwargs,
        )
        self.args = args or {}
        desc_path = self.args.get(
            "lrp_desc_path",
            os.path.join(
                "assets", "text_anchors", "ade20k_clip_vitb32_desc6.pt"
            ),
        )
        descriptions = _load_description_set(desc_path, num_classes)

        # Replace only PARSeg3's refinement module.  LCR's forward/loss code
        # and every other inherited PARSeg3 component remain untouched.
        self.prototype_attribute_refinement = (
            LanguagePrototypeAttributeRefinementHead(
                in_channels=self.channels,
                num_classes=num_classes,
                cls_attributes=cls_attributes,
                descriptions=descriptions,
                args=self.args,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        )

    @torch.no_grad()
    def fold_language_queries(self):
        """Materialize LRP queries for a text-free exported checkpoint."""
        self.prototype_attribute_refinement.proto_refiner.fold_language_queries()
        return self
