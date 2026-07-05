# -*- coding: utf-8 -*-
"""PARSeg-LTA: LCR with frozen language-anchored class embeddings (switch A).

Hypothesis: the candidate-relation scorer needs INTER-CLASS relation
knowledge; a free 150x64 embedding learned from ADE's 20k images
under-determines that geometry (one plausible root of the wall-family
collapse), while a pretrained language space provides it for free.

Design (minimal interference -- the only design law that has survived):
  * PARSegLCR v1 byte-identical (forward, candidate set, losses, weights),
    EXCEPT the relation scorer's `class_embed` is swapped for
        class_vec[c] = W @ E_text[c] + res_scale * r[c]
    with E_text a FROZEN buffer of CLIP text anchors (generated offline by
    tools/gen_text_anchors.py), W a learnable 512->relation_dim projection
    and r a zero-initialized bounded residual. The parent normalizes class
    vectors, so only the DIRECTION (language geometry) matters.
  * zero new losses, zero inference-time text: the anchors are constants
    baked into the checkpoint (registered buffer). No text model exists in
    the training graph either -- only this lookup table.

args (new): lta_anchor_path='assets/text_anchors/ade20k_clip_vitb32.pt',
            lta_res_scale=0.1
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .PARSegLCR import PARSegLCR


def resolve_asset_path(path):
    """Resolve an asset path: absolute -> cwd-relative -> repo-root-relative."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    cand = os.path.join(repo_root, path)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(
        f"text anchor asset not found: '{path}' (tried cwd and {repo_root}). "
        "Generate it once with: python tools/gen_text_anchors.py "
        "(offline, writes assets/text_anchors/ade20k_clip_vitb32.pt; commit "
        "with git add -f because .gitignore excludes *.pt)")


def load_text_anchors(path, num_classes):
    """Load frozen text anchors -> L2-normalized [num_classes, E] tensor."""
    p = resolve_asset_path(path)
    d = torch.load(p, map_location='cpu')
    emb = d['embeddings'].float()
    if emb.shape[0] != num_classes:
        raise ValueError(
            f'anchor asset has {emb.shape[0]} classes, model expects '
            f'{num_classes} ({p})')
    return F.normalize(emb, p=2, dim=-1), d


class AnchoredClassEmbed(nn.Module):
    """Drop-in replacement for the scorer's nn.Embedding.

    Lookup table = learnable projection of FROZEN language anchors plus a
    small zero-initialized residual. Callers that L2-normalize the output
    (LCR does) therefore start from pure language geometry.
    """

    def __init__(self, anchor_path, num_classes, out_dim, res_scale=0.1):
        super().__init__()
        anchors, _ = load_text_anchors(anchor_path, num_classes)
        self.register_buffer('anchors', anchors)              # [C, E] frozen
        self.proj = nn.Linear(anchors.shape[-1], out_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.residual = nn.Parameter(torch.zeros(num_classes, out_dim))
        self.res_scale = float(res_scale)

    def matrix(self):
        return self.proj(self.anchors) + self.res_scale * self.residual

    def forward(self, idx):
        return F.embedding(idx, self.matrix())


@MODELS.register_module()
class PARSegLTA(PARSegLCR):
    """PARSegLCR with language-anchored class embeddings. Nothing else."""

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
        self.lcr.class_embed = AnchoredClassEmbed(
            anchor_path=self.args.get(
                'lta_anchor_path',
                'assets/text_anchors/ade20k_clip_vitb32.pt'),
            num_classes=num_classes,
            out_dim=int(self.args.get('lcr_dim', 64)),
            res_scale=float(self.args.get('lta_res_scale', 0.1)),
        )
