# -*- coding: utf-8 -*-
"""OffSeg-SRG: reasoning over the semantic reconstruction residual.

The current CCM posterior and image-adaptive class centres reconstruct the
part of each pixel already explained by the classifier.  A compact GloRe-like
interaction graph is built only from the unexplained residual, at stride 8,
then broadcast once to refine the same feature path.  No second classifier,
fusion gate, memory bank, or extra loss is introduced.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .OffSegCCM import OffSegCCM


class SemanticResidualRegionGraph(nn.Module):
    """Project a residual field to latent regions, reason, and broadcast."""

    def __init__(self, channels: int, num_nodes: int = 32,
                 node_channels: int = 64, norm_groups: int = 32):
        super().__init__()
        if channels % norm_groups != 0:
            raise ValueError(
                f'channels={channels} must be divisible by '
                f'norm_groups={norm_groups}')
        self.channels = int(channels)
        self.num_nodes = int(num_nodes)
        self.node_channels = int(node_channels)

        self.input_norm = nn.GroupNorm(norm_groups, channels)
        self.assignment = nn.Conv2d(channels, num_nodes, 1, bias=True)
        self.value = nn.Conv2d(channels, node_channels, 1, bias=False)

        # Data-dependent region relations.  This is the graph reasoning step,
        # not a spatial attention branch: its vertices are the pooled latent
        # residual regions, and the result is reprojected through the same
        # assignment map.
        self.node_norm = nn.LayerNorm(node_channels)
        self.query = nn.Linear(node_channels, node_channels, bias=False)
        self.key = nn.Linear(node_channels, node_channels, bias=False)
        self.state = nn.Linear(node_channels, node_channels, bias=False)
        self.update = nn.Sequential(
            nn.LayerNorm(node_channels),
            nn.Linear(node_channels, 2 * node_channels),
            nn.GELU(),
            nn.Linear(2 * node_channels, node_channels),
        )

        self.out_proj = nn.Conv2d(node_channels, channels, 1, bias=False)
        # Exact CCM at step 0.  Unlike a zero scalar gate, a zero output
        # projection itself receives gradients immediately.
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, residual):
        """Args: residual [B,C,H,W], normally at stride 8."""
        batch, _, height, width = residual.shape
        encoded = self.input_norm(residual)

        assignment_logits = self.assignment(encoded).flatten(2)  # [B,N,L]
        pool = assignment_logits.softmax(dim=-1)                  # spatial
        dispatch = assignment_logits.softmax(dim=1)               # per pixel
        values = self.value(encoded).flatten(2).transpose(1, 2)   # [B,L,D]

        nodes = torch.bmm(pool, values)                            # [B,N,D]
        norm_nodes = self.node_norm(nodes)
        query = self.query(norm_nodes)
        key = self.key(norm_nodes)
        relation = torch.bmm(query, key.transpose(1, 2))
        relation = relation / math.sqrt(self.node_channels)
        relation = relation.softmax(dim=-1)                       # [B,N,N]
        reasoned = nodes + self.update(
            torch.bmm(relation, self.state(norm_nodes)))

        field = torch.bmm(dispatch.transpose(1, 2), reasoned)      # [B,L,D]
        field = field.transpose(1, 2).contiguous().view(
            batch, self.node_channels, height, width)
        return self.out_proj(field), relation


@MODELS.register_module()
class OffSegCCMSRG(OffSegCCM):
    """CCM refined once by a graph of its semantic reconstruction residual."""

    def __init__(self, in_channels, new_channels, num_classes,
                 srg_nodes=32, srg_channels=64, srg_stride=2,
                 srg_norm_groups=32, srg_detach_reconstruction=True,
                 **kwargs):
        super().__init__(in_channels=in_channels, new_channels=new_channels,
                         num_classes=num_classes, **kwargs)
        self.srg_stride = int(srg_stride)
        if self.srg_stride < 1:
            raise ValueError('srg_stride must be >= 1')
        self.srg_detach_reconstruction = bool(srg_detach_reconstruction)
        self.srg = SemanticResidualRegionGraph(
            channels=self.channels,
            num_nodes=int(srg_nodes),
            node_channels=int(srg_channels),
            norm_groups=int(srg_norm_groups))

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        feat_aligned = self._build_feature(inputs)
        masks, centres, feat, (height, width) = self._offset_learning_parts(
            feat_aligned)
        batch, classes, _ = masks.shape

        context_logits = masks.detach() if self.ccm_detach_context else masks
        context_centres = (centres.detach()
                           if self.ccm_detach_context else centres)
        metric_feat, gain = self.ccm(
            feat, context_centres, context_logits)

        ccm_raw = metric_feat @ centres.transpose(1, 2)
        ccm_logits = self.offset_learning.mask_norm(ccm_raw)       # [B,HW,K]
        posterior = ccm_logits.softmax(dim=-1)
        reconstruction_centres = centres
        if self.srg_detach_reconstruction:
            posterior = posterior.detach()
            reconstruction_centres = reconstruction_centres.detach()
        reconstruction = torch.bmm(posterior, reconstruction_centres)
        residual = metric_feat - reconstruction                    # [B,HW,C]
        residual_map = residual.transpose(1, 2).contiguous().view(
            batch, self.channels, height, width)

        if self.srg_stride > 1:
            graph_input = F.avg_pool2d(
                residual_map,
                kernel_size=self.srg_stride,
                stride=self.srg_stride,
                ceil_mode=True)
        else:
            graph_input = residual_map
        delta, relation = self.srg(graph_input)
        if delta.shape[-2:] != (height, width):
            delta = F.interpolate(
                delta, size=(height, width), mode='bilinear',
                align_corners=self.align_corners)

        refined = metric_feat + delta.flatten(2).transpose(1, 2)
        final = self.offset_learning.mask_norm(
            refined @ centres.transpose(1, 2))
        final = final.permute(0, 2, 1).contiguous().view(
            batch, classes, height, width)

        return dict(
            stage1_logits=masks.view(batch, classes, height, width),
            final_logits=final,
            ccm_gain=gain,
            srg_delta=delta,
            srg_residual=residual,
            srg_relation=relation)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        losses = super().loss_by_feat(seg_logits, batch_data_samples)
        losses['acc_srg_move'] = seg_logits['srg_delta'].abs().mean().detach()
        losses['acc_srg_residual'] = (
            seg_logits['srg_residual'].square().mean().sqrt().detach())
        return losses
