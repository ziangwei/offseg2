"""Frequency-anchored U-Mix refinement for hierarchical feature pyramids.

The implementation follows the coarse-to-fine state-update principle of
U-MixFormer, but is written as a small, dependency-light residual branch for
PARSeg3.  The original FreqFusion output remains the anchor feature.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


class _FeatureProjection(nn.Module):
    """Project one accumulated FreqFusion state into the shared mix space."""

    def __init__(self, in_channels: int, mix_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, mix_dim, 1, bias=False)
        self.faumix_norm = nn.GroupNorm(
            _group_count(mix_dim), mix_dim)
        self.act = nn.GELU()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.act(self.faumix_norm(self.proj(feature)))


class _MixAttentionBlock(nn.Module):
    """Update a native-resolution state from pooled multi-scale memory."""

    def __init__(
        self,
        dim: int,
        memory_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(
                f'mix_dim={dim} must be divisible by heads={num_heads}')
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Distinctive names make optimizer rules auditable in the config.
        self.faumix_norm_query = nn.LayerNorm(dim)
        self.faumix_norm_memory = nn.LayerNorm(memory_dim)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(memory_dim, dim, bias=False)
        self.value_proj = nn.Linear(memory_dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        hidden_dim = max(dim, int(dim * mlp_ratio))
        self.faumix_norm_ffn = nn.LayerNorm(dim)
        self.ffn_in = nn.Conv2d(dim, hidden_dim, 1)
        self.ffn_spatial = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            padding=1,
            groups=hidden_dim,
        )
        self.ffn_out = nn.Conv2d(hidden_dim, dim, 1)
        self.ffn_act = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)

    def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tokens.shape
        return tokens.reshape(
            batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query_feature: torch.Tensor,
        memory_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = query_feature.shape
        query_tokens = query_feature.flatten(2).transpose(1, 2)
        memory_tokens = memory_feature.flatten(2).transpose(1, 2)

        query = self._split_heads(
            self.query_proj(self.faumix_norm_query(query_tokens)))
        key = self._split_heads(
            self.key_proj(self.faumix_norm_memory(memory_tokens)))
        value = self._split_heads(
            self.value_proj(self.faumix_norm_memory(memory_tokens)))

        attention = (query @ key.transpose(-2, -1)) * self.scale
        # Computing the normalization in fp32 avoids large fp16 maps becoming
        # numerically brittle while preserving the model's activation dtype.
        attention = torch.softmax(attention.float(), dim=-1).to(query.dtype)
        attention = self.attn_dropout(attention)
        update = attention @ value
        update = update.transpose(1, 2).reshape(batch, -1, channels)
        query_tokens = query_tokens + self.output_dropout(
            self.output_proj(update))

        normalized = self.faumix_norm_ffn(query_tokens)
        normalized = normalized.transpose(1, 2).reshape(
            batch, channels, height, width)
        ffn_update = self.ffn_in(normalized)
        ffn_update = self.ffn_act(
            self.ffn_spatial(ffn_update) + ffn_update)
        ffn_update = self.ffn_dropout(self.ffn_out(ffn_update))
        output = query_tokens.transpose(1, 2).reshape(
            batch, channels, height, width)
        return output + ffn_update


class FreqFusionAnchoredUMix(nn.Module):
    """Coarse-to-fine mix-attention with an exact PARSeg3 identity start.

    Args:
        state_channels: Channels of the saved accumulated states ordered from
            P5 (coarsest) to P2 (finest).
        output_channels: Channels of the original aligned PARSeg3 feature.
        stage_dims: State widths inside the U-Mix branch, coarse to fine.
        num_heads: Per-stage attention head counts, coarse to fine.
        max_scale: Bound applied to the learned per-channel residual gate.
    """

    def __init__(
        self,
        state_channels: Sequence[int],
        output_channels: int,
        stage_dims: Sequence[int] = (256, 128, 64, 32),
        num_heads: Sequence[int] = (8, 4, 2, 1),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_scale: float = 0.25,
    ):
        super().__init__()
        state_channels = tuple(state_channels)
        stage_dims = tuple(stage_dims)
        num_heads = tuple(num_heads)
        if not (len(state_channels) == len(stage_dims) == len(num_heads)):
            raise ValueError(
                'state_channels, stage_dims, and num_heads must have equal length')
        if len(state_channels) < 2:
            raise ValueError('FA-U-Mix requires at least two pyramid states')
        if max_scale <= 0:
            raise ValueError('max_scale must be positive')

        self.num_states = len(state_channels)
        self.max_scale = float(max_scale)
        self.state_projections = nn.ModuleList([
            _FeatureProjection(in_channels, stage_dim)
            for in_channels, stage_dim in zip(state_channels, stage_dims)
        ])
        memory_dim = sum(stage_dims)
        self.mix_blocks = nn.ModuleList([
            _MixAttentionBlock(
                dim=stage_dim,
                memory_dim=memory_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for stage_dim, heads in zip(stage_dims, num_heads)
        ])

        # A signed linear write-back mirrors the original U-Mix fusion and
        # lets the branch correct the anchor in either direction.
        self.branch_fuse = nn.Conv2d(
            memory_dim, output_channels, 1, bias=True)
        # Do not zero the branch itself: a double-zero initialization would
        # prevent the gate from receiving a useful first-step gradient.
        self.faumix_gate = nn.Parameter(torch.zeros(output_channels))

    def _memory(self, states: Sequence[torch.Tensor]) -> torch.Tensor:
        memory_size = states[0].shape[-2:]
        pooled = [
            state if state.shape[-2:] == memory_size else
            F.adaptive_avg_pool2d(state, memory_size)
            for state in states
        ]
        return torch.cat(pooled, dim=1)

    def forward(
        self,
        base_feature: torch.Tensor,
        pyramid_states: Sequence[torch.Tensor],
        return_states: bool = False,
    ):
        if len(pyramid_states) != self.num_states:
            raise ValueError(
                f'Expected {self.num_states} states, got {len(pyramid_states)}')
        states = [
            projection(feature)
            for projection, feature in zip(
                self.state_projections, pyramid_states)
        ]

        # Earlier decoded states replace their encoder-side counterparts in
        # the memory seen by every finer stage.
        for stage, block in enumerate(self.mix_blocks):
            states[stage] = block(states[stage], self._memory(states))

        target_size = base_feature.shape[-2:]
        decoded = [
            state if state.shape[-2:] == target_size else
            F.interpolate(
                state,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
            for state in states
        ]
        residual = self.branch_fuse(torch.cat(decoded, dim=1))
        gate = self.max_scale * torch.tanh(self.faumix_gate)
        output = base_feature + gate.view(1, -1, 1, 1) * residual
        if return_states:
            return output, states
        return output


__all__ = ['FreqFusionAnchoredUMix']
