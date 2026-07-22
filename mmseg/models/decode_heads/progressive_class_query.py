"""Progressive class-query updates over accumulated FreqFusion states."""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


class _PooledStageAdapter(nn.Module):
    """Pool before channel projection to keep high-resolution cost bounded."""

    def __init__(self, in_channels: int, attention_dim: int, pool_size: int):
        super().__init__()
        self.pool_size = int(pool_size)
        self.proj = nn.Conv2d(
            in_channels, attention_dim, kernel_size=1, bias=False)
        self.spatial = nn.Conv2d(
            attention_dim,
            attention_dim,
            kernel_size=3,
            padding=1,
            groups=attention_dim,
            bias=False,
        )
        self.pcq_norm = nn.GroupNorm(
            _group_count(attention_dim), attention_dim)
        self.act = nn.GELU()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        height, width = feature.shape[-2:]
        target_size = (
            min(height, self.pool_size),
            min(width, self.pool_size),
        )
        if feature.shape[-2:] != target_size:
            feature = F.adaptive_avg_pool2d(feature, target_size)
        feature = self.proj(feature)
        feature = self.spatial(feature) + feature
        return self.act(self.pcq_norm(feature))


class _SharedClassQueryUpdate(nn.Module):
    """One parameter-shared class-to-image cross-attention update."""

    def __init__(
        self,
        query_dim: int,
        attention_dim: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        if attention_dim % num_heads:
            raise ValueError(
                f'attention_dim={attention_dim} must be divisible by '
                f'num_heads={num_heads}')
        self.num_heads = int(num_heads)
        self.head_dim = attention_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.pcq_norm_query = nn.LayerNorm(query_dim)
        self.pcq_norm_memory = nn.LayerNorm(attention_dim)
        self.query_proj = nn.Linear(query_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(
            attention_dim, attention_dim, bias=False)
        self.value_proj = nn.Linear(
            attention_dim, attention_dim, bias=False)
        self.output_proj = nn.Linear(attention_dim, query_dim, bias=False)

        hidden_dim = max(query_dim, int(query_dim * mlp_ratio))
        self.pcq_norm_ffn = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, query_dim),
        )

    def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, channels = tokens.shape
        return tokens.reshape(
            batch, length, self.num_heads, channels // self.num_heads
        ).transpose(1, 2)

    def forward(
        self,
        class_queries: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        query = self._split_heads(
            self.query_proj(self.pcq_norm_query(class_queries)))
        normalized_visual = self.pcq_norm_memory(visual_tokens)
        key = self._split_heads(self.key_proj(normalized_visual))
        value = self._split_heads(self.value_proj(normalized_visual))

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention.float(), dim=-1).to(query.dtype)
        context = attention @ value
        context = context.transpose(1, 2).reshape(
            class_queries.shape[0], class_queries.shape[1], -1)
        context = self.output_proj(context)
        # The gate outside this module controls the entire candidate update;
        # the branch itself must remain non-zero so that gate gradients live.
        update = context + self.ffn(
            self.pcq_norm_ffn(class_queries + context))
        return update


class ProgressiveClassQueryUpdater(nn.Module):
    """Evolve one class state from P5 through fused P4 and fused P3.

    Only the per-stage, per-channel gates start at zero.  Consequently this
    module is exactly the identity at initialization while each gate receives
    a first-step gradient.  The attention updater is shared across stages.
    """

    def __init__(
        self,
        state_channels: Sequence[int],
        query_dim: int,
        attention_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        pool_size: int = 16,
        max_scale: float = 0.25,
    ):
        super().__init__()
        state_channels = tuple(state_channels)
        if not state_channels:
            raise ValueError('PCQ requires at least one visual state')
        if pool_size < 1:
            raise ValueError('pool_size must be positive')
        if max_scale <= 0:
            raise ValueError('max_scale must be positive')

        self.num_stages = len(state_channels)
        self.max_scale = float(max_scale)
        self.stage_adapters = nn.ModuleList([
            _PooledStageAdapter(channels, attention_dim, pool_size)
            for channels in state_channels
        ])
        # Exactly one updater is reused at every scale.  This makes the class
        # trajectory progressive rather than three unrelated predictions.
        self.shared_updater = _SharedClassQueryUpdate(
            query_dim=query_dim,
            attention_dim=attention_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.pcq_gates = nn.Parameter(
            torch.zeros(self.num_stages, query_dim))

    def effective_gates(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.pcq_gates)

    def forward(
        self,
        class_queries: torch.Tensor,
        pyramid_states: Sequence[torch.Tensor],
        return_states: bool = False,
    ):
        if len(pyramid_states) != self.num_stages:
            raise ValueError(
                f'Expected {self.num_stages} states, got '
                f'{len(pyramid_states)}')

        query_states = [class_queries]
        queries = class_queries
        gates = self.effective_gates()
        for stage, (adapter, feature) in enumerate(zip(
                self.stage_adapters, pyramid_states)):
            visual = adapter(feature).flatten(2).transpose(1, 2)
            update = self.shared_updater(queries, visual)
            gate = gates[stage].view(1, 1, -1)
            queries = queries + gate * update
            query_states.append(queries)

        if return_states:
            return queries, query_states
        return queries


__all__ = ['ProgressiveClassQueryUpdater']
