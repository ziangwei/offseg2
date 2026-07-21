"""Persistent cross-scale streams and bounded hyper-connections.

This module deliberately has no MMSegmentation dependency.  It keeps four
semantic-scale streams alive at a common spatial resolution and lets every
context block learn how to read, preserve, and write across them.  A fixed
mode uses the exact same projections and context blocks with identity
connections, providing a compute-matched residual control.
"""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int, preferred: int = 32) -> int:
    """Return the largest useful GroupNorm divisor up to ``preferred``."""
    groups = min(int(preferred), int(channels))
    while channels % groups:
        groups -= 1
    return groups


class CrossScaleHyperConnection2d(nn.Module):
    """Read/process/write update for persistent ``[B,S,C,H,W]`` streams.

    Each effective connection is a row-sum-preserving perturbation of the
    identity::

        M = I + bound * (softmax(delta, dim=-1) - 1 / S)

    Zero logits therefore produce the identity exactly.  In ``fixed`` mode,
    the deltas are buffers rather than parameters and all three matrices stay
    at identity.  Consequently hyper and fixed variants are elementwise
    equivalent at initialization while differing only in connection topology
    during training.
    """

    def __init__(
        self,
        rate: int = 4,
        mode: str = 'hyper',
        mix_bound: float = 0.25,
    ) -> None:
        super().__init__()
        if rate < 2:
            raise ValueError(f'Connection rate must be >= 2, got {rate}')
        if mode not in {'hyper', 'fixed'}:
            raise ValueError(
                f"Connection mode must be 'hyper' or 'fixed', got {mode!r}")
        if not 0.0 < mix_bound <= 1.0:
            raise ValueError(
                f'mix_bound must lie in (0, 1], got {mix_bound}')

        self.rate = int(rate)
        self.mode = mode
        self.mix_bound = float(mix_bound)
        self.register_buffer('identity', torch.eye(self.rate), persistent=False)

        for name in ('read_delta', 'state_delta', 'write_delta'):
            value = torch.zeros(self.rate, self.rate)
            if mode == 'hyper':
                self.register_parameter(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)

    def _effective_matrix(self, delta: Tensor) -> Tensor:
        if self.mode == 'fixed':
            return self.identity
        centered = F.softmax(delta, dim=-1) - (1.0 / self.rate)
        return self.identity + self.mix_bound * centered

    def effective_matrices(self):
        """Return the current read, state, and write matrices."""
        return (
            self._effective_matrix(self.read_delta),
            self._effective_matrix(self.state_delta),
            self._effective_matrix(self.write_delta),
        )

    def forward(self, streams: Tensor, processor: nn.Module) -> Tensor:
        if streams.ndim != 5 or streams.shape[1] != self.rate:
            raise ValueError(
                f'Expected [B,{self.rate},C,H,W], got {tuple(streams.shape)}')

        read_matrix, state_matrix, write_matrix = (
            matrix.to(dtype=streams.dtype)
            for matrix in self.effective_matrices()
        )
        branch_inputs = torch.einsum(
            'ij,bjchw->bichw', read_matrix, streams)
        batch, rate, channels, height, width = branch_inputs.shape
        branch_outputs = processor(
            branch_inputs.reshape(batch * rate, channels, height, width))
        branch_outputs = branch_outputs.reshape(
            batch, rate, channels, height, width)

        preserved_state = torch.einsum(
            'ij,bjchw->bichw', state_matrix, streams)
        written = torch.einsum(
            'ij,bjchw->bichw', write_matrix, branch_outputs)
        return preserved_state + written

    def extra_repr(self) -> str:
        return (
            f'rate={self.rate}, mode={self.mode!r}, '
            f'mix_bound={self.mix_bound}')


class SharedScaleContext(nn.Module):
    """One lightweight context branch shared by all scale streams."""

    def __init__(
        self,
        channels: int,
        expand_ratio: float = 2.0,
        kernel_size: int = 5,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                f'kernel_size must be odd and >= 3, got {kernel_size}')
        hidden_channels = max(channels, int(round(channels * expand_ratio)))
        self.pchd_norm = nn.GroupNorm(_group_count(channels), channels)
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.expand = nn.Conv2d(channels, hidden_channels, 1)
        self.act = nn.GELU()
        self.project = nn.Conv2d(hidden_channels, channels, 1)
        self.context_scale = nn.Parameter(
            torch.full((channels,), float(layer_scale_init)))

    def forward(self, feature: Tensor) -> Tensor:
        feature = self.pchd_norm(feature)
        feature = self.depthwise(feature)
        feature = self.project(self.act(self.expand(feature)))
        return feature * self.context_scale.view(1, -1, 1, 1)


class _Projection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pchd_norm = nn.GroupNorm(
            _group_count(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, feature: Tensor) -> Tensor:
        return self.act(self.pchd_norm(self.conv(feature)))


class PersistentCrossScaleDecoder(nn.Module):
    """Align four pyramid levels, keep them persistent, then fuse at stride 4.

    P2 is rearranged with PixelUnshuffle instead of pooled, P3 defines the
    working resolution, and P4/P5 are interpolated upward.  All four streams
    remain distinct through every block.  A direct P2 detail path is appended
    only at the final readout before projecting to the PARSeg3 decision width.
    """

    def __init__(
        self,
        input_channels: Sequence[int],
        output_channels: int = 256,
        stream_channels: int = 64,
        depth: int = 4,
        expand_ratio: float = 2.0,
        kernel_size: int = 5,
        connection_mode: str = 'hyper',
        mix_bound: float = 0.25,
        layer_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        if len(input_channels) != 4:
            raise ValueError(
                'PersistentCrossScaleDecoder requires exactly four pyramid '
                f'levels, got {len(input_channels)}')
        if depth < 1:
            raise ValueError(f'depth must be >= 1, got {depth}')

        self.input_channels = tuple(int(c) for c in input_channels)
        self.output_channels = int(output_channels)
        self.stream_channels = int(stream_channels)
        self.depth = int(depth)
        self.connection_mode = connection_mode

        self.p2_rearrange = nn.PixelUnshuffle(2)
        projection_inputs = (
            self.input_channels[0] * 4,
            self.input_channels[1],
            self.input_channels[2],
            self.input_channels[3],
        )
        self.stream_projections = nn.ModuleList([
            _Projection(channels, self.stream_channels)
            for channels in projection_inputs
        ])
        self.processors = nn.ModuleList([
            SharedScaleContext(
                channels=self.stream_channels,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(self.depth)
        ])
        self.connections = nn.ModuleList([
            CrossScaleHyperConnection2d(
                rate=4,
                mode=connection_mode,
                mix_bound=mix_bound,
            )
            for _ in range(self.depth)
        ])

        self.detail_projection = _Projection(
            self.input_channels[0], self.stream_channels)
        self.output_projection = _Projection(
            self.stream_channels * 5, self.output_channels)

    def _initial_streams(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != 4:
            raise ValueError(f'Expected four features, got {len(features)}')
        if any(feature.ndim != 4 for feature in features):
            raise ValueError('Every pyramid feature must have shape [B,C,H,W]')

        target_size = features[1].shape[-2:]
        projected = [
            self.stream_projections[0](self.p2_rearrange(features[0])),
            self.stream_projections[1](features[1]),
            self.stream_projections[2](features[2]),
            self.stream_projections[3](features[3]),
        ]
        projected = [
            feature if feature.shape[-2:] == target_size else F.interpolate(
                feature,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
            for feature in projected
        ]
        return torch.stack(projected, dim=1)

    def forward(self, features: Sequence[Tensor], return_streams: bool = False):
        streams = self._initial_streams(features)
        for connection, processor in zip(self.connections, self.processors):
            streams = connection(streams, processor)

        detail_size = features[0].shape[-2:]
        readout_streams = [
            F.interpolate(
                streams[:, index],
                size=detail_size,
                mode='bilinear',
                align_corners=False,
            )
            for index in range(streams.shape[1])
        ]
        detail = self.detail_projection(features[0])
        output = self.output_projection(
            torch.cat([detail] + readout_streams, dim=1))
        if return_streams:
            return output, streams
        return output
