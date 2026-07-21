"""Small residual-stream primitives for hyper-connected vision backbones.

This module deliberately has no MMSegmentation dependency so that the
connection algebra can be tested in isolation.  The implementation follows
the static Hyper-Connections update with one read vector, one write vector,
and one residual-stream mixing matrix per residual sublayer.
"""

import torch
from torch import Tensor, nn


class StaticHyperConnection2d(nn.Module):
    """Static hyper-connection for ``[B, N, C, H, W]`` feature streams.

    The initialization is exactly equivalent to an ordinary residual update
    when all ``N`` streams start from the same feature: one stream is read by
    the residual branch, its output is written to every stream, and the
    residual mixing matrix starts as identity.

    Args:
        rate: Number of persistent residual streams.
        layer_id: Sublayer index used to rotate the initially selected stream.
    """

    def __init__(self, rate: int = 2, layer_id: int = 0):
        super().__init__()
        if rate < 2:
            raise ValueError(f'Hyper-connection rate must be >= 2, got {rate}')

        self.rate = int(rate)
        read_weights = torch.zeros(self.rate)
        read_weights[int(layer_id) % self.rate] = 1.0

        self.read_weights = nn.Parameter(read_weights)
        self.write_weights = nn.Parameter(torch.ones(self.rate))
        self.residual_mix = nn.Parameter(torch.eye(self.rate))

    def expand_streams(self, feature: Tensor) -> Tensor:
        """Replicate a normal ``[B,C,H,W]`` feature into residual streams."""
        if feature.ndim != 4:
            raise ValueError(
                f'Expected a 4D feature tensor, got shape {tuple(feature.shape)}')
        return feature.unsqueeze(1).expand(-1, self.rate, -1, -1, -1)

    def read_streams(self, streams: Tensor) -> Tensor:
        """Mix the persistent streams into one residual-branch input."""
        self._check_streams(streams)
        weights = self.read_weights.to(dtype=streams.dtype)
        return torch.einsum('n,bnchw->bchw', weights, streams)

    def write_streams(self, streams: Tensor, branch_output: Tensor) -> Tensor:
        """Mix old streams and write one computed branch output into them."""
        self._check_streams(streams)
        if branch_output.shape != streams.shape[:1] + streams.shape[2:]:
            raise ValueError(
                'Branch output must match one stream: '
                f'{tuple(branch_output.shape)} vs {tuple(streams.shape)}')

        residual_mix = self.residual_mix.to(dtype=streams.dtype)
        write_weights = self.write_weights.to(dtype=streams.dtype)
        residual = torch.einsum(
            'ij,bjchw->bichw', residual_mix, streams)
        injected = branch_output.unsqueeze(1) * write_weights.view(
            1, self.rate, 1, 1, 1)
        return residual + injected

    def collapse_streams(self, streams: Tensor) -> Tensor:
        """Return one stage output while conserving the stream mean."""
        self._check_streams(streams)
        return streams.mean(dim=1)

    def _check_streams(self, streams: Tensor) -> None:
        if streams.ndim != 5 or streams.shape[1] != self.rate:
            raise ValueError(
                f'Expected [B,{self.rate},C,H,W], got {tuple(streams.shape)}')

    def extra_repr(self) -> str:
        return f'rate={self.rate}'
