"""Stage-local Hyper-Connections for EfficientFormerV2-S2.

The original backbone and its state-dict hierarchy remain untouched.  This
subclass only overrides the execution of the two deep semantic stages, so an
ordinary EfficientFormerV2-S2 ImageNet checkpoint still loads every original
parameter under the same key.
"""

from typing import Callable, Dict, Sequence

from torch import Tensor, nn

from mmseg.registry import MODELS

from .efficientformer_v2 import (AttnFFN, FFN,
                                 efficientformerv2_s2_feat)
from .hyper_connection import StaticHyperConnection2d


@MODELS.register_module()
class efficientformerv2_s2_hc2_feat(efficientformerv2_s2_feat):
    """EfficientFormerV2-S2 with stage-persistent static HC streams.

    Only same-resolution residual sublayers inside ``hc_stages`` use HC.
    Streams are expanded at each selected stage entrance and mean-collapsed
    at its exit; all downsampling blocks and exported feature shapes are
    therefore byte-compatible in structure with the original backbone.
    """

    def __init__(self,
                 hc_stages: Sequence[int] = (2, 3),
                 hc_rate: int = 2,
                 **kwargs):
        # The parent loads the original pretrained checkpoint before the new
        # HC parameters are attached, avoiding false missing-key warnings.
        super().__init__(**kwargs)

        self.hc_stages = tuple(sorted(set(int(i) for i in hc_stages)))
        if not self.hc_stages:
            raise ValueError('At least one HC stage must be selected')
        if min(self.hc_stages) < 0 or max(self.hc_stages) >= len(
                self.out_indices):
            raise ValueError(
                f'hc_stages must index {len(self.out_indices)} stages, '
                f'got {self.hc_stages}')

        self.hc_rate = int(hc_rate)
        self._hc_network_to_stage: Dict[int, int] = {
            self.out_indices[stage_index]: stage_index
            for stage_index in self.hc_stages
        }
        self.hc_units = nn.ModuleDict()
        self._build_hc_units()

    @staticmethod
    def _unit_key(stage_index: int, block_index: int, branch: str) -> str:
        return f's{stage_index}_b{block_index}_{branch}'

    def _build_hc_units(self) -> None:
        for stage_index in self.hc_stages:
            network_index = self.out_indices[stage_index]
            stage = self.network[network_index]
            sublayer_id = 0
            for block_index, block in enumerate(stage):
                branches = ('attn', 'ffn') if isinstance(
                    block, AttnFFN) else ('ffn', )
                if not isinstance(block, (FFN, AttnFFN)):
                    raise TypeError(
                        f'Unsupported block in HC stage: {type(block).__name__}')
                for branch in branches:
                    key = self._unit_key(stage_index, block_index, branch)
                    self.hc_units[key] = StaticHyperConnection2d(
                        rate=self.hc_rate, layer_id=sublayer_id)
                    sublayer_id += 1

    @staticmethod
    def _residual_branch(block: nn.Module,
                         branch_input: Tensor,
                         branch: str) -> Tensor:
        if branch == 'attn':
            output = block.token_mixer(branch_input)
            if block.use_layer_scale:
                output = block.layer_scale_1 * output
        elif branch == 'ffn':
            output = block.mlp(branch_input)
            if block.use_layer_scale:
                output = block.layer_scale_2 * output
        else:
            raise ValueError(f'Unknown residual branch: {branch}')
        return block.drop_path(output)

    def _apply_hc(self,
                  streams: Tensor,
                  unit: StaticHyperConnection2d,
                  branch_fn: Callable[[Tensor], Tensor]) -> Tensor:
        branch_input = unit.read_streams(streams)
        branch_output = branch_fn(branch_input)
        return unit.write_streams(streams, branch_output)

    def _forward_hc_stage(self,
                          feature: Tensor,
                          stage: nn.Sequential,
                          stage_index: int) -> Tensor:
        first_key = self._unit_key(stage_index, 0, 'ffn')
        if first_key not in self.hc_units:
            first_key = self._unit_key(stage_index, 0, 'attn')
        streams = self.hc_units[first_key].expand_streams(feature)

        for block_index, block in enumerate(stage):
            if isinstance(block, AttnFFN):
                key = self._unit_key(stage_index, block_index, 'attn')
                streams = self._apply_hc(
                    streams,
                    self.hc_units[key],
                    lambda x, current=block: self._residual_branch(
                        current, x, 'attn'))

            key = self._unit_key(stage_index, block_index, 'ffn')
            streams = self._apply_hc(
                streams,
                self.hc_units[key],
                lambda x, current=block: self._residual_branch(
                    current, x, 'ffn'))

        return self.hc_units[first_key].collapse_streams(streams)

    def forward_tokens(self, x: Tensor):
        outs = []
        for network_index, module in enumerate(self.network):
            stage_index = self._hc_network_to_stage.get(network_index)
            if stage_index is None:
                x = module(x)
            else:
                x = self._forward_hc_stage(x, module, stage_index)

            if self.fork_feat and network_index in self.out_indices:
                outs.append(x)
        if self.fork_feat:
            return outs
        return x
