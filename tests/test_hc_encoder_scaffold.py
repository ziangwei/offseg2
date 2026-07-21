"""Dependency-light checks for the additive HC encoder experiment."""

import ast
import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from tests.test_raba_scaffold import _load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
HC_PRIMITIVE = (REPO_ROOT / 'mmseg' / 'models' / 'backbones' /
                'hyper_connection.py')
HC_ENCODER = (REPO_ROOT / 'mmseg' / 'models' / 'backbones' /
              'efficientformer_v2_hc.py')
HC_CONFIG = (REPO_ROOT / 'local_configs' / 'offseg2' / 'Base' /
             'parseg3_hc2_s34_ade20k_160k-512x512.py')


def _load_hc_primitive():
    spec = importlib.util.spec_from_file_location('hc_primitive', HC_PRIMITIVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.StaticHyperConnection2d


class TestHCEncoderScaffold(unittest.TestCase):

    @unittest.skipIf(torch is None, 'PyTorch is unavailable in the local CI')
    def test_static_hc_initialization_is_residual_equivalent(self):
        static_hc = _load_hc_primitive()
        unit = static_hc(rate=2, layer_id=1)
        feature = torch.randn(2, 4, 5, 6, requires_grad=True)
        streams = unit.expand_streams(feature)
        branch_input = unit.read_streams(streams)
        branch_output = branch_input.square() + 0.25 * branch_input
        output_streams = unit.write_streams(streams, branch_output)

        expected_feature = feature + feature.square() + 0.25 * feature
        expected_streams = expected_feature.unsqueeze(1).expand_as(
            output_streams)
        torch.testing.assert_close(
            output_streams, expected_streams, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(
            unit.collapse_streams(output_streams),
            expected_feature,
            rtol=0.0,
            atol=1e-6)
        self.assertEqual(sum(p.numel() for p in unit.parameters()), 8)

        output_streams.square().mean().backward()
        for parameter in unit.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(any(
            parameter.grad.abs().max().item() > 0
            for parameter in unit.parameters()))

    @unittest.skipIf(torch is None, 'PyTorch is unavailable in the local CI')
    def test_static_hc_matches_explicit_stream_formula(self):
        static_hc = _load_hc_primitive()
        unit = static_hc(rate=2, layer_id=0)
        with torch.no_grad():
            unit.residual_mix.copy_(torch.tensor([[0.7, -0.2], [0.1, 1.1]]))
            unit.write_weights.copy_(torch.tensor([0.4, -0.3]))

        streams = torch.randn(2, 2, 3, 4, 5)
        branch_output = torch.randn(2, 3, 4, 5)
        actual = unit.write_streams(streams, branch_output)
        expected = torch.empty_like(actual)
        for output_index in range(2):
            expected[:, output_index] = sum(
                unit.residual_mix[output_index, input_index]
                * streams[:, input_index]
                for input_index in range(2))
            expected[:, output_index] += (
                unit.write_weights[output_index] * branch_output)
        torch.testing.assert_close(actual, expected)

    def test_encoder_is_additive_subclass_with_late_stage_hc(self):
        tree = ast.parse(HC_ENCODER.read_text(encoding='utf-8'))
        classes = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        encoder = classes['efficientformerv2_s2_hc2_feat']
        base_names = {
            base.id for base in encoder.bases if isinstance(base, ast.Name)
        }
        self.assertIn('efficientformerv2_s2_feat', base_names)
        source = HC_ENCODER.read_text(encoding='utf-8')
        self.assertIn('self.network', source)
        self.assertIn('self.hc_units', source)
        self.assertNotIn('PARSeg3', source)

    def test_config_changes_backbone_only(self):
        namespace = {}
        exec(compile(
            HC_CONFIG.read_text(encoding='utf-8'), str(HC_CONFIG), 'exec'),
             namespace)
        self.assertEqual(
            namespace['_base_'],
            ['./parseg3_ade20k_160k-512x512.py'])
        self.assertEqual(
            namespace['model'],
            dict(backbone=dict(
                type='efficientformerv2_s2_hc2_feat',
                hc_stages=(2, 3),
                hc_rate=2)))
        self.assertNotIn('decode_head', namespace['model'])
        self.assertEqual(
            namespace['optim_wrapper']['paramwise_cfg']['custom_keys'],
            {'hc_units': dict(lr_mult=1.0, decay_mult=0.0)})

        resolved = _load_config(HC_CONFIG)
        self.assertEqual(resolved['model']['decode_head']['type'], 'PARSeg3')
        self.assertEqual(
            resolved['model']['backbone']['init_cfg']['checkpoint'],
            'pretrained/eformer_v2/eformer_s2_450.pth')
        self.assertEqual(
            resolved['optim_wrapper']['paramwise_cfg']['custom_keys']['head'],
            dict(lr_mult=10.0))


if __name__ == '__main__':
    unittest.main()
