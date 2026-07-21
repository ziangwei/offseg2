"""Dependency-light checks for the Persistent Cross-Scale Hyper-Decoder."""

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
CORE_MODULE = (REPO_ROOT / 'mmseg' / 'models' / 'decode_heads' /
               'persistent_cross_scale.py')
HEAD_MODULE = (REPO_ROOT / 'mmseg' / 'models' / 'decode_heads' /
               'PARSegPCHD.py')
HYPER_CONFIG = (REPO_ROOT / 'local_configs' / 'offseg2' / 'Base' /
                'parseg3_pchd4_hyper_ade20k_160k-512x512.py')
FIXED_CONFIG = (REPO_ROOT / 'local_configs' / 'offseg2' / 'Base' /
                'parseg3_pchd4_fixed_ade20k_160k-512x512.py')
P3_CONFIG = (REPO_ROOT / 'local_configs' / 'offseg2' / 'Base' /
             'parseg3_ade20k_160k-512x512.py')


def _load_core_module():
    spec = importlib.util.spec_from_file_location('pchd_core', CORE_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, 'PyTorch is unavailable in the local CI')
class TestPCHDAlgebra(unittest.TestCase):

    def _features(self):
        return [
            torch.randn(2, 8, 32, 32),
            torch.randn(2, 12, 16, 16),
            torch.randn(2, 16, 8, 8),
            torch.randn(2, 24, 4, 4),
        ]

    def _decoder(self, mode):
        core = _load_core_module()
        return core.PersistentCrossScaleDecoder(
            input_channels=(8, 12, 16, 24),
            output_channels=20,
            stream_channels=8,
            depth=2,
            expand_ratio=1.5,
            kernel_size=3,
            connection_mode=mode,
            mix_bound=0.25,
        )

    def test_hyper_and_fixed_are_initially_elementwise_equivalent(self):
        torch.manual_seed(17)
        hyper = self._decoder('hyper').eval()
        fixed = self._decoder('fixed').eval()
        fixed.load_state_dict(hyper.state_dict(), strict=True)

        features = self._features()
        with torch.no_grad():
            hyper_output, hyper_streams = hyper(
                features, return_streams=True)
            fixed_output, fixed_streams = fixed(
                features, return_streams=True)

        self.assertEqual(hyper_output.shape, (2, 20, 32, 32))
        self.assertEqual(hyper_streams.shape, (2, 4, 8, 16, 16))
        torch.testing.assert_close(
            hyper_streams, fixed_streams, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(
            hyper_output, fixed_output, rtol=0.0, atol=1e-6)

        hyper_features = [
            feature.clone().requires_grad_(True) for feature in features
        ]
        fixed_features = [
            feature.clone().requires_grad_(True) for feature in features
        ]
        hyper(hyper_features).square().mean().backward()
        fixed(fixed_features).square().mean().backward()
        for hyper_feature, fixed_feature in zip(
                hyper_features, fixed_features):
            torch.testing.assert_close(
                hyper_feature.grad,
                fixed_feature.grad,
                rtol=0.0,
                atol=1e-6,
            )

    def test_connection_matrices_start_as_identity_and_keep_row_sum(self):
        hyper = self._decoder('hyper')
        identity = torch.eye(4)
        for connection in hyper.connections:
            for matrix in connection.effective_matrices():
                torch.testing.assert_close(matrix, identity)
                torch.testing.assert_close(
                    matrix.sum(dim=-1), torch.ones(4))

        with torch.no_grad():
            hyper.connections[0].read_delta.normal_()
            hyper.connections[0].state_delta.normal_()
            hyper.connections[0].write_delta.normal_()
        for matrix in hyper.connections[0].effective_matrices():
            torch.testing.assert_close(
                matrix.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=0.0)

    def test_only_hyper_variant_has_trainable_connection_scalars(self):
        hyper = self._decoder('hyper')
        fixed = self._decoder('fixed')
        hyper_routing = sum(
            parameter.numel()
            for name, parameter in hyper.named_parameters()
            if 'connections.' in name)
        fixed_routing = sum(
            parameter.numel()
            for name, parameter in fixed.named_parameters()
            if 'connections.' in name)
        self.assertEqual(hyper_routing, 2 * 3 * 4 * 4)
        self.assertEqual(fixed_routing, 0)

        hyper(self._features()).square().mean().backward()
        for connection in hyper.connections:
            for name in ('read_delta', 'state_delta', 'write_delta'):
                gradient = getattr(connection, name).grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient.abs().max().item(), 0.0)

    def test_connection_indices_are_target_then_source(self):
        core = _load_core_module()
        connection = core.CrossScaleHyperConnection2d(
            rate=4, mode='hyper', mix_bound=0.25)
        with torch.no_grad():
            connection.state_delta[0, 3] = 4.0
        streams = torch.tensor([1.0, 2.0, 4.0, 8.0]).reshape(
            1, 4, 1, 1, 1)
        output = connection(streams, lambda value: torch.zeros_like(value))
        state_matrix = connection.effective_matrices()[1]
        expected_target_zero = torch.dot(state_matrix[0], streams.flatten())
        torch.testing.assert_close(output[0, 0, 0, 0, 0],
                                   expected_target_zero)
        torch.testing.assert_close(output[0, 3, 0, 0, 0],
                                   streams[0, 3, 0, 0, 0])


class TestPCHDScaffold(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p3 = _load_config(P3_CONFIG)
        cls.hyper = _load_config(HYPER_CONFIG)
        cls.fixed = _load_config(FIXED_CONFIG)

    def test_head_is_additive_parseg3_subclass_and_preserves_tail(self):
        tree = ast.parse(HEAD_MODULE.read_text(encoding='utf-8'))
        classes = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        head = classes['PARSegPCHD']
        bases = {base.id for base in head.bases if isinstance(base, ast.Name)}
        self.assertIn('PARSeg3', bases)
        source = HEAD_MODULE.read_text(encoding='utf-8')
        self.assertIn('self.pchd(projected_inputs)', source)
        self.assertIn('self.offset_learning(feat_aligned)', source)
        self.assertIn('self.prototype_attribute_refinement', source)
        self.assertIn('self.fusion(', source)
        self.assertNotIn('loss_by_feat', source)

    def test_configs_change_only_decoder_fusion_topology(self):
        for key in ('backbone', 'data_preprocessor', 'train_cfg',
                    'test_cfg'):
            if key in self.p3['model']:
                self.assertEqual(self.hyper['model'][key], self.p3['model'][key])
            else:
                self.assertEqual(self.hyper[key], self.p3[key])
        for key in ('train_dataloader', 'val_dataloader', 'param_scheduler'):
            self.assertEqual(self.hyper[key], self.p3[key])

        hyper_head = self.hyper['model']['decode_head']
        fixed_head = self.fixed['model']['decode_head']
        self.assertEqual(hyper_head['type'], 'PARSegPCHD')
        self.assertEqual(hyper_head['pchd_mode'], 'hyper')
        self.assertEqual(fixed_head['pchd_mode'], 'fixed')
        for key, value in hyper_head.items():
            if key != 'pchd_mode':
                self.assertEqual(fixed_head[key], value)
        self.assertEqual(hyper_head['loss_decode'],
                         self.p3['model']['decode_head']['loss_decode'])
        self.assertEqual(hyper_head['args'],
                         self.p3['model']['decode_head']['args'])
        custom_keys = self.hyper['optim_wrapper']['paramwise_cfg'][
            'custom_keys']
        for key in ('decode_head.pchd.connections', 'pchd_norm',
                    'context_scale'):
            self.assertEqual(
                custom_keys[key], dict(lr_mult=10.0, decay_mult=0.0))

    def test_no_language_teacher_or_auxiliary_loss_is_added(self):
        combined = (CORE_MODULE.read_text(encoding='utf-8') +
                    HEAD_MODULE.read_text(encoding='utf-8')).lower()
        for forbidden in ('open_clip', 'transformers', 'teacher', 'distill'):
            self.assertNotIn(forbidden, combined)
        inherited_losses = self.hyper['model']['decode_head']['args']
        self.assertEqual(
            set(inherited_losses),
            set(self.p3['model']['decode_head']['args']))


if __name__ == '__main__':
    unittest.main()
