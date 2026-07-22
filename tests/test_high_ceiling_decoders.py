"""Dependency-light checks for FA-U-Mix and Progressive Class Query."""

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
DECODE_ROOT = REPO_ROOT / 'mmseg' / 'models' / 'decode_heads'
FA_CORE = DECODE_ROOT / 'fa_umix.py'
FA_HEAD = DECODE_ROOT / 'PARSegFAUMix.py'
PCQ_CORE = DECODE_ROOT / 'progressive_class_query.py'
PCQ_HEAD = DECODE_ROOT / 'PARSegPCQ.py'
CONFIG_ROOT = REPO_ROOT / 'local_configs' / 'offseg2' / 'Base'
BASE_CONFIG = CONFIG_ROOT / 'parseg3_ade20k_160k-512x512.py'
FA_CONFIG = CONFIG_ROOT / 'parseg3_faumix_ade20k_160k-512x512.py'
PCQ_CONFIG = CONFIG_ROOT / 'parseg3_pcq_ade20k_160k-512x512.py'


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, 'PyTorch is unavailable in the local CI')
class TestFAUMixAlgebra(unittest.TestCase):

    def _module(self):
        core = _load_module('fa_umix_core', FA_CORE)
        return core.FreqFusionAnchoredUMix(
            state_channels=(8, 12, 16, 20),
            output_channels=10,
            stage_dims=(16, 12, 8, 4),
            num_heads=(4, 3, 2, 1),
            mlp_ratio=2.0,
            max_scale=0.25,
        )

    def _inputs(self, requires_grad=False):
        base = torch.randn(1, 10, 16, 16, requires_grad=requires_grad)
        states = [
            torch.randn(1, 8, 2, 2, requires_grad=requires_grad),
            torch.randn(1, 12, 4, 4, requires_grad=requires_grad),
            torch.randn(1, 16, 8, 8, requires_grad=requires_grad),
            torch.randn(1, 20, 16, 16, requires_grad=requires_grad),
        ]
        return base, states

    def test_zero_gate_is_exact_anchor_and_gate_is_trainable(self):
        torch.manual_seed(31)
        module = self._module()
        base, states = self._inputs(requires_grad=True)
        captured = {}
        hook = module.branch_fuse.register_forward_hook(
            lambda _module, _inputs, output: captured.setdefault(
                'residual', output))
        output = module(base, states)
        hook.remove()

        self.assertTrue(torch.equal(output, base))
        residual_probe = captured['residual'].detach()
        (output * residual_probe).sum().backward()
        self.assertTrue(torch.equal(base.grad, residual_probe))
        self.assertTrue(torch.isfinite(module.faumix_gate.grad).all())
        self.assertGreater(module.faumix_gate.grad.abs().sum().item(), 0.0)
        branch_gradient = module.branch_fuse.weight.grad
        self.assertIsNotNone(branch_gradient)
        self.assertEqual(branch_gradient.abs().sum().item(), 0.0)

    def test_nonzero_gate_trains_branch_and_preserves_shape(self):
        torch.manual_seed(37)
        module = self._module()
        with torch.no_grad():
            module.faumix_gate.fill_(0.05)
        base, states = self._inputs(requires_grad=True)
        output, decoded_states = module(
            base, states, return_states=True)
        self.assertEqual(output.shape, base.shape)
        self.assertEqual(
            [state.shape[1] for state in decoded_states], [16, 12, 8, 4])
        self.assertFalse(torch.equal(output, base))
        output.square().mean().backward()
        gradients = [
            parameter.grad
            for name, parameter in module.named_parameters()
            if name != 'faumix_gate'
        ]
        self.assertTrue(any(
            gradient is not None and torch.isfinite(gradient).all() and
            gradient.abs().sum().item() > 0.0
            for gradient in gradients
        ))


@unittest.skipIf(torch is None, 'PyTorch is unavailable in the local CI')
class TestProgressiveClassQueryAlgebra(unittest.TestCase):

    def _module(self, full_size=False):
        core = _load_module('pcq_core', PCQ_CORE)
        if full_size:
            return core.ProgressiveClassQueryUpdater(
                state_channels=(256, 384, 448),
                query_dim=256,
                attention_dim=64,
                num_heads=4,
                mlp_ratio=2.0,
                pool_size=16,
            )
        return core.ProgressiveClassQueryUpdater(
            state_channels=(8, 12, 16),
            query_dim=20,
            attention_dim=8,
            num_heads=2,
            mlp_ratio=2.0,
            pool_size=4,
        )

    def _inputs(self, requires_grad=False):
        queries = torch.randn(2, 5, 20, requires_grad=requires_grad)
        states = [
            torch.randn(2, 8, 2, 3, requires_grad=requires_grad),
            torch.randn(2, 12, 5, 7, requires_grad=requires_grad),
            torch.randn(2, 16, 9, 11, requires_grad=requires_grad),
        ]
        return queries, states

    def test_zero_gates_are_exact_identity_and_receive_gradient(self):
        torch.manual_seed(41)
        module = self._module()
        queries, states = self._inputs(requires_grad=True)
        updates = []
        hook = module.shared_updater.register_forward_hook(
            lambda _module, _inputs, output: updates.append(output))
        output, query_states = module(
            queries, states, return_states=True)
        hook.remove()

        self.assertTrue(torch.equal(output, queries))
        self.assertTrue(all(
            torch.equal(state, queries) for state in query_states))
        probe = sum(update.detach() for update in updates)
        (output * probe).sum().backward()
        self.assertTrue(torch.equal(queries.grad, probe))
        self.assertTrue(torch.isfinite(module.pcq_gates.grad).all())
        for stage_gradient in module.pcq_gates.grad:
            self.assertGreater(stage_gradient.abs().sum().item(), 0.0)
        adapter_gradient = module.stage_adapters[0].proj.weight.grad
        self.assertIsNotNone(adapter_gradient)
        self.assertEqual(adapter_gradient.abs().sum().item(), 0.0)

    def test_nonzero_gates_train_shared_branch_and_pool_rectangles(self):
        torch.manual_seed(43)
        module = self._module()
        with torch.no_grad():
            module.pcq_gates.fill_(0.05)
        queries, states = self._inputs(requires_grad=True)
        output = module(queries, states)
        self.assertEqual(output.shape, queries.shape)
        self.assertFalse(torch.equal(output, queries))
        pooled = module.stage_adapters[-1](states[-1])
        self.assertLessEqual(pooled.shape[-2] * pooled.shape[-1], 16)
        output.square().mean().backward()
        self.assertGreater(
            module.shared_updater.query_proj.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            module.stage_adapters[-1].proj.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_production_updater_stays_below_parameter_budget(self):
        module = self._module(full_size=True)
        parameter_count = sum(
            parameter.numel() for parameter in module.parameters())
        self.assertLess(parameter_count, 500_000)
        self.assertEqual(
            sum(1 for _ in module.shared_updater.modules()
                if isinstance(_, type(module.shared_updater))),
            1,
        )


class TestHighCeilingDecoderScaffold(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base = _load_config(BASE_CONFIG)
        cls.fa = _load_config(FA_CONFIG)
        cls.pcq = _load_config(PCQ_CONFIG)

    def test_heads_are_additive_parseg3_subclasses(self):
        for path, class_name in (
            (FA_HEAD, 'PARSegFAUMix'),
            (PCQ_HEAD, 'PARSegPCQ'),
        ):
            source = path.read_text(encoding='utf-8')
            tree = ast.parse(source)
            classes = {
                node.name: node for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef)
            }
            bases = {
                base.id for base in classes[class_name].bases
                if isinstance(base, ast.Name)
            }
            self.assertIn('PARSeg3', bases)
            self.assertIn('self.freqfusions', source)
            self.assertIn('self.align(', source)
            self.assertIn('self.prototype_attribute_refinement', source)
            self.assertIn('self.fusion(', source)
            self.assertNotIn('loss_by_feat', source)

    def test_configs_preserve_full_parseg3_protocol(self):
        for variant, expected_type in (
            (self.fa, 'PARSegFAUMix'),
            (self.pcq, 'PARSegPCQ'),
        ):
            for key in ('backbone', 'data_preprocessor', 'train_cfg',
                        'test_cfg'):
                if key in self.base['model']:
                    self.assertEqual(
                        variant['model'][key], self.base['model'][key])
                else:
                    self.assertEqual(variant[key], self.base[key])
            for key in ('train_dataloader', 'val_dataloader',
                        'param_scheduler'):
                self.assertEqual(variant[key], self.base[key])
            head = variant['model']['decode_head']
            base_head = self.base['model']['decode_head']
            self.assertEqual(head['type'], expected_type)
            self.assertEqual(head['args'], base_head['args'])
            self.assertEqual(head['loss_decode'], base_head['loss_decode'])
            for key, value in base_head.items():
                if key != 'type':
                    self.assertEqual(head[key], value)

    def test_optimizer_rules_cover_only_new_norms_and_gates(self):
        fa_keys = self.fa['optim_wrapper']['paramwise_cfg']['custom_keys']
        pcq_keys = self.pcq['optim_wrapper']['paramwise_cfg']['custom_keys']
        self.assertEqual(
            fa_keys['faumix_norm'], dict(lr_mult=10.0, decay_mult=0.0))
        self.assertEqual(
            fa_keys['faumix_gate'], dict(lr_mult=10.0, decay_mult=0.0))
        self.assertEqual(
            pcq_keys['pcq_norm'], dict(lr_mult=10.0, decay_mult=0.0))
        self.assertEqual(
            pcq_keys['pcq_gates'], dict(lr_mult=10.0, decay_mult=0.0))

    def test_no_extra_supervision_or_external_semantic_branch(self):
        combined = '\n'.join(path.read_text(encoding='utf-8').lower()
                             for path in (FA_CORE, FA_HEAD,
                                          PCQ_CORE, PCQ_HEAD))
        for forbidden in ('open_clip', 'transformers', 'teacher',
                          'distillation', 'auxiliary_head'):
            self.assertNotIn(forbidden, combined)
        for variant in (self.fa, self.pcq):
            self.assertEqual(
                set(variant['model']['decode_head']['args']),
                set(self.base['model']['decode_head']['args']),
            )


if __name__ == '__main__':
    unittest.main()
