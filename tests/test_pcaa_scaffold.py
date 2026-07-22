"""Dependency-light structural checks for PARSeg-PCAA."""

import ast
import unittest
from pathlib import Path

from tests.test_raba_scaffold import _load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
HEAD = REPO_ROOT / 'mmseg' / 'models' / 'decode_heads' / 'PARSegPCAA.py'
CONFIG_ROOT = REPO_ROOT / 'local_configs' / 'offseg2' / 'Base'
BASE_CONFIG = CONFIG_ROOT / 'parseg3_ade20k_160k-512x512.py'
PCAA_CONFIG = CONFIG_ROOT / 'parseg3_pcaa_ade20k_160k-512x512.py'


class TestPCAAScaffold(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = HEAD.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.source)
        cls.base = _load_config(BASE_CONFIG)
        cls.pcaa = _load_config(PCAA_CONFIG)

    def test_is_a_registered_parseg3_replacement_tail(self):
        classes = {
            node.name: node for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef)
        }
        bases = {
            base.id for base in classes['PARSegPCAA'].bases
            if isinstance(base, ast.Name)
        }
        self.assertIn('PARSeg3', bases)
        self.assertIn('@MODELS.register_module()', self.source)
        self.assertIn('self.freqfusions', self.source)
        self.assertIn('self.align(', self.source)

    def test_attribute_forward_has_no_base_branch_input(self):
        classes = {
            node.name: node for node in ast.walk(self.tree)
            if isinstance(node, ast.ClassDef)
        }
        for class_name in (
            'BaseFreeSpatialAttributeDecoder',
            'IndependentAttributeBranch',
        ):
            forward = next(
                node for node in classes[class_name].body
                if isinstance(node, ast.FunctionDef) and node.name == 'forward')
            arguments = [argument.arg for argument in forward.args.args]
            self.assertNotIn('base_logits', arguments)
            self.assertNotIn('base_head_logits', arguments)

        self.assertNotIn('PrototypeGuidedAttributeCalibration', self.source)
        self.assertNotIn('SpatialValueWeighting(', self.source)

    def test_config_preserves_training_protocol(self):
        for key in ('backbone', 'data_preprocessor', 'train_cfg', 'test_cfg'):
            self.assertEqual(
                self.pcaa['model'][key], self.base['model'][key])
        for key in ('train_dataloader', 'val_dataloader', 'param_scheduler'):
            self.assertEqual(self.pcaa[key], self.base[key])

        head = self.pcaa['model']['decode_head']
        base_head = self.base['model']['decode_head']
        self.assertEqual(head['type'], 'PARSegPCAA')
        self.assertEqual(head['in_channels'], base_head['in_channels'])
        self.assertEqual(head['new_channels'], base_head['new_channels'])
        self.assertEqual(head['channels'], base_head['channels'])
        self.assertEqual(head['num_classes'], 150)
        self.assertEqual(head['cls_attributes'], 12)
        self.assertEqual(head['loss_decode'], base_head['loss_decode'])
        self.assertEqual(head['args']['refinement_focusw'], 0.0)
        self.assertEqual(head['pcaa_fod_weight'], 0.01)

    def test_one_new_loss_and_no_external_model(self):
        loss_keys = {
            node.value for node in ast.walk(self.tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith('loss_')
        }
        self.assertEqual(loss_keys, {'loss_pcaa_fod'})

        lowered = self.source.lower()
        for forbidden in (
            'open_clip', 'transformers', 'stable diffusion',
            'distillation', 'teacher', 'text_anchor',
        ):
            self.assertNotIn(forbidden, lowered)

    def test_gate_and_fod_are_stable_by_construction(self):
        self.assertIn('nn.init.zeros_(self.coefficient_predictor.weight)',
                      self.source)
        self.assertIn('base_feature.detach().float()', self.source)
        self.assertIn("coefficient[:, 0:1] * base_logits", self.source)
        self.assertIn("coefficient[:, 1:2] * attribute_logits", self.source)
        optimizer_keys = self.pcaa['optim_wrapper'][
            'paramwise_cfg']['custom_keys']
        self.assertEqual(
            optimizer_keys['pcaa_norm'],
            dict(lr_mult=10.0, decay_mult=0.0))


if __name__ == '__main__':
    unittest.main()
