"""Dependency-free structural checks for the RABA experiment."""

import ast
import copy
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RABA_CONFIG = (REPO_ROOT / 'local_configs' / 'offseg2' / 'Base' /
               'raba_ade20k_160k-512x512.py')
P3_CONFIG = (REPO_ROOT / 'local_configs' / 'offseg2' / 'Base' /
             'parseg3_ade20k_160k-512x512.py')
RABA_MODULE = (REPO_ROOT / 'mmseg' / 'models' / 'decode_heads' /
               'region_attribute_bialign_head.py')


def _merge_dict(parent, child):
    """Small subset of MMEngine's config merge used by these pure configs."""
    result = copy.deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and value.get('_delete_', False):
            result[key] = {
                sub_key: copy.deepcopy(sub_value)
                for sub_key, sub_value in value.items()
                if sub_key != '_delete_'
            }
        elif (isinstance(value, dict) and key in result and
              isinstance(result[key], dict)):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_config(path):
    namespace = {'__file__': str(path)}
    code = compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    exec(code, namespace)
    base_paths = namespace.get('_base_', [])
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    merged = {}
    for base_path in base_paths:
        merged = _merge_dict(merged,
                             _load_config((path.parent / base_path).resolve()))
    current = {
        key: value
        for key, value in namespace.items()
        if not key.startswith('__') and key != '_base_'
    }
    return _merge_dict(merged, current)


class TestRABAScaffold(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raba = _load_config(RABA_CONFIG)
        cls.p3 = _load_config(P3_CONFIG)

    def test_keeps_p3_experiment_protocol(self):
        for key in ('backbone', 'data_preprocessor'):
            self.assertEqual(self.raba['model'][key], self.p3['model'][key])
        for key in ('train_dataloader', 'val_dataloader', 'train_cfg',
                    'param_scheduler'):
            self.assertEqual(self.raba[key], self.p3[key])
        self.assertEqual(
            self.raba['model']['test_cfg'],
            dict(mode='slide', crop_size=(512, 512), stride=(480, 480)))

    def test_replaces_p3_head_without_key_leakage(self):
        head = self.raba['model']['decode_head']
        self.assertEqual(head['type'], 'RegionAttributeBiAlignHead')
        self.assertEqual(head['num_classes'], 150)
        self.assertEqual(head['num_queries'], 100)
        self.assertTrue(head['final_only_loss'])
        self.assertEqual(head['transformer_decoder']['num_layers'], 3)
        self.assertEqual(
            head['pixel_decoder']['type'],
            'mmdet.P3FreqFusionPixelDecoder')
        for stale_key in ('new_channels', 'channels', 'dropout_ratio',
                          'cls_attributes', 'args', 'loss_decode'):
            self.assertNotIn(stale_key, head)
        configured_losses = {
            key for key in head if key in
            {'loss_cls', 'loss_mask', 'loss_dice'}
        }
        self.assertEqual(configured_losses,
                         {'loss_cls', 'loss_mask', 'loss_dice'})

    def test_has_no_language_or_teacher_dependency(self):
        tree = ast.parse(RABA_MODULE.read_text(encoding='utf-8'))
        imports = []
        class_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.ClassDef):
                class_names.add(node.name)
        self.assertTrue({
            'P3FreqFusionPixelDecoder', 'RegionAttributeClassifier',
            'RegionAttributeBiAlignHead'
        }.issubset(class_names))
        forbidden_roots = {'clip', 'open_clip', 'transformers'}
        self.assertFalse(
            any(module.split('.')[0] in forbidden_roots for module in imports))
        self.assertIsNone(self.raba.get('load_from'))


if __name__ == '__main__':
    unittest.main()
