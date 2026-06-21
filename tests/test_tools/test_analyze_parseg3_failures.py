import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "tools" / "analyze_parseg3_failures.py"
    spec = importlib.util.spec_from_file_location("analyze_parseg3_failures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logits_from_predictions(pred, gt, num_classes):
    logits = np.zeros((num_classes,) + pred.shape, dtype=np.float32)
    for y in range(pred.shape[0]):
        for x in range(pred.shape[1]):
            cls = int(pred[y, x])
            label = int(gt[y, x])
            logits[cls, y, x] = 3.0
            if cls != label and label < num_classes:
                logits[label, y, x] = 2.9
    return logits


class TestAnalyzePARSeg3Failures(unittest.TestCase):

    def test_summarize_prediction_arrays_separates_resolution_and_self_confident_errors(self):
        mod = _load_module()
        gt = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 0, 0, 1],
            ],
            dtype=np.int64,
        )
        base_pred = gt.copy()
        base_pred[0, 0] = 1  # interior, self-confident wrong
        base_pred[0, 1] = 1  # boundary, GT is top-2
        base_pred[1, 2] = 0  # boundary, GT is top-2
        base_pred[2, 0] = 0  # one-pixel small object, GT is top-2

        base_logits = _logits_from_predictions(base_pred, gt, 3)
        base_logits[:, 0, 0] = 0.0
        base_logits[1, 0, 0] = 7.0  # high-confidence wrong, GT not in top-2
        base_logits[2, 0, 0] = 1.0

        refine_pred = gt.copy()
        refine_pred[1, 2] = 0
        refine_pred[2, 0] = 0
        refine_logits = _logits_from_predictions(refine_pred, gt, 3)

        summary = mod.summarize_prediction_arrays(
            base_logits=base_logits,
            refine_logits=refine_logits,
            final_logits=base_logits,
            target=gt,
            self_conf_threshold=0.90,
            margin_threshold=0.50,
            small_component_max_px=1,
        )

        self.assertEqual(summary["valid_px"], 12)
        self.assertEqual(summary["base_wrong_px"], 4)
        self.assertEqual(summary["final_wrong_px"], 4)
        self.assertEqual(summary["base_self_confident_wrong_px"], 1)
        self.assertEqual(summary["base_top2_gt_hit_px"], 3)
        self.assertEqual(summary["final_wrong_top2_gt_hit_px"], 3)
        self.assertEqual(summary["boundary_wrong_px"], 3)
        self.assertEqual(summary["small_wrong_px"], 1)
        self.assertEqual(summary["resolution_wrong_px"], 3)
        self.assertEqual(summary["interior_large_wrong_px"], 1)
        self.assertEqual(summary["bw_ro_px"], 2)
        self.assertEqual(summary["bw_rw_px"], 2)

    def test_connected_component_small_mask_uses_components_not_class_frequency(self):
        mod = _load_module()
        labels = np.array(
            [
                [4, 4, 0, 0],
                [4, 0, 0, 5],
                [0, 0, 5, 5],
                [6, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        small = mod.connected_component_small_mask(labels, ignore_index=255, max_px=2)

        self.assertTrue(bool(small[3, 0]))
        self.assertFalse(bool(small[0, 0]))
        self.assertFalse(bool(small[1, 3]))


if __name__ == "__main__":
    unittest.main()
