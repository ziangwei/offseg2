import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "tools" / "analyze_parseg3_confusions.py"
    spec = importlib.util.spec_from_file_location("analyze_parseg3_confusions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logits_from_predictions(pred, gt, num_classes):
    logits = np.full((num_classes,) + pred.shape, -3.0, dtype=np.float32)
    for y in range(pred.shape[0]):
        for x in range(pred.shape[1]):
            cls = int(pred[y, x])
            label = int(gt[y, x])
            logits[cls, y, x] = 4.0
            if cls != label and label < num_classes:
                logits[label, y, x] = 3.8
    return logits


class TestAnalyzePARSeg3Confusions(unittest.TestCase):

    def test_summarize_confusion_arrays_tracks_top_pairs_and_self_confident_wrong(self):
        mod = _load_module()
        gt = np.array(
            [
                [0, 0, 1, 1],
                [0, 2, 2, 1],
                [3, 3, 2, 1],
            ],
            dtype=np.int64,
        )
        final_pred = gt.copy()
        final_pred[0, 0] = 1
        final_pred[0, 1] = 1
        final_pred[1, 1] = 1
        final_pred[2, 0] = 2
        final_logits = _logits_from_predictions(final_pred, gt, 4)
        final_logits[3, 2, 0] = -2.0  # GT is not in final top-2 for this wrong pixel.
        final_logits[1, 2, 0] = 3.8

        base_pred = final_pred.copy()
        base_logits = _logits_from_predictions(base_pred, gt, 4)
        base_logits[:, 0, 0] = -3.0
        base_logits[1, 0, 0] = 9.0  # high-confidence wrong, GT not in top-2.
        base_logits[2, 0, 0] = 1.0

        refine_pred = gt.copy()
        refine_pred[0, 1] = 1
        refine_pred[1, 1] = 1
        refine_logits = _logits_from_predictions(refine_pred, gt, 4)

        summary = mod.summarize_confusion_arrays(
            base_logits=base_logits,
            refine_logits=refine_logits,
            final_logits=final_logits,
            target=gt,
            self_conf_threshold=0.90,
            margin_threshold=0.50,
        )

        self.assertEqual(summary["valid_px"], 12)
        self.assertEqual(summary["final_wrong_px"], 4)
        self.assertEqual(summary["final_wrong_top2_gt_hit_px"], 3)
        self.assertEqual(summary["base_self_confident_wrong_px"], 1)
        self.assertEqual(summary["base_self_confident_wrong_gt_top2_px"], 0)
        self.assertEqual(summary["pair_counts"][(0, 1)], 2)
        self.assertEqual(summary["pair_final_top2_gt_hit"][(0, 1)], 2)
        self.assertEqual(summary["pair_base_same_pred"][(0, 1)], 2)
        self.assertEqual(summary["pair_refine_same_pred"][(0, 1)], 1)

        pairs = mod.top_confusion_pairs(
            summary,
            class_names=["bg", "wall", "chair", "table"],
            topk=2,
        )
        self.assertEqual(pairs[0]["gt"], 0)
        self.assertEqual(pairs[0]["pred"], 1)
        self.assertEqual(pairs[0]["gt_name"], "bg")
        self.assertEqual(pairs[0]["pred_name"], "wall")
        self.assertAlmostEqual(pairs[0]["top2_rate"], 1.0)
        self.assertAlmostEqual(pairs[0]["base_same_pred_rate"], 1.0)
        self.assertAlmostEqual(pairs[0]["refine_same_pred_rate"], 0.5)

    def test_top_failed_classes_uses_class_wrong_rate_then_wrong_count(self):
        mod = _load_module()
        summary = {
            "class_valid": {0: 3, 1: 4, 2: 3, 3: 2},
            "class_final_wrong": {0: 2, 1: 0, 2: 1, 3: 1},
            "class_final_top2_gt_hit": {0: 2, 2: 1, 3: 0},
            "class_base_self_confident_wrong": {0: 1, 3: 1},
        }

        classes = mod.top_failed_classes(
            summary,
            class_names=["bg", "wall", "chair", "table"],
            topk=3,
        )

        self.assertEqual([item["cls"] for item in classes], [0, 3, 2])
        self.assertEqual(classes[0]["name"], "bg")
        self.assertAlmostEqual(classes[0]["wrong_rate"], 2 / 3)
        self.assertAlmostEqual(classes[0]["top2_rate"], 1.0)
        self.assertAlmostEqual(classes[1]["self_conf_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
