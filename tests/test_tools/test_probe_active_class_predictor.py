import importlib.util
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "probe_active_class_predictor.py"


def load_module():
    spec = importlib.util.spec_from_file_location("probe_active_class_predictor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestProbeActiveClassPredictor(unittest.TestCase):

    def test_presence_label_ignores_ignore_index(self):
        mod = load_module()
        gt = [[0, 1, 255], [2, 2, 255]]
        present = mod.presence_from_label(gt, num_classes=4, ignore_index=255)
        self.assertEqual(present, [1.0, 1.0, 1.0, 0.0])

    def test_select_active_classes_is_recall_safe(self):
        mod = load_module()
        scores = [0.91, 0.41, 0.12, 0.03]
        active = mod.select_active_classes(scores, threshold=0.5, min_classes=3)
        self.assertEqual(active, [True, True, True, False])

    def test_gating_candidate_can_recover_false_positive_pixels(self):
        mod = load_module()
        logits = [
            [[2.0, 2.0], [2.0, 2.0]],
            [[5.0, 1.0], [5.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
        gt = [[0, 0], [0, 0]]
        scores = [0.9, 0.1, 0.05]

        base_pred = mod.argmax_logits(logits)
        gated_pred = mod.predict_with_active_gate(
            logits,
            scores,
            threshold=0.5,
            min_classes=1,
            penalty=float("inf"),
        )

        base_inter, base_union = mod.intersect_union_np(base_pred, gt, 3, 255)
        gate_inter, gate_union = mod.intersect_union_np(gated_pred, gt, 3, 255)
        self.assertLess(mod.mean_iou_np(base_inter, base_union), 1.0)
        self.assertEqual(mod.mean_iou_np(gate_inter, gate_union), 1.0)


if __name__ == "__main__":
    unittest.main()
