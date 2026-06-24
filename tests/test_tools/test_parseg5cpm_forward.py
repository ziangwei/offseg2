# -*- coding: utf-8 -*-
"""Real forward + loss smoke test for PARSeg5-CPM (CPU, tiny tensors).

Unlike the static scaffold test, this actually builds the head from its config,
runs a forward pass on dummy multi-scale features, runs loss_by_feat with fake
labels, and checks that the prototype bank gets initialized. Catches shape /
wiring bugs locally before spending server GPU time.

Self-skips if torch / mmseg / mmengine are not installed (e.g. in a CPU-only
scratch environment), so it never blocks the rest of the suite.
"""

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class TestPARSeg5CPMForward(unittest.TestCase):

    def test_forward_loss_and_bank_update(self):
        try:
            import torch
            from mmengine.config import Config
            from mmengine.structures import PixelData
            from mmseg.registry import MODELS
            from mmseg.structures import SegDataSample
            import mmseg.models.decode_heads.PARSeg5CPM  # noqa: F401 (registers head)
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"deps unavailable: {exc}")

        cfg = Config.fromfile(
            str(REPO / "local_configs/offseg2/Base/parseg5cpm_ade20k_160k-512x512.py")
        )
        head = MODELS.build(cfg.model.decode_head)
        head.train()

        # efficientformer-like 4-level features (strides 4/8/16/32) at small size.
        b = 2
        inputs = [
            torch.randn(b, 32, 64, 64),
            torch.randn(b, 64, 32, 32),
            torch.randn(b, 144, 16, 16),
            torch.randn(b, 288, 8, 8),
        ]
        out = head.forward(inputs)
        for key in [
            "base_head_logits",
            "refinement_head_logits",
            "global_logits",
            "pixel_emb",
            "final_logits",
        ]:
            self.assertIn(key, out)
        self.assertEqual(out["final_logits"].shape[0], b)
        self.assertEqual(out["final_logits"].shape[1], 150)

        # Bank starts empty => neutral global evidence.
        self.assertEqual(float(head.proto_memory.proto_bank.abs().sum()), 0.0)

        # Fake full-res labels.
        h = w = 256
        gt = torch.randint(0, 150, (b, h, w))
        samples = []
        for i in range(b):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=gt[i].unsqueeze(0))
            samples.append(sample)

        losses = head.loss_by_feat(out, samples)
        self.assertIn("loss_global", losses)
        self.assertIn("loss_global_focus", losses)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value).all(), f"non-finite loss: {name}")

        # After one update the bank should be populated for the seen classes.
        self.assertGreater(float(head.proto_memory.proto_bank.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
