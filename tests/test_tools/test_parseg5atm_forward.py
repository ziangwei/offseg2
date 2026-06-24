# -*- coding: utf-8 -*-
"""Real forward + loss smoke test for PARSeg5-ATM.

This uses unittest, not pytest. It self-skips when torch/mmengine/mmseg are not
available, so lightweight local checks can still run, while the training server
can execute the real build/forward/loss path before a long run.
"""

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class TestPARSeg5ATMForward(unittest.TestCase):

    def test_forward_loss_and_memory_update(self):
        try:
            import torch
            from mmengine.config import Config
            from mmengine.structures import PixelData
            from mmseg.registry import MODELS
            from mmseg.structures import SegDataSample
            import mmseg.models.decode_heads.PARSeg5ATM  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"deps unavailable: {exc}")

        cfg = Config.fromfile(
            str(REPO / "local_configs/offseg2/Base/parseg5atm_ade20k_160k-512x512.py")
        )
        head = MODELS.build(cfg.model.decode_head)
        head.train()

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
            "atm_logits",
            "raw_attr_tokens",
            "memory_gate",
            "final_logits",
        ]:
            self.assertIn(key, out)
        self.assertEqual(out["final_logits"].shape[0], b)
        self.assertEqual(out["final_logits"].shape[1], 150)
        self.assertEqual(out["raw_attr_tokens"].shape[1:3], (150, 12))

        memory = head.prototype_attribute_refinement.attr_memory
        self.assertEqual(float(memory.memory_count.sum()), 0.0)

        h = w = 256
        gt = torch.randint(0, 150, (b, h, w))
        samples = []
        for i in range(b):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=gt[i].unsqueeze(0))
            samples.append(sample)

        losses = head.loss_by_feat(out, samples)
        self.assertIn("loss_atm", losses)
        self.assertIn("loss_atm_focus", losses)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value).all(), f"non-finite loss: {name}")

        self.assertGreater(float(memory.memory_count.sum()), 0.0)
        self.assertGreater(float(memory.memory_token.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
