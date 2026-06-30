# -*- coding: utf-8 -*-
"""CPU smoke test for PARSeg-LCR.

This catches shape / wiring errors that scaffold tests cannot see. It self-skips
when torch/mmengine/mmseg deps are unavailable.
"""

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class TestPARSegLCRForward(unittest.TestCase):

    def test_forward_loss_and_candidate_relation_path(self):
        try:
            import torch
            from mmengine.config import Config
            from mmengine.structures import PixelData
            from mmseg.registry import MODELS
            from mmseg.structures import SegDataSample
            import mmseg.models.decode_heads.PARSegLCR  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"deps unavailable: {exc}")

        cfg = Config.fromfile(
            str(REPO / "local_configs/offseg2/Base/parseglcr_ade20k_160k-512x512.py")
        )
        head = MODELS.build(cfg.model.decode_head)
        head.train()

        b = 1
        inputs = [
            torch.randn(b, 32, 64, 64),
            torch.randn(b, 64, 32, 32),
            torch.randn(b, 144, 16, 16),
            torch.randn(b, 288, 8, 8),
        ]
        out = head.forward(inputs)
        for key in [
            "raw_base_head_logits",
            "base_head_logits",
            "refinement_head_logits",
            "lcr_relation_logits",
            "lcr_delta_logits",
            "lcr_candidate_idx",
            "final_logits",
        ]:
            self.assertIn(key, out)
        self.assertEqual(out["final_logits"].shape[:2], (b, 150))
        self.assertEqual(out["lcr_delta_logits"].shape, out["raw_base_head_logits"].shape)
        self.assertEqual(out["lcr_candidate_idx"].shape[1], 5)

        h = w = 256
        gt = torch.randint(0, 150, (b, h, w))
        samples = []
        for i in range(b):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=gt[i].unsqueeze(0))
            samples.append(sample)

        losses = head.loss_by_feat(out, samples)
        self.assertIn("loss_lcr_aux", losses)
        self.assertIn("loss_lcr_rank", losses)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value).all(), f"non-finite loss: {name}")


if __name__ == "__main__":
    unittest.main()
