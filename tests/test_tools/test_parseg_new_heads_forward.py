# -*- coding: utf-8 -*-
"""CPU smoke tests for PLCR/CDR/OSC heads.

These tests build each head from config and run a tiny forward/loss path when
torch/mmseg deps are available. They self-skip in lightweight local envs.
"""

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class TestPARSegNewHeadsForward(unittest.TestCase):

    def _deps(self):
        try:
            import torch
            from mmengine.config import Config
            from mmengine.structures import PixelData
            from mmseg.registry import MODELS
            from mmseg.structures import SegDataSample
            import mmseg.models.decode_heads.PARSegPLCR  # noqa: F401
            import mmseg.models.decode_heads.PARSegCDR  # noqa: F401
            import mmseg.models.decode_heads.PARSegOSC  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"deps unavailable: {exc}")
        return torch, Config, PixelData, MODELS, SegDataSample

    def _fake_inputs_and_samples(self, torch, PixelData, SegDataSample, batch=1):
        inputs = [
            torch.randn(batch, 32, 64, 64),
            torch.randn(batch, 64, 32, 32),
            torch.randn(batch, 144, 16, 16),
            torch.randn(batch, 288, 8, 8),
        ]
        gt = torch.randint(0, 150, (batch, 256, 256))
        samples = []
        for i in range(batch):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=gt[i].unsqueeze(0))
            samples.append(sample)
        return inputs, samples

    def _run_head(self, cfg_name, expected_losses):
        torch, Config, PixelData, MODELS, SegDataSample = self._deps()
        cfg = Config.fromfile(str(REPO / "local_configs/offseg2/Base" / cfg_name))
        head = MODELS.build(cfg.model.decode_head)
        head.train()
        inputs, samples = self._fake_inputs_and_samples(torch, PixelData, SegDataSample)
        out = head.forward(inputs)
        self.assertIn("final_logits", out)
        self.assertEqual(out["final_logits"].shape[:2], (1, 150))
        losses = head.loss_by_feat(out, samples)
        for loss_name in expected_losses:
            self.assertIn(loss_name, losses)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value).all(), f"non-finite loss: {name}")

    def test_plcr_forward_loss(self):
        self._run_head(
            "parsegplcr_ade20k_160k-512x512.py",
            ["loss_plcr_aux", "loss_plcr_rank", "loss_palx_margin", "loss_palx_center"],
        )

    def test_cdr_forward_loss(self):
        self._run_head(
            "parsegcdr_ade20k_160k-512x512.py",
            ["loss_cdr_base_rank", "loss_cdr_refinement_rank", "loss_cdr_final_rank"],
        )

    def test_osc_forward_loss(self):
        self._run_head(
            "parsegosc_ade20k_160k-512x512.py",
            ["loss_base", "loss_refinement", "loss_fusion"],
        )


if __name__ == "__main__":
    unittest.main()
