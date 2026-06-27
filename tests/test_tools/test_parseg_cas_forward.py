# -*- coding: utf-8 -*-
"""CPU smoke test for PARSeg-CAS.

This test builds the new head from its ADE config, runs a real forward pass on
tiny EfficientFormer-like multi-scale features, and checks that the new
confusion-aware attribute separation path contributes a finite margin loss.
"""

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class TestPARSegCASForward(unittest.TestCase):

    def test_cas_files_define_head_and_config(self):
        head_path = REPO / "mmseg/models/decode_heads/PARSegCAS.py"
        cfg_path = REPO / "local_configs/offseg2/Base/parsegcas_ade20k_160k-512x512.py"
        self.assertTrue(head_path.exists(), "PARSegCAS head file is missing")
        self.assertTrue(cfg_path.exists(), "PARSegCAS ADE config is missing")

        head_text = head_path.read_text(encoding="utf-8")
        cfg_text = cfg_path.read_text(encoding="utf-8")
        for token in [
            "class ConfusionAwareAttributeSeparation",
            "class CASRefinementHead",
            "class PARSegCAS",
            "loss_cas_margin",
        ]:
            self.assertIn(token, head_text)
        self.assertIn("type='PARSegCAS'", cfg_text)
        self.assertIn("cas_marginw", cfg_text)

    def test_forward_and_confusion_margin_loss(self):
        try:
            import torch
            from mmengine.config import Config
            from mmengine.structures import PixelData
            from mmseg.registry import MODELS
            from mmseg.structures import SegDataSample
            import mmseg.models.decode_heads.PARSegCAS  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"deps unavailable: {exc}")

        cfg = Config.fromfile(
            str(REPO / "local_configs/offseg2/Base/parsegcas_ade20k_160k-512x512.py")
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
            "cas_refinement_logits",
            "cas_margin_logits",
            "final_logits",
        ]:
            self.assertIn(key, out)
        self.assertEqual(out["final_logits"].shape[:2], (b, 150))
        self.assertEqual(out["cas_refinement_logits"].shape, out["refinement_head_logits"].shape)
        self.assertEqual(out["cas_margin_logits"].shape, out["refinement_head_logits"].shape)

        h = w = 256
        gt = torch.randint(0, 150, (b, h, w))
        samples = []
        for i in range(b):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=gt[i].unsqueeze(0))
            samples.append(sample)

        losses = head.loss_by_feat(out, samples)
        self.assertIn("loss_cas_margin", losses)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value).all(), f"non-finite loss: {name}")


if __name__ == "__main__":
    unittest.main()
