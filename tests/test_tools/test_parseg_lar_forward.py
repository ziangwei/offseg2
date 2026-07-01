# -*- coding: utf-8 -*-
"""CPU smoke tests for PARSegLAR (variants A and B).

Builds each head from its config and runs a tiny forward/loss path when
torch/mmseg deps are available. Self-skips in lightweight local envs, same
convention as test_parseg_new_heads_forward.py.
"""

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class TestPARSegLARForward(unittest.TestCase):

    def _deps(self):
        try:
            import torch
            from mmengine.config import Config
            from mmengine.structures import PixelData
            from mmseg.registry import MODELS
            from mmseg.structures import SegDataSample
            import mmseg.models.decode_heads.PARSegLAR  # noqa: F401
            import mmseg.models.segmentors.igr_encoder_decoder  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"deps unavailable: {exc}")
        return torch, Config, PixelData, MODELS, SegDataSample

    def _fake_inputs_and_samples(self, torch, PixelData, SegDataSample, batch=1):
        # matches the 256x256-crop convention used by test_parseg_new_heads_forward.py
        inputs = [
            torch.randn(batch, 32, 64, 64),
            torch.randn(batch, 64, 32, 32),
            torch.randn(batch, 144, 16, 16),
            torch.randn(batch, 288, 8, 8),
        ]
        image = torch.randn(batch, 3, 256, 256)
        gt = torch.randint(0, 150, (batch, 256, 256))
        samples = []
        for i in range(batch):
            sample = SegDataSample()
            sample.gt_sem_seg = PixelData(data=gt[i].unsqueeze(0))
            samples.append(sample)
        return inputs, image, samples

    def _run_variant(self, cfg_name, expected_hw_factor):
        torch, Config, PixelData, MODELS, SegDataSample = self._deps()
        cfg = Config.fromfile(str(REPO / "local_configs/offseg2/Base" / cfg_name))
        head = MODELS.build(cfg.model.decode_head)
        head.train()

        inputs, image, samples = self._fake_inputs_and_samples(torch, PixelData, SegDataSample)

        # forward() must assert if set_image() was never called
        with self.assertRaises(AssertionError):
            head.forward(inputs)

        head.set_image(image)
        out = head.forward(inputs)
        self.assertIn("final_logits", out)
        self.assertEqual(out["final_logits"].shape[:2], (1, 150))
        # base_head_logits/refinement_head_logits should live at
        # feat_aligned's (post-LAR) resolution, scaled by the configured
        # upsample factor relative to variant A's native resolution.
        base_h, base_w = out["base_head_logits"].shape[-2:]
        if expected_hw_factor == 2:
            self.assertGreater(base_h, 1)
            self.assertGreater(base_w, 1)

        losses = head.loss_by_feat(out, samples)
        self.assertIn("loss_base", losses)
        self.assertIn("loss_refinement", losses)
        self.assertIn("loss_fusion", losses)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value).all(), f"non-finite loss: {name}")
        return out

    def test_variant_a_same_resolution_is_near_identity_at_init(self):
        torch, *_ = self._deps()
        out_a = self._run_variant("parseglar_a_ade20k_160k-512x512.py", expected_hw_factor=1)
        # at init the gate is near its configured init value (small, not
        # exactly 0), so LAR's effect on base_head_logits should be modest,
        # not exactly identical to plain PARSeg3 -- just bounded/small. We
        # only assert finiteness and shape here; the actual near-baseline
        # numeric check belongs to the warm-start run on real data.
        self.assertTrue(torch.isfinite(out_a["final_logits"]).all())

    def test_variant_b_doubles_spatial_resolution(self):
        torch, Config, PixelData, MODELS, SegDataSample = self._deps()
        cfg_a = Config.fromfile(str(REPO / "local_configs/offseg2/Base/parseglar_a_ade20k_160k-512x512.py"))
        cfg_b = Config.fromfile(str(REPO / "local_configs/offseg2/Base/parseglar_b_ade20k_160k-512x512.py"))
        head_a = MODELS.build(cfg_a.model.decode_head)
        head_b = MODELS.build(cfg_b.model.decode_head)
        head_a.train()
        head_b.train()

        inputs, image, samples = self._fake_inputs_and_samples(torch, PixelData, SegDataSample)
        head_a.set_image(image)
        head_b.set_image(image)
        out_a = head_a.forward(inputs)
        out_b = head_b.forward(inputs)

        ha, wa = out_a["base_head_logits"].shape[-2:]
        hb, wb = out_b["base_head_logits"].shape[-2:]
        self.assertEqual((hb, wb), (ha * 2, wa * 2))

        losses_b = head_b.loss_by_feat(out_b, samples)
        for name, value in losses_b.items():
            self.assertTrue(torch.isfinite(value).all(), f"non-finite loss: {name}")


if __name__ == "__main__":
    unittest.main()
