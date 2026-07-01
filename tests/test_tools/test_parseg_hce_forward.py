# -*- coding: utf-8 -*-
"""CPU smoke tests for PARSegHCE-v2.

Builds the head from its config and runs a tiny forward/loss path when
torch/mmseg deps are available. Self-skips in lightweight local envs, same
convention as test_parseg_lar_forward.py / test_parseg_new_heads_forward.py.
"""

import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class TestPARSegHCEForward(unittest.TestCase):

    def _deps(self):
        try:
            import torch
            from mmengine.config import Config
            from mmengine.structures import PixelData
            from mmseg.registry import MODELS
            from mmseg.structures import SegDataSample
            import mmseg.models.decode_heads.PARSegHCE  # noqa: F401
        except Exception as exc:  # pragma: no cover - env dependent
            self.skipTest(f"deps unavailable: {exc}")
        return torch, Config, PixelData, MODELS, SegDataSample

    def _fake_inputs_and_samples(self, torch, PixelData, SegDataSample, batch=1):
        # matches the 256x256-crop convention used by test_parseg_lar_forward.py
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

    def test_forward_uses_learned_candidate_mask_and_losses_are_finite(self):
        torch, Config, PixelData, MODELS, SegDataSample = self._deps()
        cfg = Config.fromfile(str(REPO / "local_configs/offseg2/Base/parseghce_ade20k_160k-512x512.py"))
        head = MODELS.build(cfg.model.decode_head)
        head.train()

        inputs, samples = self._fake_inputs_and_samples(torch, PixelData, SegDataSample)
        out = head.forward(inputs)

        self.assertIn("final_logits", out)
        self.assertEqual(out["final_logits"].shape[:2], (1, 150))
        self.assertIn("hce_candidate_weights", out)
        self.assertIn("hce_delta_logits", out)
        self.assertEqual(out["hce_candidate_weights"].shape[:2], (1, 150))
        self.assertEqual(out["hce_delta_logits"].shape[:2], (1, 150))

        # at init the gate is small (~0.05/0.30), so the corrected base
        # logits should be close to raw base logits -- a bounded candidate-
        # masked perturbation, not a dense hard override.
        raw = out["raw_base_head_logits"]
        gated = out["base_head_logits"]
        self.assertTrue(torch.isfinite(gated).all())
        self.assertFalse(torch.equal(raw, gated))  # gate is > 0, so *some* delta is applied
        max_abs_delta = (gated - raw).abs().max().item()
        self.assertLess(max_abs_delta, 1.0, "gate should keep the init perturbation small")

        candidate = out["hce_candidate_weights"]
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertTrue(((candidate >= 0) & (candidate <= 1)).all())
        active_per_pixel = (candidate > 0).sum(dim=1)
        self.assertLessEqual(active_per_pixel.max().item(), cfg.model.decode_head.args.hce_candidate_topk)

        losses = head.loss_by_feat(out, samples)
        self.assertIn("loss_base", losses)
        self.assertIn("loss_refinement", losses)
        self.assertIn("loss_fusion", losses)
        self.assertIn("loss_hce_sparsity", losses)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value).all(), f"non-finite loss: {name}")

    def test_zero_candidate_budget_recovers_plain_parseg3_base_path(self):
        # With candidate_topk=0 the learned residual path is exactly masked
        # out. This keeps the init/fallback behavior well-defined without
        # relying on any manual class group.
        torch, Config, PixelData, MODELS, SegDataSample = self._deps()

        cfg = Config.fromfile(str(REPO / "local_configs/offseg2/Base/parseghce_ade20k_160k-512x512.py"))
        head_cfg = dict(cfg.model.decode_head)
        head_cfg['args'] = dict(head_cfg['args'], hce_candidate_topk=0)
        head = MODELS.build(head_cfg)
        head.train()

        inputs, samples = self._fake_inputs_and_samples(torch, PixelData, SegDataSample)
        out = head.forward(inputs)

        delta = out["base_head_logits"] - out["raw_base_head_logits"]
        self.assertTrue(torch.all(delta == 0))
        self.assertTrue(torch.all(out["hce_candidate_weights"] == 0))

    def test_dense_candidate_mode_exercises_sparsity_loss(self):
        # candidate_topk=num_classes keeps all learned relation candidates
        # active, deterministically exercising the sparsity regularizer.
        torch, Config, PixelData, MODELS, SegDataSample = self._deps()

        cfg = Config.fromfile(str(REPO / "local_configs/offseg2/Base/parseghce_ade20k_160k-512x512.py"))
        head_cfg = dict(cfg.model.decode_head)
        num_classes = cfg.model.decode_head['num_classes']
        head_cfg['args'] = dict(head_cfg['args'], hce_candidate_topk=num_classes)
        head = MODELS.build(head_cfg)
        head.train()

        inputs, samples = self._fake_inputs_and_samples(torch, PixelData, SegDataSample)
        out = head.forward(inputs)
        losses = head.loss_by_feat(out, samples)

        self.assertIn("loss_hce_sparsity", losses)
        self.assertTrue(torch.isfinite(losses["loss_hce_sparsity"]).all())
        self.assertGreater(losses["loss_hce_sparsity"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
