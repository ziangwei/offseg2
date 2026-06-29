# -*- coding: utf-8 -*-
"""Static guardrails for PARSegIGR wiring.

The real IGR forward path needs torch/mmseg. These checks catch cheap config and
interface mistakes before a server run spends GPU time.
"""

import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class TestPARSegIGRScaffold(unittest.TestCase):

    def test_default_config_points_to_known_base_and_uses_power_of_two_points(self):
        cfg = (REPO / "local_configs/offseg2/Base/parseg_igr_ade20k_160k-512x512.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth",
            cfg,
        )
        self.assertIn("subdivision_num_points=8192", cfg)

    def test_head_exposes_final_logits_for_analysis_compatibility(self):
        head = (REPO / "mmseg/models/decode_heads/PARSegIGR.py").read_text(encoding="utf-8")
        self.assertIn("final_logits=coarse", head)
        self.assertIn("loss_point", head)


if __name__ == "__main__":
    unittest.main()
