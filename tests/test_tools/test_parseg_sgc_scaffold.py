import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEAD = ROOT / "mmseg" / "models" / "decode_heads" / "PARSegSGC.py"
FULL_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parsegsgc_ade20k_160k-512x512.py"
FT_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parsegsgc_ft_ade20k_160k-512x512.py"


class TestPARSegSGCScaffold(unittest.TestCase):

    def test_head_uses_positive_spatial_gate_anchored_to_parseg_final(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("class PARSegSGC(PARSeg3)", text)
        self.assertIn("class SelectiveCorrectionGate", text)
        self.assertIn('out["parseg_final_logits"] = out["final_logits"]', text)
        self.assertIn("gate = self.sgc_gate_max * torch.sigmoid(gate_logits)", text)
        self.assertIn('gds_delta = gds["logits"] - out["parseg_final_logits"].detach()', text)
        self.assertIn('out["final_logits"] = out["parseg_final_logits"] + gate * gds_delta', text)
        self.assertNotIn("torch.tanh(self", text)

    def test_selector_is_trained_by_gds_beats_parseg_ce_target(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("def _selector_target", text)
        self.assertIn("parseg_ce", text)
        self.assertIn("gds_ce", text)
        self.assertRegex(text, r"gds_ce\s*<\s*parseg_ce\s*-\s*self\.sgc_selector_margin")
        self.assertIn("binary_cross_entropy_with_logits", text)

    def test_freeze_api_and_configs_define_ft_gate(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("set_parseg_base_requires_grad", text)
        self.assertIn("set_parseg_base_train_mode", text)

        full = FULL_CFG.read_text(encoding="utf-8")
        ft = FT_CFG.read_text(encoding="utf-8")

        self.assertIn("mmseg.models.decode_heads.PARSegSGC", full)
        self.assertIn("type='PARSegSGC'", full)
        self.assertIn("sgc_freeze_parseg=False", full)
        self.assertIn("train_cfg = dict(val_interval=8000)", full)

        self.assertIn("mmseg.models.segmentors.gds_encoder_decoder", ft)
        self.assertIn("type='GDSEncoderDecoder'", ft)
        self.assertIn("sgc_freeze_parseg=True", ft)
        self.assertIn("load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'", ft)
        self.assertIn("max_iters = 40000", ft)
        self.assertIn("val_interval=8000", ft)


if __name__ == "__main__":
    unittest.main()
