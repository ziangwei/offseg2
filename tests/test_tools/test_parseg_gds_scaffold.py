import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEAD = ROOT / "mmseg" / "models" / "decode_heads" / "PARSegGDS.py"
SEGMENTOR = ROOT / "mmseg" / "models" / "segmentors" / "gds_encoder_decoder.py"
FULL_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parseggds_ade20k_160k-512x512.py"
FT_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parseggds_ft_ade20k_160k-512x512.py"


class TestPARSegGDSScaffold(unittest.TestCase):

    def test_head_is_residual_and_preserves_parseg3_path(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("class PARSegGDS(PARSeg3)", text)
        self.assertIn("class AttributeGeometrySeparation", text)
        self.assertIn('out["parseg_final_logits"] = out["final_logits"]', text)
        self.assertRegex(text, r"gds_alpha\s*=\s*nn\.Parameter\(torch\.zeros\(1\)\)")
        self.assertIn("torch.tanh(self.gds_alpha)", text)
        self.assertIn('out["final_logits"] = out["final_logits"] + gate * gds_delta', text)

    def test_margin_uses_independent_parseg_logits_not_corrected_final(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn('seg_logits["parseg_final_logits"]', text)
        self.assertRegex(text, r"ref_logits\s*=\s*resize\(\s*seg_logits\[\"parseg_final_logits\"\]")
        margin_body = re.search(
            r"def _gds_margin_loss\(.*?def loss_by_feat",
            text,
            flags=re.S,
        )
        self.assertIsNotNone(margin_body)
        self.assertIn("ref_logits.detach()", margin_body.group(0))

    def test_freeze_segmentor_freezes_backbone_and_decode_base(self):
        text = SEGMENTOR.read_text(encoding="utf-8")
        self.assertIn("class GDSEncoderDecoder(EncoderDecoder)", text)
        self.assertIn("self.backbone.parameters()", text)
        self.assertIn("set_parseg_base_requires_grad", text)
        self.assertIn("set_parseg_base_train_mode", text)

    def test_configs_define_ft_gate_and_full_training_paths(self):
        full = FULL_CFG.read_text(encoding="utf-8")
        ft = FT_CFG.read_text(encoding="utf-8")

        self.assertIn("mmseg.models.decode_heads.PARSegGDS", full)
        self.assertIn("type='PARSegGDS'", full)
        self.assertIn("gds_freeze_parseg=False", full)
        self.assertIn("train_cfg = dict(val_interval=8000)", full)

        self.assertIn("mmseg.models.segmentors.gds_encoder_decoder", ft)
        self.assertIn("type='GDSEncoderDecoder'", ft)
        self.assertIn("gds_freeze_parseg=True", ft)
        self.assertIn("load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'", ft)
        self.assertIn("max_iters = 40000", ft)
        self.assertIn("val_interval=8000", ft)
        self.assertIn("interval=8000", ft)


if __name__ == "__main__":
    unittest.main()
