import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEAD = ROOT / "mmseg" / "models" / "decode_heads" / "PARSegPALX.py"
FULL_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parsegpalx_ade20k_160k-512x512.py"
FT_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parsegpalx_ft_ade20k_160k-512x512.py"


class TestPARSegPALXScaffold(unittest.TestCase):

    def test_palx_replaces_internal_refinement_not_external_residual(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("class PALXRefinementHead", text)
        self.assertIn("class PARSegPALX(PARSeg3)", text)
        self.assertIn("self.prototype_attribute_refinement = PALXRefinementHead", text)
        self.assertIn("refinement_head_logits = refine[\"refinement_head_logits\"]", text)
        self.assertIn("final_logits = self.fusion(base_head_logits, refinement_head_logits)", text)
        self.assertNotIn("gds_delta", text)
        self.assertNotIn("sgc_gate", text)
        self.assertNotRegex(text, r"final_logits\s*=\s*.*\+\s*gate")

    def test_palx_exposes_internal_pal_geometry_for_losses(self):
        text = HEAD.read_text(encoding="utf-8")
        for key in [
            '"palx_class_cos"',
            '"palx_class_feats"',
            '"palx_refinement_feats"',
            '"palx_route_prob"',
            '"calibrated_attr_tokens"',
        ]:
            self.assertIn(key, text)
        self.assertIn("loss_palx_center", text)
        self.assertIn("loss_palx_margin", text)

    def test_margin_uses_detached_base_logits_as_training_negative_source(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("def _palx_margin_loss", text)
        body = re.search(r"def _palx_margin_loss\(.*?def _palx_center_loss", text, flags=re.S)
        self.assertIsNotNone(body)
        self.assertIn("base_logits.detach()", body.group(0))
        self.assertIn("topk", body.group(0))

    def test_freeze_api_keeps_only_pal_refinement_trainable_for_ft(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("set_parseg_base_requires_grad", text)
        self.assertIn('name.startswith("prototype_attribute_refinement.")', text)
        self.assertIn("set_parseg_base_train_mode", text)

    def test_configs_define_full_and_finetune_paths(self):
        full = FULL_CFG.read_text(encoding="utf-8")
        ft = FT_CFG.read_text(encoding="utf-8")

        self.assertIn("mmseg.models.decode_heads.PARSegPALX", full)
        self.assertIn("type='PARSegPALX'", full)
        self.assertIn("palx_freeze_parseg=False", full)
        self.assertIn("train_cfg = dict(val_interval=8000)", full)

        self.assertIn("mmseg.models.segmentors.gds_encoder_decoder", ft)
        self.assertIn("type='GDSEncoderDecoder'", ft)
        self.assertIn("palx_freeze_parseg=True", ft)
        self.assertIn("load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'", ft)
        self.assertIn("max_iters = 40000", ft)
        self.assertIn("val_interval=8000", ft)


if __name__ == "__main__":
    unittest.main()
