import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEAD = ROOT / "mmseg" / "models" / "decode_heads" / "PARSegDGM.py"
CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parsegdgm_ade20k_160k-512x512.py"
FT_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parsegdgm_ft_ade20k_160k-512x512.py"


class TestPARSegDGMScaffold(unittest.TestCase):

    def test_dgm_is_internal_base_decision_not_final_posthoc_override(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("class PARSegDGM(PARSeg3)", text)
        self.assertIn("class NormalizedMetricClassifier", text)
        self.assertIn('raw_base_head_logits = self.offset_learning(feat_aligned)', text)
        self.assertIn('base_head_logits = raw_base_head_logits + gate * metric["logits"]', text)
        self.assertIn("feat_aligned,", text)
        self.assertIn("base_head_logits,", text)
        self.assertIn("final_logits = self.fusion(base_head_logits, refinement_head_logits)", text)
        self.assertNotRegex(text, r'final_logits\s*=\s*final_logits\s*\+')
        self.assertNotRegex(text, r'final_logits\s*=\s*.*dgm_metric_logits')

    def test_dgm_exposes_metric_geometry_and_losses(self):
        text = HEAD.read_text(encoding="utf-8")
        for key in [
            '"dgm_metric_logits"',
            '"dgm_class_cos"',
            '"dgm_metric_feat"',
            '"dgm_class_weight"',
        ]:
            self.assertIn(key, text)
        for loss_name in [
            "loss_dgm_aux",
            "loss_dgm_margin",
            "loss_dgm_pull",
            "loss_dgm_weight_sep",
        ]:
            self.assertIn(loss_name, text)

    def test_margin_uses_metric_nearest_negatives_not_base_candidates(self):
        text = HEAD.read_text(encoding="utf-8")
        body = re.search(r"def _dgm_margin_loss\(.*?def _dgm_pull_loss", text, flags=re.S)
        self.assertIsNotNone(body)
        body_text = body.group(0)
        self.assertIn("neg_ref = class_cos.detach().clone()", body_text)
        self.assertIn("neg_ref.topk", body_text)
        self.assertNotIn("base_logits", body_text)
        self.assertNotIn("raw_base_head_logits", body_text)

    def test_config_registers_full_training_head(self):
        cfg = CFG.read_text(encoding="utf-8")
        self.assertIn("mmseg.models.decode_heads.PARSegDGM", cfg)
        self.assertIn("type='PARSegDGM'", cfg)
        self.assertIn("dgm_gate_init=0.05", cfg)
        self.assertIn("dgm_marginw=0.15", cfg)
        self.assertIn("dgm_pullw=0.05", cfg)
        self.assertIn("train_cfg = dict(val_interval=8000)", cfg)
        self.assertIn("interval=8000", cfg)

    def test_warm_start_finetune_config_keeps_model_trainable(self):
        cfg = FT_CFG.read_text(encoding="utf-8")
        self.assertIn("_base_ = ['./parsegdgm_ade20k_160k-512x512.py']", cfg)
        self.assertIn("load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'", cfg)
        self.assertNotIn("freeze_base=True", cfg)
        self.assertNotIn("dgm_freeze_parseg=True", cfg)
        self.assertIn("max_iters = 40000", cfg)
        self.assertIn("val_interval=8000", cfg)
        self.assertIn("lr=0.00002", cfg)


if __name__ == "__main__":
    unittest.main()
