import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEAD = ROOT / "mmseg" / "models" / "decode_heads" / "PARSegLCR.py"
FULL_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parseglcr_ade20k_160k-512x512.py"
FT_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parseglcr_ft_ade20k_160k-512x512.py"
CONTINUE_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parseglcr_ade20k_200k-512x512.py"
RESUME_SCRIPT = ROOT / "tools" / "train_parseglcr_200k_resume.sh"
POST40K_CFG = ROOT / "local_configs" / "offseg2" / "Base" / "parseglcr_post40k_from160k_ade20k-512x512.py"
POST40K_SCRIPT = ROOT / "tools" / "train_parseglcr_post40k_from160k.sh"


class TestPARSegLCRScaffold(unittest.TestCase):

    def test_lcr_is_dynamic_candidate_relation_not_global_metric(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("class LocalCandidateRelation", text)
        self.assertIn("class PARSegLCR(PARSeg3)", text)
        self.assertIn("topk_idx = raw_base_logits.detach().topk", text)
        self.assertIn("self.class_embed", text)
        self.assertIn("candidate_delta", text)
        self.assertNotIn("class_weight = nn.Parameter", text)
        self.assertNotIn("NormalizedMetricClassifier", text)

    def test_lcr_injects_before_pal_refinement_not_after_final_logits(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("raw_base_head_logits = self.offset_learning(feat_aligned)", text)
        self.assertIn("base_head_logits = raw_base_head_logits + gate * relation[\"delta_logits\"]", text)
        self.assertIn("feat_aligned,", text)
        self.assertIn("base_head_logits,", text)
        self.assertIn("final_logits = self.fusion(base_head_logits, refinement_head_logits)", text)
        self.assertNotRegex(text, r'final_logits\s*=\s*final_logits\s*\+')
        self.assertNotRegex(text, r'final_logits\s*=\s*.*delta_logits')

    def test_lcr_trains_candidate_misranking_directly(self):
        text = HEAD.read_text(encoding="utf-8")
        self.assertIn("loss_lcr_aux", text)
        self.assertIn("loss_lcr_rank", text)
        body = re.search(r"def _lcr_rank_loss\(.*?def loss_by_feat", text, flags=re.S)
        self.assertIsNotNone(body)
        body_text = body.group(0)
        self.assertIn("gt_in_candidates", body_text)
        self.assertIn("candidate_scores", body_text)
        self.assertIn("raw_pred", body_text)
        self.assertIn("lcr_rank_margin", body_text)

    def test_configs_define_full_and_warm_start_runs(self):
        full = FULL_CFG.read_text(encoding="utf-8")
        ft = FT_CFG.read_text(encoding="utf-8")

        self.assertIn("mmseg.models.decode_heads.PARSegLCR", full)
        self.assertIn("type='PARSegLCR'", full)
        self.assertIn("lcr_topk=5", full)
        self.assertIn("lcr_rankw=0.20", full)
        self.assertIn("train_cfg = dict(val_interval=8000)", full)

        self.assertIn("_base_ = ['./parseglcr_ade20k_160k-512x512.py']", ft)
        self.assertIn("load_from = 'work_dirs/parseg3_ade20k_160k-512x512_4x4_try1/iter_160000.pth'", ft)
        self.assertNotIn("freeze_base=True", ft)
        self.assertIn("max_iters = 40000", ft)
        self.assertIn("lr=0.00002", ft)

    def test_continue_200k_config_and_script_resume_from_lcr_checkpoint(self):
        cfg = CONTINUE_CFG.read_text(encoding="utf-8")
        script = RESUME_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("_base_ = ['./parseglcr_ade20k_160k-512x512.py']", cfg)
        self.assertIn("parseglcr_ade20k_160k-512x512_4x4_try1/iter_160000.pth", cfg)
        self.assertIn("max_iters = 200000", cfg)
        self.assertIn("end=max_iters", cfg)
        self.assertIn("save_last=True", cfg)

        self.assertIn("set -euo pipefail", script)
        self.assertIn("parseglcr_ade20k_200k-512x512.py", script)
        self.assertIn("parseglcr_ade20k_160k-512x512_4x4_try1/iter_160000.pth", script)
        self.assertIn("parseglcr_ade20k_200k-512x512_4x4_try1", script)
        self.assertIn("[[ ! -f \"$CKPT\" ]]", script)
        self.assertIn("--resume", script)
        self.assertIn("load_from=\"$CKPT\"", script)

    def test_post40k_config_loads_lcr_checkpoint_without_strict_resume(self):
        cfg = POST40K_CFG.read_text(encoding="utf-8")
        script = POST40K_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("_base_ = ['./parseglcr_ade20k_160k-512x512.py']", cfg)
        self.assertIn("parseglcr_ade20k_160k-512x512_4x4_try1/iter_160000.pth", cfg)
        self.assertIn("max_iters = 40000", cfg)
        self.assertIn("lr=0.000001", cfg)
        self.assertIn("eta_min=1e-8", cfg)
        self.assertIn("begin=0, end=max_iters", cfg)
        self.assertNotIn("lr=0.00002", cfg)
        self.assertNotIn("'head': dict(lr_mult=10.)", cfg)
        self.assertNotIn("LinearLR", cfg)
        self.assertNotIn("resume = True", cfg)

        self.assertIn("parseglcr_post40k_from160k_ade20k-512x512.py", script)
        self.assertIn("parseglcr_post40k_from160k_ade20k-512x512_4x4_try1", script)
        self.assertIn("[[ ! -f \"$CKPT\" ]]", script)
        self.assertIn("load_from=\"$CKPT\"", script)
        self.assertNotIn("--resume", script)


if __name__ == "__main__":
    unittest.main()
