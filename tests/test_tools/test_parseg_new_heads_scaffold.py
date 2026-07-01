import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestPARSegNewHeadsScaffold(unittest.TestCase):

    def test_plcr_uses_pal_features_for_local_candidate_relation(self):
        head = (ROOT / "mmseg/models/decode_heads/PARSegPLCR.py").read_text(encoding="utf-8")
        cfg = (ROOT / "local_configs/offseg2/Base/parsegplcr_ade20k_160k-512x512.py").read_text(encoding="utf-8")

        self.assertIn("class PALCandidateRelation", head)
        self.assertIn("class PARSegPLCR(PARSegPALX)", head)
        self.assertIn('"palx_class_feats"', head)
        self.assertIn('"palx_refinement_feats"', head)
        self.assertIn("candidate_delta", head)
        self.assertIn("topk_idx = raw_base_logits.detach().topk", head)
        self.assertIn("base_head_logits = raw_base_head_logits + gate * relation[\"delta_logits\"]", head)
        self.assertIn("loss_plcr_rank", head)
        self.assertIn("loss_plcr_aux", head)
        self.assertIn("mmseg.models.decode_heads.PARSegPLCR", cfg)
        self.assertIn("type='PARSegPLCR'", cfg)
        self.assertIn("plcr_topk=5", cfg)

    def test_cdr_is_training_only_candidate_ranking(self):
        head = (ROOT / "mmseg/models/decode_heads/PARSegCDR.py").read_text(encoding="utf-8")
        cfg = (ROOT / "local_configs/offseg2/Base/parsegcdr_ade20k_160k-512x512.py").read_text(encoding="utf-8")

        self.assertIn("class PARSegCDR(PARSeg3)", head)
        self.assertNotIn("def forward", head)
        self.assertIn("def _candidate_rank_loss", head)
        self.assertIn("topk_idx = logits.detach().topk", head)
        self.assertIn("loss_cdr_final_rank", head)
        self.assertIn("loss_cdr_base_rank", head)
        self.assertIn("loss_cdr_refinement_rank", head)
        self.assertIn("mmseg.models.decode_heads.PARSegCDR", cfg)
        self.assertIn("type='PARSegCDR'", cfg)
        self.assertIn("cdr_topk=5", cfg)

    def test_osc_is_pre_decision_omni_scale_context_decoder(self):
        head = (ROOT / "mmseg/models/decode_heads/PARSegOSC.py").read_text(encoding="utf-8")
        cfg = (ROOT / "local_configs/offseg2/Base/parsegosc_ade20k_160k-512x512.py").read_text(encoding="utf-8")

        self.assertIn("class OmniScaleContext", head)
        self.assertIn("class PARSegOSC(PARSeg3)", head)
        self.assertIn("self.local_dw", head)
        self.assertIn("self.dilated_dw", head)
        self.assertIn("self.global_pool", head)
        self.assertIn("scale_gate", head)
        self.assertIn("context_delta = self.osc(feat_aligned)", head)
        self.assertIn("feat_aligned = feat_aligned + self._osc_gate() * context_delta", head)
        self.assertIn("base_head_logits = self.offset_learning(feat_aligned)", head)
        self.assertNotIn("final_logits = final_logits +", head)
        self.assertIn("mmseg.models.decode_heads.PARSegOSC", cfg)
        self.assertIn("type='PARSegOSC'", cfg)
        self.assertIn("osc_gate_max=0.35", cfg)


if __name__ == "__main__":
    unittest.main()
