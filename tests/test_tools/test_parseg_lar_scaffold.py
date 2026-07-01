import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestPARSegLARScaffold(unittest.TestCase):

    def test_lar_head_uses_image_only_guidance_and_local_attender(self):
        head = (ROOT / "mmseg/models/decode_heads/PARSegLAR.py").read_text(encoding="utf-8")

        self.assertIn("class DualBranchGuideEncoder", head)
        self.assertIn("class LocalAttender", head)
        self.assertIn("class LocalReprojectionGate", head)
        self.assertIn("class PARSegLAR(PARSeg3)", head)
        # guidance must come only from the image, never from feat_aligned / backbone features
        self.assertIn("def forward(self, image):", head)
        self.assertIn("guide_full = self.guide_encoder(img)", head)
        # attender output must stay a combination of `value` (feat_aligned), matching
        # both papers' own "output is a linear/convex combination of V" formulation.
        # Its init should be center-biased, not a uniform 3x3 blur.
        self.assertIn("nn.init.zeros_(self.weight_conv.weight)", head)
        self.assertIn("nn.init.zeros_(self.weight_conv.bias)", head)
        self.assertIn("center_idx = self.offsets.index((0, 0))", head)
        self.assertIn("self.weight_conv.bias[center_idx].fill_(self.center_bias)", head)
        self.assertIn("F.softmax(self.weight_conv(guide), dim=1)", head)
        self.assertIn("guide and value spatial sizes must match", head)
        # LAR-A should be a method-level same-resolution reprojection block:
        # the strength is spatially predicted from image guidance, not a
        # PARSeg3 warm-start scalar knob.
        self.assertIn("self.reprojection_gate = LocalReprojectionGate", head)
        self.assertIn("gate = self.reprojection_gate(guide)", head)
        self.assertIn("feat_aligned = feat_aligned + gate * (enriched - feat_aligned)", head)
        self.assertNotIn("def _lar_gate", head)
        self.assertNotIn("self.lar_alpha", head)
        # variant B (upsample_factor>1) is honestly NOT identity-safe -- must not
        # pretend otherwise by gating it the same way
        self.assertIn("feat_aligned = enriched", head)
        # must not reuse NAF/UPLiFT's own released (VFM-pretrained) weights
        self.assertNotIn("load_state_dict_from_url", head)
        self.assertNotIn("torch.hub", head)
        self.assertNotIn("pretrained=True", head)
        # needs the image via the existing (unmodified) IGREncoderDecoder hook
        self.assertIn("def set_image(self, img):", head)
        self.assertIn("self._cur_img", head)

    def test_lar_a_config_is_from_scratch_method_and_lar_b_stays_ablation(self):
        cfg_a = (ROOT / "local_configs/offseg2/Base/parseglar_a_ade20k_160k-512x512.py").read_text(encoding="utf-8")
        cfg_b = (ROOT / "local_configs/offseg2/Base/parseglar_b_ade20k_160k-512x512.py").read_text(encoding="utf-8")

        self.assertIn("mmseg.models.decode_heads.PARSegLAR", cfg_a)
        self.assertIn("mmseg.models.segmentors.igr_encoder_decoder", cfg_a)
        self.assertIn("type='IGREncoderDecoder'", cfg_a)
        self.assertIn("freeze_base=False", cfg_a)
        self.assertIn("type='PARSegLAR'", cfg_a)
        self.assertIn("lar_upsample_factor=1", cfg_a)
        self.assertIn("lar_gate_init=0.05", cfg_a)
        self.assertIn("lar_gate_max=0.30", cfg_a)
        self.assertNotIn("load_from", cfg_a)
        self.assertNotIn("iter_160000.pth", cfg_a)
        self.assertNotIn("max_iters = 32000", cfg_a)
        self.assertNotIn("optimizer=dict(type='AdamW', lr=0.00002", cfg_a)

        # Variant B remains available only as a high-resolution ablation; the
        # main LAR-A config is the from-scratch method setting.
        self.assertIn("lar_upsample_factor=2", cfg_b)

    def test_lar_reuses_igr_segmentor_file_unmodified(self):
        # IGREncoderDecoder already does exactly what LAR needs (stash the
        # image via set_image in extract_feat); confirm we did not fork it.
        segmentor = (ROOT / "mmseg/models/segmentors/igr_encoder_decoder.py").read_text(encoding="utf-8")
        self.assertIn("class IGREncoderDecoder(EncoderDecoder)", segmentor)
        self.assertIn("if hasattr(self.decode_head, 'set_image'):", segmentor)


if __name__ == "__main__":
    unittest.main()
