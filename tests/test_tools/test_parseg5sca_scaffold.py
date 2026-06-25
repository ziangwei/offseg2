import ast
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class TestPARSeg5SCAScaffold(unittest.TestCase):

    def test_sca_decode_head_is_self_contained_and_defines_expected_classes(self):
        path = REPO / "mmseg/models/decode_heads/PARSeg5SCA.py"
        self.assertTrue(path.exists(), "missing PARSeg5SCA decode head")
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        classes = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }

        self.assertIn("SemanticContentAssignment", classes)
        self.assertIn("SCARefinementHead", classes)
        self.assertIn("PARSeg5SCA", classes)
        self.assertIn("assignment", text)
        self.assertIn("region_pixel_logits", text)
        self.assertIn("loss_region", text)
        self.assertNotIn("top2", text.lower())
        self.assertNotIn("PARSeg5EAF", text)
        self.assertNotIn("PARSeg5ICAR", text)
        self.assertNotIn("PARSeg5CPM", text)
        self.assertNotIn("PARSeg5ATM", text)

    def test_sca_config_imports_head_and_adds_region_assignment_losses(self):
        path = REPO / "local_configs/offseg2/Base/parseg5sca_ade20k_160k-512x512.py"
        self.assertTrue(path.exists(), "missing PARSeg5SCA config")
        cfg = path.read_text(encoding="utf-8")

        self.assertIn("mmseg.models.decode_heads.PARSeg5SCA", cfg)
        self.assertIn("type='PARSeg5SCA'", cfg)
        self.assertIn("sca_num_slots", cfg)
        self.assertIn("regionw", cfg)
        self.assertIn("assignment_entropyw", cfg)
        self.assertIn("assignment_balancew", cfg)

    def test_sca_design_doc_contains_train_and_analysis_command(self):
        path = REPO / "PARSeg5SCA_设计笔记.md"
        self.assertTrue(path.exists(), "missing PARSeg5SCA design note")
        doc = path.read_text(encoding="utf-8")

        self.assertIn("PARSeg5-SCA", doc)
        self.assertIn("Semantic Content Assignment", doc)
        self.assertIn("loss_region", doc)
        self.assertIn("tools/train_test_analyze.sh", doc)


if __name__ == "__main__":
    unittest.main()
