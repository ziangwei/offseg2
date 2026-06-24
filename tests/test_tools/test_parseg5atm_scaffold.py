import ast
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class TestPARSeg5ATMScaffold(unittest.TestCase):

    def test_atm_decode_head_is_self_contained_and_defines_expected_classes(self):
        path = REPO / "mmseg/models/decode_heads/PARSeg5ATM.py"
        self.assertTrue(path.exists(), "missing PARSeg5ATM decode head")
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        classes = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }

        self.assertIn("CrossImageAttributeTokenMemory", classes)
        self.assertIn("AttributeTransitionRefinementHead", classes)
        self.assertIn("PARSeg5ATM", classes)
        self.assertIn("memory_token", text)
        self.assertIn("dist.all_reduce", text)
        self.assertIn("memory = self.memory_token.detach().clone()", text)
        self.assertNotIn("return self.norm(nudged)", text)
        self.assertNotIn("PARSeg5EAF", text)
        self.assertNotIn("PARSeg5ICAR", text)
        self.assertNotIn("PARSeg5CPM", text)

    def test_atm_config_imports_head_and_adds_memory_losses(self):
        path = REPO / "local_configs/offseg2/Base/parseg5atm_ade20k_160k-512x512.py"
        self.assertTrue(path.exists(), "missing PARSeg5ATM config")
        cfg = path.read_text(encoding="utf-8")

        self.assertIn("mmseg.models.decode_heads.PARSeg5ATM", cfg)
        self.assertIn("type='PARSeg5ATM'", cfg)
        self.assertIn("atmw", cfg)
        self.assertIn("atm_focusw", cfg)
        self.assertIn("atm_momentum", cfg)

    def test_atm_design_doc_contains_training_command(self):
        path = REPO / "PARSeg5ATM_设计笔记.md"
        self.assertTrue(path.exists(), "missing PARSeg5ATM design note")
        doc = path.read_text(encoding="utf-8")

        self.assertIn("PARSeg5-ATM", doc)
        self.assertIn("memory_token[c,a]", doc)
        self.assertIn("loss_atm_focus", doc)
        self.assertIn("tools/train_test_analyze.sh", doc)


if __name__ == "__main__":
    unittest.main()
