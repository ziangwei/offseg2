import ast
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class TestPARSeg5Scaffold(unittest.TestCase):

    def _module_classes(self, relative_path):
        path = REPO / relative_path
        self.assertTrue(path.exists(), f"missing file: {relative_path}")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }

    def test_new_decode_heads_define_expected_registered_classes(self):
        eaf_classes = self._module_classes("mmseg/models/decode_heads/PARSeg5EAF.py")
        icar_classes = self._module_classes("mmseg/models/decode_heads/PARSeg5ICAR.py")

        self.assertIn("EvidenceAwareCorrectionFusion", eaf_classes)
        self.assertIn("PARSeg5EAF", eaf_classes)
        self.assertIn("IndependentPrototypeGuidedAttributeCalibration", icar_classes)
        self.assertIn("PARSeg5ICAR", icar_classes)

    def test_configs_import_and_select_new_heads(self):
        eaf_cfg = (REPO / "local_configs/offseg2/Base/parseg5eaf_ade20k_160k-512x512.py").read_text(
            encoding="utf-8"
        )
        icar_cfg = (REPO / "local_configs/offseg2/Base/parseg5icar_ade20k_160k-512x512.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("mmseg.models.decode_heads.PARSeg5EAF", eaf_cfg)
        self.assertIn("type='PARSeg5EAF'", eaf_cfg)
        self.assertIn("mmseg.models.decode_heads.PARSeg5ICAR", icar_cfg)
        self.assertIn("type='PARSeg5ICAR'", icar_cfg)

    def test_icar_is_not_coupled_to_eaf_module(self):
        icar = (REPO / "mmseg/models/decode_heads/PARSeg5ICAR.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("PARSeg5EAF", icar)

    def test_design_doc_contains_train_and_test_commands(self):
        doc = (REPO / "PARSeg5_设计与运行.md").read_text(encoding="utf-8")

        self.assertIn("parseg3_ade20k_160k-512x512_4x4_try2", doc)
        self.assertIn("parseg5eaf_ade20k_160k-512x512", doc)
        self.assertIn("parseg5icar_ade20k_160k-512x512", doc)
        self.assertIn("tools/train_test_analyze.sh", doc)
        self.assertIn("tools/dist_test.sh", doc)

    def test_train_test_analyze_script_writes_workdir_reports(self):
        path = REPO / "tools" / "train_test_analyze.sh"
        self.assertTrue(path.exists(), "missing train-test-analysis helper script")
        script = path.read_text(encoding="utf-8")

        self.assertIn("tools/dist_train.sh", script)
        self.assertIn("tools/dist_test.sh", script)
        self.assertIn("tools/analyze_parseg3_failures.py", script)
        self.assertIn("tools/analyze_parseg3_confusions.py", script)
        self.assertIn("run_conclusion.txt", script)


if __name__ == "__main__":
    unittest.main()
