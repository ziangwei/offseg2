import importlib.util
import pathlib
from types import SimpleNamespace
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "probe_cgr_redecision.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("probe_cgr_redecision", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCGRProbeScaffold(unittest.TestCase):

    def test_default_groups_cover_diagnosed_confusions(self):
        tool = _load_tool()
        groups = tool.default_confusion_groups()
        by_name = {group["name"]: set(group["classes"]) for group in groups}

        self.assertIn("wall_family", by_name)
        self.assertTrue({"wall", "ceiling", "door", "windowpane", "cabinet", "mirror", "curtain"} <= by_name["wall_family"])
        self.assertEqual(by_name["building_tree"], {"building", "tree"})
        self.assertEqual(by_name["road_sidewalk"], {"road", "sidewalk"})
        self.assertEqual(by_name["rug_floor"], {"rug", "floor"})
        self.assertEqual(by_name["armchair_sofa"], {"armchair", "sofa"})

    def test_group_resolution_is_name_based_and_skips_missing_classes(self):
        tool = _load_tool()
        classes = [
            "wall", "building", "tree", "floor", "rug", "sofa", "armchair",
            "road", "sidewalk", "ceiling", "door", "windowpane", "cabinet",
            "mirror", "curtain",
        ]

        resolved = tool.resolve_confusion_groups(tool.default_confusion_groups(), classes)
        resolved_by_name = {group.name: group for group in resolved}

        self.assertEqual(resolved_by_name["building_tree"].class_ids, [1, 2])
        self.assertEqual(resolved_by_name["road_sidewalk"].class_ids, [7, 8])
        self.assertIn(0, resolved_by_name["wall_family"].class_ids)
        self.assertIn(11, resolved_by_name["wall_family"].class_ids)
        self.assertEqual(resolved_by_name["wall_family"].missing, [])

        sparse = tool.resolve_confusion_groups(tool.default_confusion_groups(), ["wall", "ceiling"])
        self.assertEqual([group.name for group in sparse], ["wall_family"])
        self.assertEqual(sparse[0].class_ids, [0, 1])
        self.assertIn("door", sparse[0].missing)

    def test_fixed_probe_pipeline_resizes_image_and_label_together(self):
        tool = _load_tool()
        cfg = SimpleNamespace(
            val_dataloader=dict(
                dataset=dict(
                    pipeline=[
                        dict(type="LoadImageFromFile"),
                        dict(type="Resize", scale=(2048, 512), keep_ratio=True),
                        dict(type="LoadAnnotations", reduce_zero_label=True),
                        dict(type="PackSegInputs"),
                    ]
                )
            )
        )

        pipeline = tool._fixed_probe_pipeline(cfg, (512, 512))

        self.assertEqual([step["type"] for step in pipeline], [
            "LoadImageFromFile",
            "LoadAnnotations",
            "Resize",
            "PackSegInputs",
        ])
        self.assertEqual(pipeline[1]["reduce_zero_label"], True)
        self.assertEqual(pipeline[2]["scale"], (512, 512))
        self.assertEqual(pipeline[2]["keep_ratio"], False)

    def test_probe_uses_pure_gt_feature_prototypes_not_base_confidence(self):
        text = TOOL.read_text(encoding="utf-8")

        self.assertIn("refinement_feat_proj", text)
        self.assertIn("probe_size", text)
        self.assertIn("keep_ratio=False", text)
        self.assertIn("gt_sem_seg", text)
        self.assertIn("prototype_banks", text)
        self.assertIn("run_kmeans", text)
        self.assertIn("feature_affinity_smooth", text)
        self.assertIn("correction", text)
        self.assertIn("damage", text)
        self.assertIn("baseline_mIoU", text)
        self.assertIn("cgr_mIoU", text)
        self.assertNotIn("p_base * confidence", text)
        self.assertNotIn("base_conf", text)


if __name__ == "__main__":
    unittest.main()
