"""No game imports, screenshots or inputs. Development-only model invariants."""
import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from offline.benchmark_stats import detector_digest, load_ocr, summarize
from offline.stat_templates import (Policy, Templates, calibrate, corpus_signature,
                                    ink, load_corpus, split_digits)

ROOT = Path(os.environ.get("TAILSTOPIA_REPO", Path(__file__).resolve().parents[1]))
PACKAGE = Path(__file__).resolve().parents[1]


class StatTemplateOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.dev, cls.evaluation, cls.excluded = load_corpus(ROOT)
        cls.locked = json.loads((PACKAGE / "offline/stat_template_policy.json").read_text())

    def model(self, kind):
        return Templates(self.dev, kind, Policy(**self.locked["methods"][kind]["policy"]))

    def test_source_split_removes_cross_source_pixel_duplicate(self):
        self.assertEqual((len(self.dev), len(self.evaluation), len(self.excluded)), (8, 7, 1))
        self.assertFalse({c.pixel_hash for c in self.dev} & {c.pixel_hash for c in self.evaluation})
        self.assertEqual(self.excluded[0].name, "failure_20261104_5_7.png")

    def test_holdout_cannot_create_a_template_bank(self):
        with self.assertRaisesRegex(ValueError, "development"):
            Templates(self.evaluation, "whole", Policy(.9, .1))

    def test_holdout_cannot_calibrate_thresholds(self):
        with self.assertRaisesRegex(ValueError, "evaluation"):
            calibrate(self.evaluation, "digit")

    def test_frozen_policy_matches_development_and_algorithm(self):
        self.assertEqual(self.locked["development_signature"], corpus_signature(self.dev))
        self.assertEqual(self.locked["detector_sha256"], detector_digest())
        for kind in ("whole", "digit"):
            self.assertEqual(self.locked["methods"][kind]["selected"]["wrong"], 0)

    def test_whole_number_inventory_is_real_development_labels_only(self):
        self.assertEqual(self.model("whole").supported, [3, 5, 6, 7, 11, 12])

    def test_digit_inventory_includes_real_segments_from_11_and_12(self):
        model = self.model("digit")
        self.assertEqual(model.supported, [1, 2, 3, 5, 6, 7])
        self.assertEqual(model.training_failures, [])

    def test_joined_reference_12_is_split_by_column_valley(self):
        case = next(c for c in self.dev if c.value == 12)
        mask = ink(case.image)
        n, *_ = cv2.connectedComponentsWithStats(mask)
        self.assertEqual(n-1, 1)
        self.assertEqual(len(split_digits(mask)), 2)

    def test_eleven_does_not_collapse_to_one_digit(self):
        case = next(c for c in self.dev if c.value == 11)
        self.assertEqual(len(split_digits(ink(case.image))), 2)
        self.assertIn(self.model("digit").read(case.image).value, (None, 11))

    def test_identical_competing_labels_return_unknown(self):
        model = self.model("whole")
        case = next(c for c in self.dev if c.value == 7)
        for value, vector, aspect in list(model.bank):
            if value == 7:
                model.bank.append((17, vector, aspect))
        result = model.read(case.image)
        self.assertIsNone(result.value)
        self.assertAlmostEqual(result.margin, 0, places=5)

    def test_blank_and_clipped_crops_are_unknown(self):
        for kind in ("whole", "digit"):
            model = self.model(kind)
            for fill in (0, 180, 255):
                self.assertIsNone(model.read(Image.new("RGB", (24, 18), (fill,)*3)).value)
            self.assertIsNone(model.read(Image.new("RGB", (30, 18), "white")).value)

    def test_model_is_read_only_and_tiny(self):
        for kind in ("whole", "digit"):
            model = self.model(kind)
            self.assertLess(model.template_bytes, 20000)
            self.assertTrue(all(not vector.flags.writeable for _, vector, _ in model.bank))

    def test_negative_fixtures_are_saved_crops_not_rendered_numbers(self):
        base = PACKAGE / "tests/fixtures/stat_template_negative"
        rows = json.loads((base / "manifest.json").read_text())
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(r["value"] is None and "source_sha256" in r and "panel_box" in r for r in rows))

    def test_baseline_adapter_does_not_import_live_module(self):
        with patch.dict("sys.modules", {"pyautogui": None, "mss": None}):
            ns, digest, target = load_ocr(ROOT)
        self.assertEqual(len(digest), 64)
        self.assertIn("target_met", ns)
        self.assertIn("stats", target)

    def test_negative_accept_is_not_miscounted_as_a_numeric_success(self):
        rows = [dict(split="negative", method="whole", truth=None, value=7, times=[]),
                dict(split="negative", method="whole", truth=None, value=None, times=[])]
        result = summarize(rows)[0]
        self.assertEqual(result["negative_false_accepts"], 1)
        self.assertIsNone(result["exact_accuracy"])
        self.assertEqual(result["wrong"], 1)

    def test_prototype_has_no_capture_or_input_imports(self):
        tree = ast.parse((PACKAGE / "offline/stat_templates.py").read_text())
        imports = [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        imports += [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(any(bad in name for bad in ("pyautogui", "mss", "ImageGrab", "mission_runtime")) for name in imports))


if __name__ == "__main__":
    unittest.main()
