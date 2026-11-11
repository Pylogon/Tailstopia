"""Offline new-game profile: frozen source split, abstention and crop geometry."""
from dataclasses import replace
import json
import os
from pathlib import Path
import unittest

from PIL import Image

from offline.new_stat_templates import NativeTemplates, calibrate_sources, load_samples, native_ink
from offline.benchmark_new_stats import code_hash, development_hash
from offline.stat_templates import Policy, ink


PACKAGE = Path(os.environ.get("STAT_NEW_PACKAGE", Path(__file__).resolve().parents[1]))


class NewGameStatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = load_samples(PACKAGE/"tests/fixtures/stat_new")
        cls.dev = [s for s in cls.samples if s.split == "development"]
        cls.raw = [s for s in cls.samples if s.split == "evaluation"]
        cls.policy = json.loads((PACKAGE/"offline/new_stat_policy.json").read_text())

    def model(self, kind):
        p = self.policy["methods"][kind]["selected"]
        return NativeTemplates(self.dev, kind, Policy(p["threshold"], p["margin"]))

    def test_frozen_source_policy_and_dependency_code_match(self):
        self.assertEqual(self.policy["development_signature"], development_hash(self.dev))
        self.assertEqual(self.policy["detector_sha256"], code_hash())
        self.assertEqual((len(self.dev), len(self.raw)), (24, 16))
        self.assertEqual({s.source for s in self.dev}, {"1", "2", "3"})
        self.assertEqual({s.source for s in self.raw}, {"4", "5"})

    def test_pixel_dedup_leaves_three_unique_test_crops(self):
        seen = {s.pixel_hash for s in self.dev}
        evaluation = []
        for sample in self.raw:
            if sample.pixel_hash not in seen:
                seen.add(sample.pixel_hash)
                evaluation.append(sample)
        self.assertEqual([s.name for s in evaluation], ["4_VIT_7.png", "4_LCK_8.png", "5_VIT_5.png"])

    def test_eval_cannot_train_or_select_thresholds(self):
        with self.assertRaises(ValueError):
            NativeTemplates(self.raw, "whole", Policy(.97, .12))
        with self.assertRaises(ValueError):
            calibrate_sources(self.raw, "whole")

    def test_native_eleven_pixel_glyph_is_accepted_without_changing_old_profile(self):
        sample = self.dev[0]
        self.assertEqual(native_ink(sample.image).shape[0], 11)
        self.assertIsNone(ink(sample.image))

    def test_digits_zero_to_nine_come_from_development_only(self):
        model = self.model("digit")
        self.assertEqual(model.supported, list(range(10)))
        self.assertEqual(model.training_failures, [])
        self.assertEqual(model.training_hashes, frozenset(s.pixel_hash for s in self.dev))

    def test_locked_reference_regressions_keep_ambiguous_eight_unknown(self):
        expected = {"4_VIT_7.png": 7, "4_LCK_8.png": None, "5_VIT_5.png": 5}
        for kind in ("whole", "digit"):
            model = self.model(kind)
            for sample in self.raw:
                if sample.name in expected:
                    with self.subTest(kind=kind, case=sample.name):
                        result = model.read(sample.image)
                        self.assertEqual(result.value, expected[sample.name])
                        if sample.value == 8:
                            self.assertLess(result.score, model.policy.threshold)

    def test_identical_competitors_are_unknown(self):
        model = self.model("whole")
        sample = next(s for s in self.dev if s.value == 7)
        model.bank += [(17, v, a) for label, v, a in model.bank if label == 7]
        self.assertIsNone(model.read(sample.image).value)

    def test_blank_and_wrong_size_are_unknown(self):
        for kind in ("whole", "digit"):
            model = self.model(kind)
            self.assertIsNone(model.read(Image.new("RGB", (24, 18), "white")).value)
            self.assertIsNone(model.read(Image.new("RGB", (25, 18), "white")).value)


if __name__ == "__main__":
    unittest.main()
