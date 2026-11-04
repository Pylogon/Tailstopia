"""Recorded digit crops and offline control checks; no game/window imports."""
import ast
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image, ImageOps
import pytesseract


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "reroll_ocr"
TESSERACT = (r"C:\Program Files\Tesseract-OCR\tesseract.exe"
             if Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").is_file()
             else shutil.which("tesseract"))


def helpers():
    names = {"prepared_ocr_image", "read_digit", "ocr_number_fast", "ocr_number_robust", "ocr_number_fallback",
             "capture_and_read_fast", "stable_stats", "verify_potential_match"}
    tree = ast.parse((ROOT / "tailstopia_auto_reroll.py").read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"Image": Image, "ImageOps": ImageOps, "pytesseract": pytesseract,
          "re": re, "Counter": Counter, "time": SimpleNamespace(sleep=Mock()),
          "stop_requested": threading.Event(), "SCREEN_STABILITY_SECONDS": 0.12,
          "READ_CONFIRM_SECONDS": 0.18, "OCR_RETRY_ATTEMPTS": 2,
          "OCR_RETRY_DELAY_SECONDS": 0.25, "save_debug": Mock()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "reroll_ocr_helpers", "exec"), ns)
    return ns


@unittest.skipUnless(TESSERACT, "Tesseract executable is required for recorded OCR integration")
class RecordedOCRTests(unittest.TestCase):
    def setUp(self):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT
        self.ns = helpers()

    def test_fast_reads_all_recorded_digits(self):
        for fixture in json.loads((FIXTURES / "manifest.json").read_text()):
            with self.subTest(fixture=fixture["file"]), Image.open(FIXTURES / fixture["file"]) as crop:
                self.assertEqual(self.ns["ocr_number_fast"](crop), fixture["value"])

    def test_robust_reads_all_recorded_digits(self):
        for fixture in json.loads((FIXTURES / "manifest.json").read_text()):
            with self.subTest(fixture=fixture["file"]), Image.open(FIXTURES / fixture["file"]) as crop:
                self.assertEqual(self.ns["ocr_number_robust"](crop), fixture["value"])

    def test_failure_eleven_agrees_across_both_recovery_image_styles(self):
        with Image.open(FIXTURES / "failure_20261104_3_11.png") as crop:
            self.assertEqual(self.ns["read_digit"](crop, 140, 7, normalize=True), 11)
            self.assertEqual(self.ns["read_digit"](crop, 180, 7), 11)

    def test_uniform_backgrounds_are_not_numbers(self):
        for color in (0, 180, 255):
            with self.subTest(color=color):
                self.assertIsNone(self.ns["ocr_number_fast"](Image.new("RGB", (24, 18), (color,) * 3)))


class OCRSafetyTests(unittest.TestCase):
    def setUp(self):
        self.ns = helpers()

    def test_grayscale_strokes_survive_preprocessing(self):
        image = Image.new("L", (24, 18), 220)
        image.putpixel((10, 8), 90)
        image.putpixel((11, 8), 120)
        prepared = self.ns["prepared_ocr_image"](image, 140, normalize=True)
        self.assertTrue(any(0 < p < 255 for p in prepared.tobytes()))
        self.assertLess(prepared.width, 24 * 8 + 20)

    def test_fast_still_requires_two_agreeing_reads(self):
        self.ns["read_digit"] = Mock(side_effect=[11, 1, None])
        self.ns["ocr_number_fallback"] = Mock(return_value=None)
        self.assertIsNone(self.ns["ocr_number_fast"](None))

    def test_robust_still_rejects_competing_readings(self):
        self.ns["read_digit"] = Mock(side_effect=[11, 11, 11, 11, 1, 1])
        self.ns["ocr_number_fallback"] = Mock(return_value=None)
        self.assertIsNone(self.ns["ocr_number_robust"](None))

    def test_recovery_rejects_disagreeing_image_styles(self):
        self.ns["read_digit"] = Mock(side_effect=[11, 1])
        self.assertIsNone(self.ns["ocr_number_fallback"](None))

    def test_robust_recovery_rejects_competing_votes(self):
        self.ns["read_digit"] = Mock(side_effect=[11, 11, 11, 11, 1, 1])
        self.assertIsNone(self.ns["ocr_number_fallback"](None, robust=True))

    def test_uncertain_read_retries_without_reroll(self):
        self.ns["capture_and_read_fast"] = Mock(return_value=(None, {}, None, "OCR ambiguous: VIT=None"))
        with self.assertRaisesRegex(RuntimeError, "OCR still unclear"):
            self.ns["stable_stats"](None, 0, 0)
        self.assertEqual(self.ns["capture_and_read_fast"].call_count, 3)
        self.ns["save_debug"].assert_called_once()

    def test_candidate_still_requires_two_fresh_robust_reads(self):
        expected = {"VIT": 11}
        self.ns["capture_stats"] = Mock(side_effect=[({"frame": 1}, None), ({"frame": 2}, None)])
        self.ns["robust_stats_from_crops"] = Mock(return_value=expected)
        self.ns["target_met"] = Mock(return_value=True)
        self.assertTrue(self.ns["verify_potential_match"](None, 0, 0, expected))
        self.assertEqual(self.ns["robust_stats_from_crops"].call_count, 2)
        self.assertEqual(self.ns["robust_stats_from_crops"].call_args_list[0].args, ({"frame": 1},))
        self.assertEqual(self.ns["robust_stats_from_crops"].call_args_list[1].args, ({"frame": 2},))

    def test_candidate_disagreement_stops_and_saves_debug(self):
        self.ns["capture_stats"] = Mock(return_value=({}, None))
        self.ns["robust_stats_from_crops"] = Mock(return_value={"VIT": 1})
        self.ns["target_met"] = Mock(return_value=True)
        with self.assertRaisesRegex(RuntimeError, "Candidate verification failed"):
            self.ns["verify_potential_match"](None, 0, 0, {"VIT": 11})
        self.ns["save_debug"].assert_called_once()

    def test_f9_stops_before_another_ocr_capture(self):
        self.ns["stop_requested"].set()
        self.ns["capture_and_read_fast"] = Mock()
        with self.assertRaisesRegex(RuntimeError, "F9 pressed"):
            self.ns["stable_stats"](None, 0, 0)
        self.ns["capture_and_read_fast"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
