"""Offline safety tests: mocked inputs/clock and real native-font number crops."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from PIL import Image
from offline.new_stat_templates import NativeTemplates, load_samples
from offline.stat_templates import Policy

from talesoftails_auto_reroll import (BUTTON_BOX, LiveIO, Reader, SETTINGS,
                                    STAT_BOXES, run, select_game_window, target_met, wait_checked)
from tailstopia_macro_replay import StopRequested


ROOT = Path(__file__).resolve().parents[1]
GOOD = {"AGI": 8, "CHA": 10}
BAD = {"AGI": 10, "CHA": 5}


class FakeIO:
    def __init__(self):
        self.clicks = 0
        self.waits = []
        self.captures = 0

    def check(self):
        pass

    def capture(self):
        self.captures += 1
        return object()

    def click_random(self):
        self.clicks += 1

    def wait(self, seconds):
        self.waits.append(seconds)


def fake_reader(sequence):
    reader = Mock()
    reader.read.side_effect = [(v, {}) for v in sequence]
    return reader


class RerollTests(unittest.TestCase):
    def test_focused_game_selected_when_two_clients_match(self):
        candidates = [dict(hwnd=i, title="TalesofTails", size=(1280, 600)) for i in (1, 2)]
        self.assertEqual(select_game_window(candidates, 2), 2)

    def test_vscode_title_match_does_not_count_as_game(self):
        candidates = [dict(hwnd=1, title="TalesofTails", size=(1280, 600)),
                      dict(hwnd=2, title="TalesofTails - Visual Studio Code", size=(1366, 700))]
        self.assertEqual(select_game_window(candidates, 2), 1)

    def test_exact_game_title_preferred_over_other_same_size_title(self):
        candidates = [dict(hwnd=1, title="TalesofTails", size=(1280, 600)),
                      dict(hwnd=2, title="TalesofTails Launcher", size=(1280, 600))]
        self.assertEqual(select_game_window(candidates, 3), 1)

    def test_two_unfocused_clients_fail_safely(self):
        candidates = [dict(hwnd=i, title="TalesofTails", size=(1280, 600)) for i in (1, 2)]
        with self.assertRaisesRegex(RuntimeError, "Cannot select one"):
            select_game_window(candidates, 3)

    def test_no_compatible_client_fails_safely(self):
        with self.assertRaises(RuntimeError):
            select_game_window([dict(hwnd=1, title="TalesofTails", size=(1920, 1080))], 1)

    def test_both_boundaries_required(self):
        for values in (GOOD, dict(AGI=9, CHA=9), dict(AGI=9, CHA=11, VIT=6, INT=6, TAL=6),
                       dict(AGI=12, CHA=10, VIT=6, INT=5, TAL=5)):
            self.assertTrue(target_met(values))
        for values in (dict(AGI=7, CHA=10, VIT=7, INT=7, TAL=7),
                       dict(AGI=10, CHA=8), dict(AGI=8, CHA=9),
                       dict(GOOD, AGI=None), dict(GOOD, AGI=True)):
            self.assertFalse(target_met(values))

    def test_old_agi_exact_and_lck_constraints_removed(self):
        self.assertTrue(target_met(dict(GOOD, LCK=None)))
        self.assertNotEqual(GOOD["AGI"], 12)

    def test_unknown_required_stats_never_allow_next_roll(self):
        for name in ("AGI", "CHA"):
            io = FakeIO()
            with self.subTest(stat=name), self.assertRaisesRegex(RuntimeError, "UNKNOWN"):
                run(io, fake_reader([dict(GOOD, **{name: None})] * 4), Mock())
            self.assertEqual(io.clicks, 1)

    def test_total_condition_removed_and_other_stats_not_required(self):
        self.assertEqual(set(STAT_BOXES), {"AGI", "CHA"})
        self.assertTrue(target_met(GOOD))
        self.assertTrue(target_met(dict(GOOD, VIT=None, INT=None, TAL=None)))
        self.assertTrue(target_met(dict(GOOD, VIT=1, INT=1, TAL=1)))

    def test_success_requires_two_fresh_reads_and_no_extra_click(self):
        io = FakeIO()
        self.assertEqual(run(io, fake_reader([GOOD, GOOD]), Mock()), GOOD)
        self.assertEqual(io.clicks, 1)
        self.assertEqual(io.waits, [2.75, .2])
        self.assertEqual(io.captures, 3)  # Pre-click UI guard + two confirmations.

    def test_stable_failure_rerolls_only_after_full_wait(self):
        io = FakeIO()
        self.assertEqual(run(io, fake_reader([BAD, BAD, GOOD, GOOD]), Mock()), GOOD)
        self.assertEqual(io.clicks, 2)
        self.assertEqual(io.waits, [2.75, .2, 2.75, .2])

    def test_unknown_does_not_discard_roll(self):
        io = FakeIO()
        reader = fake_reader([dict(GOOD, AGI=None)] * 4)
        with self.assertRaisesRegex(RuntimeError, "UNKNOWN"):
            run(io, reader, Mock())
        self.assertEqual(io.clicks, 1)

    def test_disagreement_does_not_claim_success_or_reroll(self):
        io = FakeIO()
        with self.assertRaisesRegex(RuntimeError, "unstable"):
            run(io, fake_reader([GOOD, BAD, GOOD, BAD]), Mock())
        self.assertEqual(io.clicks, 1)

    def test_unknown_can_recover_by_reading_only(self):
        io = FakeIO()
        self.assertEqual(run(io, fake_reader([dict(GOOD, TAL=None), GOOD, GOOD]), Mock()), GOOD)
        self.assertEqual(io.clicks, 1)

    def test_unverified_ui_never_clicks(self):
        io = FakeIO()
        reader = fake_reader([])
        reader.check_panel.side_effect = RuntimeError("wrong screen")
        with self.assertRaisesRegex(RuntimeError, "wrong screen"):
            run(io, reader, Mock())
        self.assertEqual(io.clicks, 0)

    def test_f9_and_focus_loss_interrupt_wait_before_read(self):
        for exception in (StopRequested("F9"), RuntimeError("lost focus")):
            io = FakeIO()
            io.wait = Mock(side_effect=exception)
            reader = fake_reader([])
            with self.assertRaises(type(exception)):
                run(io, reader, Mock())
            self.assertEqual(io.clicks, 1)
            reader.read.assert_not_called()

    def test_roll_budget_stops_without_extra_click(self):
        io = FakeIO()
        with self.assertRaisesRegex(RuntimeError, "Roll limit"):
            run(io, fake_reader([BAD, BAD]), Mock(), replace(SETTINGS, max_rolls=1))
        self.assertEqual(io.clicks, 1)

    def test_interruptible_wait_checks_every_25ms(self):
        now = [0.0]
        intervals = []
        def sleep(seconds):
            intervals.append(seconds)
            now[0] += seconds
        wait_checked(2.75, Mock(), lambda: now[0], sleep)
        self.assertAlmostEqual(now[0], 2.75)
        self.assertLessEqual(max(intervals), .025)
        def stop():
            if now[0] >= 2.8:
                raise StopRequested("F9")
        with self.assertRaises(StopRequested):
            wait_checked(2.75, stop, lambda: now[0], sleep)
        self.assertLess(now[0], 2.85)

    def test_click_releases_on_stop_during_hold(self):
        window, mouse = Mock(), Mock()
        window.geometry.return_value = (43, 95, 1280, 600)
        io = LiveIO(window, mouse)
        io.wait = Mock(side_effect=StopRequested("F9"))
        with self.assertRaises(StopRequested):
            io.click_random()
        mouse.moveTo.assert_called_once_with(738, 462, _pause=False)
        mouse.mouseUp.assert_called_once_with(button="left", _pause=False)

    def test_wrong_client_size_prevents_input(self):
        window, mouse = Mock(), Mock()
        window.geometry.return_value = (0, 0, 1920, 1080)
        with self.assertRaisesRegex(RuntimeError, "Expected client"):
            LiveIO(window, mouse).click_random()
        mouse.mouseDown.assert_not_called()


class NativeReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = Reader(ROOT)

    def panel(self):
        # Test composition only: real glyphs at the documented client offsets.
        image = Image.new("RGB", (155, 180), "white")
        fixtures = ROOT / "tests/fixtures/stat_new"
        for name, file in {"AGI": "1_AGI_10.png", "CHA": "1_CHA_5.png"}.items():
            with Image.open(fixtures / file) as crop:
                image.paste(crop, STAT_BOXES[name][:2])
        with Image.open(fixtures / "random_stats_normal.png") as button:
            image.paste(button, BUTTON_BOX[:2])
        return image

    def test_actual_native_font_crops_and_coordinates(self):
        values, detail = self.reader.read(self.panel())
        self.assertEqual(values, BAD)
        self.assertGreaterEqual(detail["AGI"]["score"], .97)

    def test_previous_holdout_eight_accepted_by_live_profile(self):
        with Image.open(ROOT / "tests/fixtures/stat_new/4_LCK_8.png") as image:
            self.assertEqual(self.reader.model.read(image).value, 8)

    def test_live_failure_eight_is_training_regression_not_holdout(self):
        sample, = load_samples(ROOT / "tests/fixtures/stat_new_live")
        self.assertEqual(sample.split, "development")
        prediction = self.reader.model.read(sample.image)
        self.assertEqual(prediction.value, 8)
        self.assertGreater(prediction.score, .99)

    def test_lower_threshold_alone_recovers_failed_eight(self):
        development = [s for s in load_samples(ROOT / "tests/fixtures/stat_new")
                       if s.split == "development"]
        sample, = load_samples(ROOT / "tests/fixtures/stat_new_live")
        old = NativeTemplates(development, "whole", Policy(.97, .12))
        relaxed = NativeTemplates(development, "whole", Policy(.96, .12))
        self.assertIsNone(old.read(sample.image).value)
        self.assertEqual(relaxed.read(sample.image).value, 8)

    def test_existing_corpus_has_no_numeric_misreads_with_live_profile(self):
        # Mixed development/evaluation regression, not an independent accuracy estimate.
        for sample in load_samples(ROOT / "tests/fixtures/stat_new"):
            with self.subTest(sample=sample.name):
                self.assertEqual(self.reader.model.read(sample.image).value, sample.value)

    def test_real_non_numeric_tiles_rejected(self):
        folder = ROOT / "tests/fixtures/stat_new_live"
        spec = json.loads((folder / "manifest.json").read_text())["negative_atlas"]
        with Image.open(folder / spec["file"]) as atlas:
            for i in range(len(spec["panel_boxes"])):
                with self.subTest(tile=i):
                    tile = atlas.crop((0, 18*i, 24, 18*(i+1)))
                    self.assertIsNone(self.reader.model.read(tile).value)

    def test_live_profile_keeps_margin_and_frozen_offline_policy(self):
        frozen = json.loads((ROOT / "offline/new_stat_policy.json").read_text())
        self.assertEqual(self.reader.model.policy, Policy(.96, .12))
        self.assertEqual(frozen["methods"]["whole"]["selected"]["threshold"], .97)

    def test_blank_screen_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "screen not verified"):
            self.reader.read(Image.new("RGB", (155, 180), "white"))


if __name__ == "__main__":
    unittest.main()
