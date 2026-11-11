"""Offline safety tests: mocked inputs/clock and real native-font number crops."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import Mock

from PIL import Image

from talesoftails_auto_reroll import (BUTTON_BOX, LiveIO, Reader, SETTINGS,
                                    STAT_BOXES, run, select_game_window, target_met, wait_checked)
from tailstopia_macro_replay import StopRequested


ROOT = Path(__file__).resolve().parents[1]
GOOD = {"AGI": 12, "LCK": 10, "TAL": 8}
BAD = {"AGI": 10, "LCK": 9, "TAL": 7}


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
        for values in (GOOD, dict(AGI=12, LCK=9, TAL=9), dict(AGI=12, LCK=9, TAL=10)):
            self.assertTrue(target_met(values))
        for values in (dict(AGI=12, LCK=12, TAL=7), dict(AGI=12, LCK=9, TAL=8),
                       dict(AGI=None, LCK=12, TAL=7), dict(AGI=True, LCK=22, TAL=7)):
            self.assertFalse(target_met(values))

    def test_agi_must_equal_twelve_even_if_both_sums_pass(self):
        for agi in (11, 13):
            self.assertFalse(target_met(dict(AGI=agi, TAL=10, LCK=10)))

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

    def panel(self, source="1", lck="9"):
        # Test composition only: real glyphs at the documented client offsets.
        image = Image.new("RGB", (155, 180), "white")
        fixtures = ROOT / "tests/fixtures/stat_new"
        for name, file in {"AGI": "1_AGI_10.png", "TAL": "1_TAL_7.png",
                           "LCK": f"{source}_LCK_{lck}.png"}.items():
            with Image.open(fixtures / file) as crop:
                image.paste(crop, STAT_BOXES[name][:2])
        with Image.open(fixtures / "random_stats_normal.png") as button:
            image.paste(button, BUTTON_BOX[:2])
        return image

    def test_actual_native_font_crops_and_coordinates(self):
        values, detail = self.reader.read(self.panel())
        self.assertEqual(values, BAD)
        self.assertGreaterEqual(detail["AGI"]["score"], .97)

    def test_known_holdout_eight_remains_unknown(self):
        values, _ = self.reader.read(self.panel("4", "8"))
        self.assertIsNone(values["LCK"])

    def test_blank_screen_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "screen not verified"):
            self.reader.read(Image.new("RGB", (155, 180), "white"))


if __name__ == "__main__":
    unittest.main()
