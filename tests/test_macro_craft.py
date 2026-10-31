import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import macro_craft
from tailstopia_macro_replay import recording, StopRequested


class FakeAPI:
    FAILSAFE = True
    def __init__(self):
        self.now = 0.0
        self.buttons = set()
        self.calls = []
    def clock(self):
        return self.now
    def sleep(self,seconds):
        self.now += seconds
    def failSafeCheck(self):
        pass
    def mouseDown(self,button,x=None,y=None,**kwargs):
        self.buttons.add(button)
        self.calls.append((self.now,"down",button,x,y))
    def mouseUp(self,button,x=None,y=None,**kwargs):
        self.buttons.discard(button)
        self.calls.append((self.now,"up",button,x,y))


class CraftTests(unittest.TestCase):
    def setUp(self):
        self.actions,self.duration = recording(macro_craft.RECORDING_PATH)
        self.api = FakeAPI()

    def run_craft(self,loops=3,delay=0,check=lambda:None):
        with contextlib.redirect_stdout(io.StringIO()):
            return macro_craft.run_loops(self.actions,self.duration,self.api,check,loops,delay,
                                         clock=self.api.clock,sleep=self.api.sleep)

    def test_recording_loops_preserve_clicks_holds_timestamps_and_original_coordinates(self):
        self.assertEqual(len(self.actions),4)  # Six JSON rows include two hold summaries.
        self.assertEqual(self.run_craft(3,.25),3)
        self.assertEqual(len(self.api.calls),12)
        expected = [(0.6627,"down","left",885,417),(.788,"up","left",885,417),
                    (3.2389,"down","left",884,526),(3.335,"up","left",884,526)]
        for index in range(3):
            for actual,recorded in zip(self.api.calls[index*4:(index+1)*4],expected):
                self.assertAlmostEqual(actual[0],index*(4.4731+.25)+recorded[0])
                self.assertEqual(actual[1:],recorded[1:])
        self.assertAlmostEqual(self.api.now,3*4.4731+2*.25)
        self.assertFalse(self.api.buttons)

    def test_f9_mid_hold_releases_mouse_and_does_not_start_another_loop(self):
        def check():
            if self.api.now >= .7:
                raise StopRequested("F9 pressed")
        with self.assertRaises(StopRequested):
            self.run_craft(check=check)
        self.assertFalse(self.api.buttons)
        self.assertTrue(self.api.FAILSAFE)
        self.assertEqual([call[1] for call in self.api.calls],["down","up"])
        self.assertLess(self.api.now,.72)

    def test_focus_loss_mid_hold_releases_mouse(self):
        def check():
            if self.api.buttons:
                raise RuntimeError("Game lost focus or minimized")
        with self.assertRaisesRegex(RuntimeError,"lost focus"):
            self.run_craft(check=check)
        self.assertFalse(self.api.buttons)
        self.assertEqual([call[1] for call in self.api.calls],["down","up"])

    def test_f9_during_extra_loop_delay_stops_before_next_loop(self):
        def check():
            if self.api.now > 4.6:
                raise StopRequested("F9 pressed")
        with self.assertRaises(StopRequested):
            self.run_craft(delay=10,check=check)
        self.assertEqual(len(self.api.calls),4)
        self.assertFalse(self.api.buttons)
        self.assertLess(self.api.now,4.62)

    def test_held_f8_starts_one_batch_only(self):
        class Window:
            f8 = iter([False,True,True,True])
            f9 = iter([False,False,True])
            activations = 0
            def pressed(self,vk):
                return next(self.f8 if vk == 0x77 else self.f9)
            def activate(self):
                self.activations += 1
            def check(self):
                pass
        window = Window()
        with patch("macro_craft.run_loops",return_value=5) as runner, \
             patch("macro_craft.time.sleep"),contextlib.redirect_stdout(io.StringIO()):
            macro_craft.hotkeys(window,self.api,self.actions,self.duration,5,0)
        self.assertEqual(window.activations,1)
        runner.assert_called_once()

    def test_dry_run_cli_sends_no_input_and_finishes_two_loops(self):
        out = io.StringIO()
        with patch("sys.argv",["macro_craft.py","--dry-run","--loops","2"]),contextlib.redirect_stdout(out):
            self.assertEqual(macro_craft.main(),0)
        self.assertIn("Dry-run complete: 2 loops; virtual duration 8.9462s.",out.getvalue())

    def test_invalid_loop_count_or_delay_sends_no_input(self):
        for loops,delay in ((0,0),(-1,0),(True,0),(1,-1),(1,float("nan"))):
            with self.subTest(loops=loops,delay=delay),self.assertRaises(ValueError):
                self.run_craft(loops,delay)
        self.assertFalse(self.api.calls)


if __name__ == "__main__":
    unittest.main()
