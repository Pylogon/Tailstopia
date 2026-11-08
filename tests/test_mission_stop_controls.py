"""Mission live stop controls tested with a fake OS/input backend."""
import contextlib
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tailstopia_macro_replay import StopRequested, replay
from tailstopia_workflow.game_window import GameWindow
from tailstopia_workflow.mission_runtime import LiveSession, SimulatedInput


class CornerInput(SimulatedInput):
    FAILSAFE = True

    def __init__(self):
        self.sent = []

    def failSafeCheck(self):
        if self.FAILSAFE:
            raise RuntimeError("simulated pointer at screen corner")

    def keyDown(self, key, **kwargs):
        self.failSafeCheck()
        self.sent.append(("keyDown", key))

    def keyUp(self, key, **kwargs):
        self.failSafeCheck()
        self.sent.append(("keyUp", key))

    def mouseDown(self, **kwargs):
        self.failSafeCheck()
        self.sent.append(("mouseDown", kwargs.get("button", "left")))

    def mouseUp(self, **kwargs):
        self.failSafeCheck()
        self.sent.append(("mouseUp", kwargs.get("button", "left")))


class MissionStopControlTests(unittest.TestCase):
    def setUp(self):
        self.f9, self.focused = False, True
        window = GameWindow.__new__(GameWindow)
        window.hwnd = 123
        window.api = SimpleNamespace(
            GetAsyncKeyState=lambda vk: 0x8000 if vk == 0x78 and self.f9 else 0,
            GetForegroundWindow=lambda: 123 if self.focused else 456,
            IsIconic=lambda hwnd: False)
        self.backend = CornerInput()
        directory = self.enterContext(tempfile.TemporaryDirectory())
        # Exercise the real LiveSession constructor without importing PyAutoGUI.
        with patch.dict("sys.modules", {"pyautogui": self.backend}):
            self.session = LiveSession({"ui": {"click_hold": 0}}, directory, window)
        self.enterContext(self.session.log)

    def test_live_disables_mouse_corner_stop(self):
        self.assertFalse(self.backend.FAILSAFE)
        self.assertEqual(self.backend.PAUSE, 0)
        self.session.check()
        self.session.api.keyDown("a")
        self.session.api.mouseDown(button="right")
        self.session.api.release_all()
        self.assertFalse(self.session.api.held_keys or self.session.api.held_buttons)
        self.assertFalse(self.backend.FAILSAFE)

    def test_f9_still_stops_at_a_screen_corner(self):
        self.f9 = True
        with self.assertRaisesRegex(StopRequested, "F9 pressed"):
            self.session.check()
        self.assertEqual(self.backend.sent, [])

    def test_focus_loss_still_stops(self):
        self.focused = False
        with self.assertRaisesRegex(RuntimeError, "Game lost focus"):
            self.session.check()
        self.assertEqual(self.backend.sent, [])

    def test_f9_during_replay_releases_held_inputs_and_sends_no_next_action(self):
        actions = [{"t": 0, "type": "key_down", "key": "a"},
                   {"t": 0, "type": "mouse_down", "button": "right", "x": 100, "y": 100},
                   {"t": 10, "type": "key_down", "key": "s"}]
        clock = [0.0]
        def check():
            if self.session.api.held_keys and self.session.api.held_buttons:
                self.f9 = True
            self.session.check()
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(StopRequested, "F9 pressed"):
                replay(actions, 11, self.session.api, check, clock=lambda: clock[0],
                       sleep=lambda seconds: clock.__setitem__(0, clock[0]+seconds), verbose=False)
        self.assertEqual(self.backend.sent[:2], [("keyDown", "a"), ("mouseDown", "right")])
        self.assertCountEqual(self.backend.sent[2:], [("keyUp", "a"), ("mouseUp", "right")])
        self.assertFalse(self.session.api.held_keys or self.session.api.held_buttons)
        self.assertFalse(self.backend.FAILSAFE)


if __name__ == "__main__":
    unittest.main()
