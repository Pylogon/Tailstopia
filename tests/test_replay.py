from pathlib import Path
import unittest
from unittest.mock import patch
from tailstopia_macro_replay import recording, replay, validate_inputs, DEFAULT_RECORDING, StopRequested


class FakeAPI:
    KEYBOARD_KEYS = ["a", "s"]
    FAILSAFE = True
    def __init__(self):
        self.now = 0.0
        self.keys = set()
        self.buttons = set()
        self.calls = []
    def clock(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds
    def failSafeCheck(self):
        pass
    def keyDown(self, key, **kwargs):
        self.keys.add(key)
        self.calls.append((self.now, "key_down", key))
    def keyUp(self, key, **kwargs):
        self.keys.discard(key)
        self.calls.append((self.now, "key_up", key))
    def mouseDown(self, button, **kwargs):
        self.buttons.add(button)
        self.calls.append((self.now, "mouse_down", button))
    def mouseUp(self, button, **kwargs):
        self.buttons.discard(button)
        self.calls.append((self.now, "mouse_up", button))


class ReplayTests(unittest.TestCase):
    def test_actual_recording_preserves_overlapping_holds(self):
        api = FakeAPI()
        actions, duration = recording(DEFAULT_RECORDING)
        self.assertEqual(len(actions), 6)
        seen_overlap = []
        def check():
            if api.keys == {"a","s"} and api.buttons == {"right"}:
                seen_overlap.append(True)
        replay(validate_inputs(actions, api), duration, api, check, api.clock, api.sleep)
        self.assertTrue(seen_overlap)
        self.assertFalse(api.keys or api.buttons)
        self.assertAlmostEqual(api.now, 7.6563)
        for call, event in zip(api.calls, actions):
            self.assertAlmostEqual(call[0], event["t"])
            self.assertEqual(call[1], event["type"])
        times = {(kind, value):t for t,kind,value in api.calls}
        self.assertAlmostEqual(times["key_up","a"] - times["key_down","a"], 2.0236)
        self.assertAlmostEqual(times["key_up","s"] - times["key_down","s"], 1.9932)
        self.assertAlmostEqual(times["mouse_up","right"] - times["mouse_down","right"], 2.7613)

    def test_quiet_replay_preserves_dispatch_and_callback(self):
        api = FakeAPI()
        actions, duration = recording(DEFAULT_RECORDING)
        dispatched = []
        with patch("builtins.print") as printer:
            replay(actions, duration, api, lambda: None, api.clock, api.sleep,
                   on_action=lambda action, at: dispatched.append((action, at)), verbose=False)
        printer.assert_not_called()
        self.assertEqual(len(dispatched), len(actions))
        self.assertEqual(len(api.calls), len(actions))
        for event, (at, kind, _) in zip(actions, api.calls):
            self.assertAlmostEqual(event["t"], at)
            self.assertEqual(event["type"], kind)
        self.assertFalse(api.keys or api.buttons)

    def test_stop_releases_overlapping_inputs(self):
        api = FakeAPI()
        actions, duration = recording(DEFAULT_RECORDING)
        def check():
            if api.now >= 3:
                raise StopRequested("F9 pressed")
        with self.assertRaises(StopRequested):
            replay(actions, duration, api, check, api.clock, api.sleep)
        self.assertFalse(api.keys or api.buttons)
        self.assertTrue(api.FAILSAFE)
        self.assertLess(api.now, 3.02)

    def test_focus_loss_releases_inputs(self):
        api = FakeAPI()
        actions, duration = recording(DEFAULT_RECORDING)
        def check():
            if api.buttons:
                raise RuntimeError("Game lost focus")
        with self.assertRaisesRegex(RuntimeError, "lost focus"):
            replay(actions, duration, api, check, api.clock, api.sleep)
        self.assertFalse(api.keys or api.buttons)

    def test_reserved_key_rejected(self):
        with self.assertRaises(ValueError):
            validate_inputs([{"type":"key_down","key":"f9"}], FakeAPI())
