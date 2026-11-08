"""Mouse-corner emergency regression without desktop input or live captures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import tailstopia_macro_replay
from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.mission_config import load_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession, InputTrace, SimulatedInput
from tailstopia_workflow.mission_vision import MissionVision

ROOT = Path(tailstopia_macro_replay.__file__).resolve().parent


class CornerBackend(SimulatedInput):
    class FailSafeException(Exception):
        pass

    def __init__(self):
        self.FAILSAFE = True
        self.corner = False
        self.sent = []
        self.position = Mock(return_value=(0, 0))

    def failSafeCheck(self):
        if self.FAILSAFE and self.corner:
            raise self.FailSafeException("original PyAutoGUI diagnostic")

    def keyDown(self, *args, **kwargs):
        self.failSafeCheck()
        self.sent.append(("keyDown", args, kwargs))

    def keyUp(self, *args, **kwargs):
        self.failSafeCheck()
        self.sent.append(("keyUp", args, kwargs))

    def mouseDown(self, **kwargs):
        self.failSafeCheck()
        self.sent.append(("mouseDown", (), kwargs))

    def mouseUp(self, **kwargs):
        self.failSafeCheck()
        self.sent.append(("mouseUp", (), kwargs))

    def moveTo(self, *args, **kwargs):
        self.failSafeCheck()
        self.sent.append(("moveTo", args, kwargs))

    scroll = moveTo
    hscroll = moveTo


class MissionFailsafeTests(unittest.TestCase):
    def test_guard_converts_emergency_to_stop_and_logs_position_once(self):
        backend, events = CornerBackend(), []
        trace = InputTrace(backend, lambda kind, **kw: events.append((kind, kw)))
        backend.corner = True
        with self.assertRaisesRegex(StopRequested, "Mouse corner emergency stop") as error:
            trace.failSafeCheck()
        self.assertIsInstance(error.exception.__cause__, backend.FailSafeException)
        self.assertEqual(events[0][0], "emergency_stop")
        self.assertEqual(events[0][1]["mouse_position"], [0, 0])
        self.assertEqual(events[0][1]["operation"], "failSafeCheck")
        self.assertEqual(len(events), 1)
        self.assertTrue(backend.FAILSAFE)
        self.assertEqual(backend.sent, [])

    def test_no_pointer_queries_or_extra_input_during_normal_checks(self):
        backend = CornerBackend()
        event = Mock()
        trace = InputTrace(backend, event)
        for _ in range(20):
            trace.failSafeCheck()
        backend.position.assert_not_called()
        event.assert_not_called()
        self.assertEqual(backend.sent, [])

    def test_emergency_inside_each_input_operation_is_also_a_stop(self):
        operations = [("keyDown", ("a",), {}), ("keyUp", ("a",), {}),
                      ("mouseDown", (), {"button": "right"}), ("mouseUp", (), {"button": "right"}),
                      ("moveTo", (100, 200), {}), ("scroll", (1,), {}), ("hscroll", (1,), {})]
        for method, args, kwargs in operations:
            with self.subTest(method=method):
                backend, event = CornerBackend(), Mock()
                trace = InputTrace(backend, event)
                backend.corner = True
                with self.assertRaises(StopRequested):
                    getattr(trace, method)(*args, **kwargs)
                self.assertEqual(event.call_args.kwargs["operation"], method)
                self.assertEqual(backend.sent, [])
                trace.release_all()
                self.assertFalse(trace.held_keys or trace.held_buttons)
                self.assertTrue(backend.FAILSAFE)

    def test_failed_pointer_diagnostic_does_not_prevent_emergency_stop(self):
        backend, event = CornerBackend(), Mock()
        backend.corner = True
        backend.position.side_effect = OSError("position unavailable")
        with self.assertRaises(StopRequested):
            InputTrace(backend, event).failSafeCheck()
        self.assertIsNone(event.call_args.kwargs["mouse_position"])

    def test_unrelated_errors_remain_failures(self):
        for backend in (CornerBackend(), SimulatedInput()):
            with self.subTest(backend=type(backend).__name__):
                original = RuntimeError("input device error")
                backend.failSafeCheck = Mock(side_effect=original)
                event = Mock()
                with self.assertRaises(RuntimeError) as caught:
                    InputTrace(backend, event).failSafeCheck()
                self.assertIs(caught.exception, original)
                event.assert_not_called()

    def test_recorded_collect_wait_emergency_releases_inputs_and_stops_flow(self):
        config, base = load_mission_config(ROOT / "examples/mission_95.json")
        config["max_loops"] = 2
        vision = MissionVision(config, base)

        class Interrupted(FixtureSession):
            def enter(self, state, loop):
                super().enter(state, loop)
                if state == "COLLECT_REWARDS":
                    self.collect_at = self.now()
                    # Exercise cleanup even if a key/button happened to be held.
                    self.api.keyDown("a")
                    self.api.mouseDown(button="right")

            def check(self):
                if self.state == "COLLECT_REWARDS" and self.now()-self.collect_at >= .7:
                    self.api.backend.corner = True
                self.api.failSafeCheck()

        with tempfile.TemporaryDirectory() as directory:
            session = Interrupted(config, directory, ROOT / "resource")
            session.api.backend = CornerBackend()
            with session.log, contextlib.redirect_stdout(io.StringIO()):
                result = MissionFlow(config, base, session, vision).run()
            rows = [json.loads(line) for line in (Path(directory) / "execution.jsonl").read_text().splitlines()]
            self.assertTrue(result.stopped)
            self.assertEqual(result.state, "COLLECT_REWARDS")
            self.assertEqual(result.loops_completed, 0)
            self.assertEqual(sum(e["kind"] == "emergency_stop" for e in rows), 1)
            self.assertTrue(rows[-1]["stopped"])
            self.assertFalse(any(e["kind"] == "click" and
                                 (e["label"].startswith("reward_") or e["label"] == "camp") for e in rows))
            self.assertFalse(any(e["kind"] == "state_entry" and e["loop"] == 2 for e in rows))
            self.assertTrue(any(e["kind"] == "debug_image" and "_stopped.png" in e["path"] for e in rows))
            self.assertFalse(session.api.held_keys or session.api.held_buttons)
            self.assertTrue(session.api.FAILSAFE)
            if session.last_image is not None:
                session.last_image.close()


if __name__ == "__main__":
    unittest.main()
