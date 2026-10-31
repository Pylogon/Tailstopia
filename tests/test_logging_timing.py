"""Logging changes preserve input order, virtual timing and capture counts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tailstopia_macro_replay import replay
from tailstopia_workflow.logger import ExecutionLog
from tailstopia_workflow.mission_config import load_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT = Path(__file__).resolve().parents[1]


class LoggingTests(unittest.TestCase):
    def test_persistent_handle_opens_once_and_flushes_every_event(self):
        with tempfile.TemporaryDirectory() as directory:
            log = ExecutionLog(directory)
            path = Path(directory)/"execution.jsonl"
            original_open = Path.open
            appends = []
            def tracked_open(target,*args,**kwargs):
                if target == path and args and args[0] == "a":
                    appends.append(target)
                return original_open(target,*args,**kwargs)
            with patch.object(Path,"open",tracked_open):
                with log:
                    stream = log._stream
                    for index in range(3):
                        log.emit("action",index=index)
                        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                        self.assertEqual([row["index"] for row in rows],list(range(index+1)))
                self.assertTrue(stream.closed)
                self.assertIsNone(log._stream)
            self.assertEqual(len(appends),1)

    def test_exception_closes_handle_and_one_shot_callers_still_append(self):
        with tempfile.TemporaryDirectory() as directory:
            log = ExecutionLog(directory)
            with self.assertRaisesRegex(RuntimeError,"injected"):
                with log:
                    stream = log._stream
                    log.emit("before_failure")
                    raise RuntimeError("injected")
            self.assertTrue(stream.closed)
            log.emit("after_failure")
            self.assertIsNone(log._stream)
            rows = [json.loads(line) for line in (Path(directory)/"execution.jsonl").read_text().splitlines()]
            self.assertEqual([row["kind"] for row in rows],["before_failure","after_failure"])

    def test_timing_is_aggregated_resettable_and_does_not_emit_extra_events(self):
        with tempfile.TemporaryDirectory() as directory:
            log = ExecutionLog(directory)
            with log, patch("tailstopia_workflow.logger.time.perf_counter",side_effect=[1,1.02,2,2.03]):
                log.emit("first")
                log.emit("second")
            self.assertEqual(log.timing_summary(),{"calls":2,"seconds":0.05})
            log.reset_timing()
            self.assertEqual(log.timing_summary(),{"calls":0,"seconds":0.0})
            self.assertEqual(len((Path(directory)/"execution.jsonl").read_text().splitlines()),2)

    def test_unserializable_event_still_closes_persistent_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            log = ExecutionLog(directory)
            with self.assertRaises(TypeError):
                with log:
                    stream = log._stream
                    log.emit("bad",value=object())
            self.assertTrue(stream.closed)
            self.assertIsNone(log._stream)


class TimingTests(unittest.TestCase):
    def test_vision_capture_and_processing_are_parts_of_existing_measurement(self):
        flow = MissionFlow.__new__(MissionFlow)
        flow.timings = {}
        captures = []
        def capture(box):
            captures.append(box)
            return "pixels"
        flow.session = SimpleNamespace(capture=capture)
        with patch("tailstopia_workflow.mission_flow.time.perf_counter",side_effect=[1,2,5,9]):
            value = flow.measure("vision",lambda take:take((1,2,3,4)),flow.capture)
        self.assertEqual(value,"pixels")
        self.assertEqual(captures,[(1,2,3,4)])
        self.assertEqual(flow.timings["vision"],{"calls":1,"seconds":8})
        self.assertEqual(flow.timings["vision_capture"],{"calls":1,"seconds":3})
        self.assertEqual(flow.timings["vision_processing"],{"calls":1,"seconds":5})

    def test_failed_capture_is_timed_once_and_propagates_original_error(self):
        flow = MissionFlow.__new__(MissionFlow)
        flow.timings = {}
        def capture(box):
            raise RuntimeError("lost focus")
        flow.session = SimpleNamespace(capture=capture)
        with self.assertRaisesRegex(RuntimeError,"lost focus"):
            flow.measure("vision",lambda take:take(None),flow.capture)
        for key in ("vision","vision_capture","vision_processing"):
            self.assertEqual(flow.timings[key]["calls"],1)

    def test_quiet_replay_keeps_dispatch_timestamps_callbacks_and_duration(self):
        actions = [{"type":"key_down","key":"a","t":0.1},{"type":"key_up","key":"a","t":0.3}]
        results = []
        for verbose in (True,False):
            now,calls,reports = [0.0],[],[]
            def sleep(seconds):
                now[0] += seconds
            def key_down(key,**kwargs):
                calls.append((now[0],"down",key))
            def key_up(key,**kwargs):
                calls.append((now[0],"up",key))
            api = SimpleNamespace(FAILSAFE=True,failSafeCheck=lambda:None,keyDown=key_down,keyUp=key_up)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                replay(actions,0.5,api,lambda:None,clock=lambda:now[0],sleep=sleep,
                       on_action=lambda action,at:reports.append((action,at)),verbose=verbose)
            results.append((calls,reports,now[0]))
            self.assertEqual(bool(output.getvalue()),verbose)
        self.assertEqual(results[0],results[1])
        self.assertEqual(results[0][0],[(0.1,"down","a"),(0.3,"up","a")])
        self.assertEqual(results[0][2],0.5)

    def test_mission_quiet_and_verbose_preserve_trace_and_capture_count(self):
        config,base = load_mission_config(ROOT/"examples/mission_flow.json")
        config["max_loops"] = 1
        vision = MissionVision(config,base)
        class Counting(FixtureSession):
            captures = 0
            def capture(self,box):
                self.captures += 1
                return super().capture(box)
        traces = []
        for verbose in (False,True):
            with tempfile.TemporaryDirectory() as directory:
                session = Counting(config,directory,ROOT/"resource")
                with session.log, contextlib.redirect_stdout(io.StringIO()):
                    result = MissionFlow(config,base,session,vision,verbose_actions=verbose).run("OPEN_ROOM_LIST")
                self.assertEqual(result.status,"SUCCESS")
                self.assertIsNone(session.log._stream)
                rows = [json.loads(line) for line in (Path(directory)/"execution.jsonl").read_text().splitlines()]
                states = [row for row in rows if row["kind"] == "state_result"]
                self.assertEqual(len(states),4)
                for row in states:
                    timing = row["timing"]
                    self.assertGreater(timing["logging"]["calls"],0)
                    if "vision" in timing:
                        self.assertEqual(timing["vision_capture"]["calls"],timing["vision"]["calls"])
                        self.assertAlmostEqual(timing["vision"]["seconds"],
                                               timing["vision_capture"]["seconds"]+timing["vision_processing"]["seconds"],places=5)
                for row in rows:
                    for key in ("time","wall_elapsed","timing"):
                        row.pop(key,None)
                traces.append((rows,session.captures,session.now()))
                if session.last_image is not None:
                    session.last_image.close()
        self.assertEqual(traces[0],traces[1])
        self.assertEqual(traces[0][1],7)


    def test_live_launcher_closes_log_and_releases_inputs_on_exception_or_f9(self):
        import tailstopia_mission
        from tailstopia_macro_replay import StopRequested
        config,base = load_mission_config(ROOT/"examples/mission_flow.json")
        for error in (RuntimeError("injected failure"),StopRequested("F9")):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                config["output"] = directory
                sessions,streams = [],[]
                class Window:
                    def __init__(self,*args):
                        self.f8,self.f9 = iter([False,True,False]),iter([False,True])
                    def pressed(self,vk):
                        return next(self.f8 if vk == 0x77 else self.f9)
                    def activate(self):
                        pass
                class FakeLive(FixtureSession):
                    def __init__(self,current_config,output,window):
                        super().__init__(current_config,output,ROOT/"resource")
                        sessions.append(self)
                class BrokenFlow:
                    def __init__(self,current_config,current_base,session,vision,**kwargs):
                        self.session = session
                    def run(self,through):
                        streams.append(self.session.log._stream)
                        self.session.api.keyDown("a")
                        raise error
                with patch("tailstopia_mission.load_mission_config",return_value=(config,base)), \
                     patch("tailstopia_workflow.game_window.GameWindow",Window), \
                     patch("tailstopia_workflow.mission_runtime.LiveSession",FakeLive), \
                     patch("tailstopia_workflow.mission_vision.MissionVision",return_value=object()), \
                     patch("tailstopia_mission.MissionFlow",BrokenFlow), \
                     patch("sys.argv",["tailstopia_mission.py"]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(tailstopia_mission.main(),0)
                self.assertEqual(len(streams),1)
                self.assertTrue(streams[0].closed)
                self.assertIsNone(sessions[0].log._stream)
                self.assertFalse(sessions[0].api.held_keys or sessions[0].api.held_buttons)
                path = next(Path(directory).glob("*/execution.jsonl"))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([row["method"] for row in rows if row["kind"] == "action"],["keyDown","keyUp"])


if __name__ == "__main__":
    unittest.main()
