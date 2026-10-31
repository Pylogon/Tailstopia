import copy
import csv
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from tailstopia_workflow.config import load_config, validate
from tailstopia_workflow.controller import Controller, Status
from tailstopia_workflow.control import Control, Stopped
from tailstopia_workflow.input_adapter import MockInputAdapter
from tailstopia_workflow.logger import ExecutionLog
from tailstopia_workflow.macro_parser import load_macro
from tailstopia_workflow.reward_selector import select_reward
from tailstopia_workflow.screen import MockScreen, RecordedScreen

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = load_config(ROOT / "examples/workflow.json")
        self.config["max_loops"] = 1
        self.config["output"] = str(self.directory)

    def run_flow(self, screen=None, realtime=False, control=None):
        log = ExecutionLog(self.directory)
        adapter = MockInputAdapter(log)
        result = Controller(self.config, adapter, screen or MockScreen(), log, control, realtime).run()
        events = [json.loads(line) for line in (self.directory / "execution.jsonl").read_text().splitlines()]
        self.assertFalse(adapter.held)
        return result, events

    def test_happy_path_and_no_action_replay_on_retry(self):
        self.config["max_loops"] = 2
        result, events = self.run_flow(MockScreen({"stage": [False, False, True]}))
        self.assertEqual(result.status, Status.SUCCESS)
        self.assertEqual(result.loops_completed, 2)
        self.assertEqual(sum(e["kind"] == "state_retry" for e in events), 2)
        self.assertEqual(sum(e["kind"] == "action" and e["state"] == "ENTER_STAGE" for e in events), 8)
        self.assertEqual(sum(e["kind"] == "action" and e["state"] == "PLAY_MACRO_A" for e in events), 4)

    def test_ocr_failure_contract(self):
        result, events = self.run_flow(MockScreen({"complete": ["OCR unavailable"]}))
        self.assertEqual((result.status, result.state), (Status.FAIL, "VERIFY_COMPLETE"))
        self.assertIn("OCR unavailable", result.reason)
        self.assertTrue(list(self.directory.glob("*_failure.png")))
        self.assertFalse(any(e.get("state") == "SELECT_REWARD" for e in events))

    def test_stage_timeout(self):
        policy = self.config["states"]["VERIFY_STAGE"]
        policy.update(timeout=0.02, retry_delay=0.1)
        result, events = self.run_flow(MockScreen({"stage": [False]}), realtime=True)
        self.assertEqual(result.state, "VERIFY_STAGE")
        self.assertIn("TimeoutError", result.reason)
        self.assertEqual(sum(e["kind"] == "action" and e["state"] == "ENTER_STAGE" for e in events), 4)

    def test_reward_missing_uses_fallback(self):
        result, events = self.run_flow()
        self.assertEqual(result.status, Status.SUCCESS)
        selection = next(e for e in events if e["kind"] == "reward_selection")
        self.assertEqual(selection["reason"], "configured fallback")
        self.assertEqual(selection["action"]["x"], 100)

    def test_reward_missing_without_fallback_stops(self):
        self.config["reward"]["fallback"] = None
        result, _ = self.run_flow()
        self.assertEqual((result.status, result.state), (Status.FAIL, "SELECT_REWARD"))

    def test_return_fail_never_replays_return(self):
        result, events = self.run_flow(MockScreen({"start": [True, False]}))
        self.assertEqual((result.status, result.state), (Status.FAIL, "RETURN_START"))
        self.assertEqual(result.loops_completed, 0)
        self.assertEqual(sum(e["kind"] == "action" and e["state"] == "RETURN_START" for e in events), 1)

    def test_stop_before_first_action(self):
        control = Control()
        control.stop()
        result, events = self.run_flow(control=control)
        self.assertEqual(result.status, Status.FAIL)
        self.assertFalse(any(e["kind"] == "action" for e in events))

    def test_invalid_second_macro_preflight(self):
        self.config["macro_b"] = str(self.directory / "missing.csv")
        result, events = self.run_flow()
        self.assertEqual(result.status, Status.FAIL)
        self.assertFalse(any(e["kind"] == "action" for e in events))

    def test_parser_summaries_and_pause_release(self):
        path = self.directory / "record.json"
        rows = [{"t":0,"type":"key_down","key":"w"},
                {"t":1,"type":"key_hold","key":"w","ended_by":"pause"},
                {"t":1,"type":"key_down","key":"a"},
                {"t":2,"type":"key_up","key":"a"},
                {"t":2,"type":"key_hold","key":"a","ended_by":"release"}]
        path.write_text(json.dumps({"events":rows}))
        events = load_macro(path)
        self.assertEqual([e["type"] for e in events], ["key_down","key_up","key_down","key_up"])
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=["t","type","key","ended_by"])
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual(load_macro(csv_path), events)

    def test_parser_rejects_bad_recordings(self):
        path = self.directory / "bad.json"
        for rows in ([{"t":-1,"type":"key_down","key":"w"}],
                     [{"t":0,"type":"key_down","key":"w"}],
                     [{"t":float("nan"),"type":"scroll"}],
                     [{"t":0,"type":"unknown"}]):
            path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                load_macro(path)

    def test_priority_and_unknown_fallback(self):
        action, reason = select_reward([{"name":"common","x":1,"y":2},{"name":"rare","x":3,"y":4}],
                                       ["rare","common"], None)
        self.assertEqual(action["x"], 3)
        self.assertIn("rare", reason)
        self.assertEqual(select_reward([{"name":"unknown"}], [], {"x":5,"y":6})[0]["x"], 5)

    def test_dry_run_honors_virtual_deadline(self):
        self.config["states"]["ENTER_STAGE"]["timeout"] = 0.05
        result, _ = self.run_flow()
        self.assertEqual((result.status, result.state), (Status.FAIL, "ENTER_STAGE"))
        self.assertIn("TimeoutError", result.reason)

    def test_recorded_failure_saves_original_frame(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Install Pillow for recorded-image tests")
        frame, template = self.directory / "frame.png", self.directory / "template.png"
        Image.new("RGB", (8,8), "white").save(frame)
        Image.new("RGB", (4,4), "black").save(template)
        screen = RecordedScreen({"start":[frame]},
                                {"start":{"template":str(template),"roi":[0,0,4,4],"max_error":0}})
        result, _ = self.run_flow(screen)
        self.assertEqual(result.status, Status.FAIL)
        with Image.open(next(self.directory.glob("*_failure.png"))) as image:
            self.assertEqual(image.size, (8,8))
            self.assertEqual(image.getpixel((0,0)), (255,255,255))

    def test_invalid_config(self):
        for key, value in (("max_loops",0),("max_loops",True)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                validate(config)

    def test_pause_resume_and_stop_while_paused(self):
        control = Control()
        control.pause()
        frozen = control.now()
        finished = threading.Event()
        worker = threading.Thread(target=lambda: (control.checkpoint(), finished.set()))
        worker.start()
        self.assertFalse(finished.wait(0.03))
        self.assertAlmostEqual(control.now(), frozen, places=3)
        control.resume()
        worker.join(1)
        self.assertTrue(finished.is_set())
        control.pause()
        stopped = threading.Event()
        def waiter():
            try:
                control.checkpoint()
            except Stopped:
                stopped.set()
        worker = threading.Thread(target=waiter)
        worker.start()
        control.stop()
        worker.join(1)
        self.assertTrue(stopped.is_set())

    def test_recorded_screenshot_workflow(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Install Pillow for recorded-image tests")
        frame = self.directory / "frame.png"
        template = self.directory / "template.png"
        Image.new("RGB", (12,12), "white").save(frame)
        Image.new("RGB", (4,4), "white").save(template)
        spec = {"template":str(template),"roi":[0,0,4,4],"max_error":0}
        screen = RecordedScreen({name:[frame] for name in ("start","stage","complete","rewards")},
                                {name:spec for name in ("start","stage","complete")},
                                [{**spec,"name":"rare","x":4,"y":5}])
        result, events = self.run_flow(screen)
        self.assertEqual(result.status, Status.SUCCESS)
        self.assertEqual(next(e for e in events if e["kind"] == "reward_scan")["items"][0]["name"], "rare")
        Image.new("RGB", (4,4), "black").save(template)
        self.assertFalse(screen.check("stage"))
        debug = self.directory / "debug.png"
        screen.save_debug(debug)
        with Image.open(debug) as image:
            self.assertEqual(image.size, (12,12))


if __name__ == "__main__":
    unittest.main()
