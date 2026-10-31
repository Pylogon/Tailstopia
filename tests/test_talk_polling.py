"""Talk polling observes only after the delay and never retries on a poll miss."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.mission_config import load_mission_config, validate_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self):
        self.time = 0.0
        self.clicks, self.captures, self.events = [], [], []
        self.clicked_at = None
        self.ready = lambda:False
        self.stop_at = None
        self.error = RuntimeError("Game lost focus")
        self.capture_cost = 0.0
        self.talk_found = True

    def now(self):
        return self.time

    def sleep(self,seconds):
        self.time += seconds

    def check(self):
        if self.stop_at is not None and self.time >= self.stop_at-1e-9:
            raise self.error

    def event(self,kind,**fields):
        self.events.append({"kind":kind,"at":self.time,**fields})

    def capture(self,target):
        self.captures.append((self.time,target))
        self.time += self.capture_cost

    def click(self,point,label):
        self.clicks.append((self.time,point,label))
        self.time += .05
        self.clicked_at = self.time


class TalkPollingTests(unittest.TestCase):
    def setUp(self):
        self.config,self.base = load_mission_config(ROOT/"examples/mission_flow.json")
        self.session = FakeSession()
        self.flow = MissionFlow.__new__(MissionFlow)
        self.flow.session,self.flow.config = self.session,self.config
        self.flow.state,self.flow.deadline = "OPEN_MISSION",35
        self.flow.timings = {}
        def match(name,capture):
            capture(name)
            found = self.session.ready() if name == "room_list" else self.session.talk_found
            return SimpleNamespace(found=found,score=1.0 if found else 0.0,
                                   point=(550,300) if name == "talk" else (1120,386))
        self.flow.vision = SimpleNamespace(match=match)

    def room_times(self):
        return [at for at,target in self.session.captures if target == "room_list"]

    def test_already_open_returns_without_click_or_wait(self):
        self.session.ready = lambda:True
        self.flow.open_mission()
        self.assertEqual(self.session.clicks,[])
        self.assertEqual(self.room_times(),[0])
        self.assertEqual(self.session.now(),0)

    def test_first_poll_succeeds_at_five_seconds_with_no_extra_capture(self):
        self.session.ready = lambda:self.session.clicked_at is not None
        self.flow.open_mission()
        self.assertEqual(len(self.session.clicks),1)
        self.assertEqual(len(self.room_times()),2)  # Entry guard + first poll only.
        self.assertAlmostEqual(self.room_times()[1],5.05)
        self.assertAlmostEqual(self.session.now(),5.05)
        self.assertFalse(any(e["kind"] == "retry" for e in self.session.events))

    def test_late_room_list_is_found_inside_window_without_reclick_or_fallback(self):
        self.session.ready = lambda:self.session.now() >= 7.3
        self.flow.open_mission()
        self.assertEqual(len(self.session.clicks),1)
        self.assertEqual(len(self.room_times()),7)
        self.assertAlmostEqual(self.session.now(),7.55)
        self.assertFalse(any(target == "talk" for at,target in self.session.captures))
        self.assertFalse(any(e["kind"] == "retry" for e in self.session.events))

    def test_absent_target_polls_for_full_five_seconds_after_initial_delay(self):
        self.assertFalse(self.flow.wait_room_list(5,5))
        self.assertAlmostEqual(self.session.now(),10)
        self.assertEqual(len(self.room_times()),11)
        for actual,expected in zip(self.room_times(),[5+i*.5 for i in range(11)]):
            self.assertAlmostEqual(actual,expected)
        self.assertEqual(self.session.clicks,[])

    def test_poll_interval_initial_delay_and_window_are_configurable(self):
        self.config["talk"].update(first_wait=3,poll_timeout=5,poll_interval=1)
        self.session.ready = lambda:self.session.now() >= 6
        self.assertTrue(self.flow.wait_room_list(self.config["talk"]["first_wait"],self.config["talk"]["poll_timeout"]))
        self.assertEqual(len(self.room_times()),4)
        for actual,expected in zip(self.room_times(),[3,4,5,6]):
            self.assertAlmostEqual(actual,expected)
        self.assertAlmostEqual(self.session.now(),6)

    def test_fallback_starts_only_after_window_and_clicks_verified_text_directly(self):
        self.session.ready = lambda:len(self.session.clicks) >= 2
        self.flow.open_mission()
        self.assertEqual(len(self.session.clicks),2)
        first,second = self.session.clicks
        self.assertEqual(first[1],self.config["talk"]["point_client"])
        self.assertEqual(second[1],(550,300))
        self.assertGreaterEqual(second[0]-first[0]+1e-9,.05+5+5)
        retries = [e for e in self.session.events if e["kind"] == "retry"]
        self.assertEqual(len(retries),1)
        self.assertGreaterEqual(retries[0]["at"]+1e-9,10.05)
        self.assertEqual(sum(target == "talk" for at,target in self.session.captures),1)
        # Successful fallback retains its original settle time and one final capture.
        self.assertAlmostEqual(self.session.now()-(second[0]+.05),self.config["talk"]["fallback_wait"])
        self.assertEqual(len(self.room_times()),13)  # Entry + eleven polls + fallback result.

    def test_last_poll_success_prevents_fallback_capture_and_click(self):
        self.session.ready = lambda:self.session.now() >= 10.04
        self.flow.open_mission()
        self.assertEqual(len(self.session.clicks),1)
        self.assertAlmostEqual(self.session.now(),10.05)
        self.assertEqual(len(self.room_times()),12)  # Entry + eleven polls.
        self.assertFalse(any(target == "talk" for at,target in self.session.captures))

    def test_missing_talk_template_sends_no_fallback_click(self):
        self.session.talk_found = False
        with self.assertRaisesRegex(RuntimeError,"Cannot find talk; no click sent"):
            self.flow.open_mission()
        self.assertEqual(len(self.session.clicks),1)
        self.assertAlmostEqual(self.session.now(),10.05)
        self.assertEqual(sum(target == "talk" for at,target in self.session.captures),1)

    def test_fallback_result_uses_original_maximum_wait_and_never_clicks_again(self):
        with self.assertRaisesRegex(RuntimeError,"Talk fallback did not open Room List"):
            self.flow.open_mission()
        self.assertEqual(len(self.session.clicks),2)
        last_click_end = self.session.clicks[-1][0]+.05
        self.assertAlmostEqual(self.session.now()-last_click_end,self.config["talk"]["fallback_wait"])
        self.assertEqual(self.session.clicks[1][1],(550,300))

    def test_slow_capture_does_not_spin_or_schedule_more_than_eleven_poll_captures(self):
        self.session.capture_cost = .1
        self.assertFalse(self.flow.wait_room_list(5,5))
        self.assertLessEqual(len(self.room_times()),11)
        self.assertGreaterEqual(self.session.now(),10)
        self.assertLessEqual(self.session.now(),10.100001)
        self.assertEqual(len(set(self.room_times())),len(self.room_times()))
        self.assertEqual(self.session.clicks,[])

    def test_focus_loss_or_f9_between_polls_stops_without_another_capture_or_click(self):
        for error in (RuntimeError("Game lost focus"),StopRequested("F9")):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                self.session.stop_at,self.session.error = 5.2,error
                with self.assertRaises(type(error)):
                    self.flow.open_mission()
                self.assertEqual(len(self.session.clicks),1)
                self.assertEqual(len(self.room_times()),2)  # Entry + first unsuccessful poll.

    def test_state_deadline_still_interrupts_polling_before_fallback(self):
        self.flow.deadline = 7
        with self.assertRaises(TimeoutError):
            self.flow.open_mission()
        self.assertEqual(len(self.session.clicks),1)
        self.assertTrue(all(at < 7 for at in self.room_times()))
        self.assertFalse(any(target == "talk" for at,target in self.session.captures))

    def test_invalid_poll_values_are_rejected(self):
        for key,value in (("poll_interval",0),("poll_interval",-1),("poll_interval",True),
                          ("poll_interval",float("nan")),("poll_timeout",0),("poll_timeout",-1)):
            with self.subTest(key=key,value=value):
                config,_ = load_mission_config(ROOT/"examples/mission_flow.json")
                config["talk"][key] = value
                with self.assertRaises(ValueError):
                    validate_mission_config(config)

    def test_state_timeout_during_initial_delay_safe_stops_and_saves_last_frame(self):
        self.config["max_loops"] = 1
        self.config["timeouts"]["OPEN_MISSION"] = 4
        vision = MissionVision(self.config,self.base)
        class Counting(FixtureSession):
            open_captures = 0
            def capture(session,box):
                if session.state == "OPEN_MISSION":session.open_captures += 1
                return super().capture(box)
        with tempfile.TemporaryDirectory() as directory:
            session = Counting(self.config,directory,ROOT/"resource")
            with session.log,contextlib.redirect_stdout(io.StringIO()):
                result = MissionFlow(self.config,self.base,session,vision).run()
            rows = [json.loads(line) for line in (Path(directory)/"execution.jsonl").read_text().splitlines()]
            self.assertEqual((result.status,result.state),("FAIL","OPEN_MISSION"))
            self.assertIn("TimeoutError",result.reason)
            self.assertEqual(session.open_captures,1)
            self.assertEqual(sum(r["kind"] == "click" and r["label"] == "talk" for r in rows),1)
            self.assertTrue(list(Path(directory).glob("*_failure.png")))
            self.assertFalse(session.api.held_keys or session.api.held_buttons)
            self.assertFalse(any(r["kind"] == "recording_start" and r["recording"] == "mission" for r in rows))
            if session.last_image is not None:session.last_image.close()


if __name__ == "__main__":
    unittest.main()
