import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
try:
    from tailstopia_workflow.mission_config import load_mission_config, validate_mission_config
    from tailstopia_workflow.mission_flow import MissionFlow
    from tailstopia_workflow.mission_runtime import FixtureSession, LiveSession, Session, SimulatedInput
    from tailstopia_workflow.mission_vision import MissionVision
    from tailstopia_macro_replay import StopRequested
    AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name not in {"cv2","numpy"}:
        raise
    AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(AVAILABLE,"Use .venv Python with OpenCV/NumPy")
class MissionFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config,self.base = load_mission_config(ROOT/"examples/mission_flow.json")
        self.config["max_loops"] = 1
        self.vision = MissionVision(self.config,self.base)

    def run_case(self,kind=None,through=None):
        session = (kind or FixtureSession)(self.config,self.directory,ROOT/"resource")
        flow = MissionFlow(self.config,self.base,session,self.vision)
        with contextlib.redirect_stdout(io.StringIO()):
            result = flow.run(through)
        events = [json.loads(row) for row in (self.directory/"execution.jsonl").read_text().splitlines()]
        self.assertFalse(session.api.held_keys or session.api.held_buttons)
        return result,events,session

    def test_all_steps_happy_path_two_loops(self):
        self.config["max_loops"] = 2
        result,events,session = self.run_case()
        self.assertEqual((result.status,result.loops_completed),("SUCCESS",2))
        self.assertEqual(sum(e["kind"] == "recording_start" and e["recording"] == "start" for e in events),2)
        self.assertEqual(sum(e["kind"] == "recording_start" and e["recording"] == "mission" for e in events),2)
        self.assertEqual(sum(e["kind"] == "click" and e["label"].startswith("reward_") for e in events),14)
        self.assertEqual(sum(e["kind"] == "click" and e["label"] == "camp" for e in events),2)
        timers = [e for e in events if e["kind"] == "result_timer"]
        from tailstopia_macro_replay import recording
        _,duration = recording(self.base/self.config["mission_recording"])
        self.assertAlmostEqual(timers[0]["mission_elapsed"],duration)
        self.assertAlmostEqual(timers[0]["remaining"],max(0,self.config["result"]["after_mission_start"]-duration))
        mission = next(e["active_at"] for e in events if e["kind"] == "recording_start" and e["recording"] == "mission")
        first_result = next(e["active_at"] for e in events if e["state"] == "WAIT_RESULT" and e["kind"] == "recognition")
        self.assertAlmostEqual(first_result-mission,max(duration,self.config["result"]["after_mission_start"]))
        self.assertLess(session.now(),300)

    def test_talk_template_fallback_after_first_click_polling_expires(self):
        class Fallback(FixtureSession):
            def click_effect(self,label):
                if label == "talk" and self.click_counts.get(label,0) < 1:
                    self.click_counts[label] = self.click_counts.get(label,0)+1
                else:
                    super().click_effect(label)
        result,events,_ = self.run_case(Fallback,"OPEN_MISSION")
        self.assertEqual(result.status,"SUCCESS")
        clicks = [e for e in events if e["kind"] == "click" and e["label"] == "talk"]
        self.assertEqual(len(clicks),2)
        point = [a+b for a,b in zip(self.config["reference_client_origin"],self.config["talk"]["point_client"])]
        self.assertEqual(clicks[0]["desktop_point"],point)
        self.assertNotEqual(clicks[1]["desktop_point"],point)
        self.assertEqual(sum(e["kind"] == "recording_start" for e in events),1)

    def test_talk_absent_fails_without_random_fallback_click(self):
        class NoTalk(FixtureSession):
            def enter(self,state,loop):
                super().enter(state,loop)
                if state == "OPEN_MISSION":
                    self.frame = "intiate.png"
            def click_effect(self,label):
                if label != "talk":
                    super().click_effect(label)
        result,events,_ = self.run_case(NoTalk)
        self.assertEqual((result.status,result.state),("FAIL","OPEN_MISSION"))
        self.assertEqual(sum(e["kind"] == "click" and e["label"] == "talk" for e in events),1)
        self.assertFalse(any(e.get("recording") == "mission" for e in events))
        self.assertTrue(list(self.directory.glob("*_failure.png")))

    def test_room_list_retry_does_not_replay_macro(self):
        class Retry(FixtureSession):
            def click_effect(self,label):
                if label == "room_list" and not self.click_counts.get(label):
                    self.click_counts[label] = 1
                else:
                    super().click_effect(label)
        result,events,_ = self.run_case(Retry,"OPEN_ROOM_LIST")
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual(sum(e["kind"] == "click" and e["label"] == "room_list" for e in events),2)
        self.assertEqual(sum(e["kind"] == "recording_start" for e in events),1)

    def test_delayed_create_is_seen_before_repeat_click(self):
        class Delayed(FixtureSession):
            clicked_at = None
            def click_effect(self,label):
                if label == "room_list":
                    self.clicked_at = self.now()
                else:
                    super().click_effect(label)
            def capture(self,box):
                if self.state == "OPEN_ROOM_LIST" and self.clicked_at is not None and self.now()-self.clicked_at >= 1.5:
                    self.frame = "room list.png"
                return super().capture(box)
        result,events,_ = self.run_case(Delayed,"OPEN_ROOM_LIST")
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual(sum(e["kind"] == "click" and e["label"] == "room_list" for e in events),1)

    def test_create_retry_only(self):
        class Retry(FixtureSession):
            def click_effect(self,label):
                if label == "create" and not self.click_counts.get(label):
                    self.click_counts[label] = 1
                else:
                    super().click_effect(label)
        result,events,_ = self.run_case(Retry,"CREATE_ROOM")
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual(sum(e["kind"] == "click" and e["label"] == "create" for e in events),2)

    def test_stage_loading_timeout_never_reclicks_start(self):
        self.config["timeouts"]["WAIT_STAGE_LOADING"] = .6
        class NoLoading(FixtureSession):
            def enter(self,state,loop):
                super().enter(state,loop)
                if state == "WAIT_STAGE_LOADING":
                    self.frame = "mission.png"
        result,events,_ = self.run_case(NoLoading)
        self.assertEqual((result.status,result.state),("FAIL","WAIT_STAGE_LOADING"))
        self.assertIn("TimeoutError",result.reason)
        self.assertEqual(sum(e["kind"] == "click" and e["label"] == "start" for e in events),1)
        self.assertFalse(any(e.get("recording") == "mission" for e in events))

    def test_missed_loading_both_directions_completes_two_loops_without_repeating_actions(self):
        self.config["max_loops"] = 2
        class MissedLoading(FixtureSession):
            def enter(self,state,loop):
                super().enter(state,loop)
                if state in ("WAIT_STAGE_LOADING","WAIT_RETURN_LOADING"):
                    self.frame = "intiate.png"
        result,events,_ = self.run_case(MissedLoading)
        self.assertEqual((result.status,result.loops_completed),("SUCCESS",2))
        self.assertEqual(sum(e["kind"] == "loading_skipped" for e in events),4)
        for label in ("start","camp"):
            self.assertEqual(sum(e["kind"] == "click" and e["label"] == label for e in events),2)
        for name in ("start","mission"):
            self.assertEqual(sum(e["kind"] == "recording_start" and e["recording"] == name for e in events),2)
        self.assertEqual(sum(e["kind"] == "camera_calibrated" for e in events),2)
        for loop in (1,2):
            clicked = next(e["active_at"] for e in events if e["loop"] == loop and e["kind"] == "click" and e["label"] == "start")
            accepted = next(e["active_at"] for e in events if e["loop"] == loop and e["kind"] == "loading_skipped" and e["state"] == "WAIT_STAGE_LOADING")
            self.assertGreaterEqual(accepted-clicked,self.config["stage"]["min_entry_wait"])

    def test_stage_time_timeout_after_loading(self):
        self.config["stage"]["min_entry_wait"] = 0
        self.config["timeouts"]["VERIFY_STAGE"] = .6
        class NotReady(FixtureSession):
            def enter(self,state,loop):
                super().enter(state,loop)
                if state == "VERIFY_STAGE":
                    self.frame = "loading.png"
        result,_,_ = self.run_case(NotReady)
        self.assertEqual((result.status,result.state),("FAIL","VERIFY_STAGE"))

    def test_result_timeout_stops_before_rewards(self):
        self.config["result"]["after_mission_start"] = 31.4192
        self.config["timeouts"]["WAIT_RESULT"] = 1.1
        class NoResult(FixtureSession):
            def enter(self,state,loop):
                super().enter(state,loop)
                if state == "WAIT_RESULT":
                    self.frame = "startinagme .png"
        result,events,_ = self.run_case(NoResult)
        self.assertEqual((result.status,result.state),("FAIL","WAIT_RESULT"))
        self.assertFalse(any(e["kind"] == "click" and e["label"].startswith("reward_") for e in events))

    def test_reward_missing_retries_first_only_then_stops(self):
        class NoReward(FixtureSession):
            def click_effect(self,label):
                if label != "reward_1":
                    super().click_effect(label)
        result,events,_ = self.run_case(NoReward)
        self.assertEqual((result.status,result.state),("FAIL","COLLECT_REWARDS"))
        clicks = [e["label"] for e in events if e["kind"] == "click"]
        self.assertEqual(clicks.count("reward_1"),6)
        self.assertNotIn("reward_2",clicks)
        self.assertNotIn("camp",clicks)

    def test_delayed_pickup_prevents_duplicate_first_click(self):
        class Delayed(FixtureSession):
            pickup_at = None
            def click_effect(self,label):
                if label == "reward_1":
                    self.pickup_at = self.now()+.8
                else:
                    super().click_effect(label)
            def capture(self,box):
                if self.pickup_at is not None and self.now() >= self.pickup_at:
                    self.received = True
                return super().capture(box)
        result,events,_ = self.run_case(Delayed)
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual(sum(e["kind"] == "click" and e["label"] == "reward_1" for e in events),1)

    def test_existing_received_item_stops_before_first_click(self):
        class AlreadyOccupied(FixtureSession):
            def enter(self,state,loop):
                super().enter(state,loop)
                if state == "COLLECT_REWARDS":
                    self.received = True
        result,events,_ = self.run_case(AlreadyOccupied)
        self.assertEqual((result.status,result.state),("FAIL","COLLECT_REWARDS"))
        self.assertFalse(any(e["kind"] == "click" and e["label"] == "reward_1" for e in events))

    def test_return_fail_stops_before_second_loop(self):
        self.config["max_loops"] = 2
        self.config["timeouts"]["WAIT_RETURN_LOADING"] = .6
        class NoReturn(FixtureSession):
            def enter(self,state,loop):
                super().enter(state,loop)
                if state == "WAIT_RETURN_LOADING":
                    self.frame = "reward1.png"
        result,events,_ = self.run_case(NoReturn)
        self.assertEqual((result.status,result.state,result.loops_completed),("FAIL","WAIT_RETURN_LOADING",0))
        self.assertEqual(sum(e["kind"] == "click" and e["label"] == "camp" for e in events),1)
        self.assertFalse(any(e["loop"] == 2 for e in events))

    def test_stop_with_multiple_held_inputs_releases_all(self):
        class Cancelled(FixtureSession):
            def check(self):
                if len(self.api.held_keys) == 2 and self.api.held_buttons:
                    raise StopRequested("F9 requested")
        result,events,_ = self.run_case(Cancelled)
        self.assertTrue(result.stopped)
        self.assertEqual(result.state,"PLAY_START")
        self.assertFalse(any(e["kind"] == "click" for e in events))

    def test_recognition_exception_safe_stops(self):
        original = self.vision.match
        def broken(name,capture):
            if name == "room_list":
                raise ValueError("recognition failure injection")
            return original(name,capture)
        self.vision.match = broken
        result,events,_ = self.run_case()
        self.assertEqual((result.status,result.state),("FAIL","OPEN_MISSION"))
        self.assertIn("recognition failure injection",result.reason)
        self.assertFalse(any(e["kind"] == "click" for e in events))

    def test_invalid_mission_preflight_sends_nothing(self):
        self.config["mission_recording"] = "missing.json"
        session = FixtureSession(self.config,self.directory,ROOT/"resource")
        with self.assertRaises(FileNotFoundError):
            MissionFlow(self.config,self.base,session,self.vision)
        self.assertFalse(session.api.held_keys or session.api.held_buttons)
        self.assertFalse((self.directory/"execution.jsonl").exists())

    def test_templates_match_positive_and_reject_other_buttons(self):
        cases = {"room_list":"mission.png","create":"room list.png","start":"startmission.png",
                 "result":"reward1.png","camp":"reward1.png","talk":"red panda2.png"}
        for name,filename in cases.items():
            with self.subTest(name=name):
                with Image.open(ROOT/"resource"/filename) as source:
                    capture = lambda box:self.vision.crop_reference(source,box,[43,95])
                    self.assertTrue(self.vision.match(name,capture).found)
        for filename in ("room list.png","startmission.png","loading.png"):
            with Image.open(ROOT/"resource"/filename) as source:
                self.assertFalse(self.vision.match("room_list",lambda box:self.vision.crop_reference(source,box,[43,95])).found)

    def test_all_reward_variants_same_result_and_empty_slot(self):
        for filename in ("reward1.png","reward2.png","reward3.png","reward4.png"):
            with Image.open(ROOT/"resource"/filename) as source:
                self.assertTrue(self.vision.match("result",lambda box:self.vision.crop_reference(source,box,[43,95])).found)
                slot = self.vision.crop_reference(source,self.vision.reward_box,[43,95])
                self.assertTrue(self.vision.empty_slot(slot))
                self.assertFalse(self.vision.change(self.vision.empty,slot)[0])

    def test_invalid_limits_rejected(self):
        self.config["max_loops"] = 0
        with self.assertRaises(ValueError):
            validate_mission_config(self.config)

    def test_pause_waits_until_holds_end_and_freezes_clock(self):
        class FakeTime:
            current = 0.0
            def monotonic(self):
                return self.current
            def sleep(self,seconds):
                self.current += seconds
        class Window:
            values = iter([True,False,False,True])
            def pressed(self,vk):
                return next(self.values)
            def check(self):
                pass
        session = LiveSession.__new__(LiveSession)
        session.window = Window()
        session.paused_total,session.pause_started = 0,None
        session.pause_requested,session.was_f7 = False,False
        with patch("tailstopia_workflow.mission_runtime.time",FakeTime()):
            Session.__init__(session,self.config,self.directory,SimulatedInput())
            session.api.held_keys.add("a")
            with contextlib.redirect_stdout(io.StringIO()):
                session.check()
                self.assertTrue(session.pause_requested)
                self.assertEqual(session.api.held_keys,{"a"})
                session.api.keyUp("a")
                session.check()
            self.assertFalse(session.pause_requested)
            self.assertGreater(session.paused_total,0)
            self.assertAlmostEqual(session.now(),0)


    def test_transient_reward_change_is_not_accepted(self):
        class Transient(FixtureSession):
            count = 0
            def capture(self,box):
                if self.state == "COLLECT_REWARDS" and tuple(box) == tuple(self.config["reward"]["received_box_client"]):
                    if self.received:
                        self.count += 1
                        # Alternate a visible icon and an empty slot. This must
                        # not satisfy two consecutive pickup observations.
                        self.received = bool(self.count % 2)
                    image = super().capture(box)
                    if self.count:
                        self.received = True
                    return image
                return super().capture(box)
        result,events,_ = self.run_case(Transient)
        self.assertEqual((result.status,result.state),("FAIL","COLLECT_REWARDS"))
        self.assertFalse(any(e["kind"] == "reward_verified" for e in events))

    def test_f9_interrupts_pause(self):
        class Window:
            calls = 0
            def pressed(self,vk):
                return True
            def check(self):
                self.calls += 1
                if self.calls >= 2:
                    raise StopRequested("F9 in pause")
        session = LiveSession.__new__(LiveSession)
        session.window = Window()
        session.paused_total,session.pause_started = 0,None
        session.pause_requested,session.was_f7 = False,False
        Session.__init__(session,self.config,self.directory,SimulatedInput())
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(StopRequested):
                session.check()
        self.assertIsNone(session.pause_started)
        self.assertFalse(session.api.held_keys or session.api.held_buttons)


    def test_return_time_timeout_after_loading_prevents_next_loop(self):
        self.config["max_loops"] = 2
        self.config["timeouts"]["VERIFY_RETURN_START"] = .6
        class StillLoading(FixtureSession):
            def enter(self,state,loop):
                super().enter(state,loop)
                if state == "VERIFY_RETURN_START":
                    self.frame = "loading.png"
        result,events,_ = self.run_case(StillLoading)
        self.assertEqual((result.status,result.state,result.loops_completed),("FAIL","VERIFY_RETURN_START",0))
        self.assertFalse(any(e["loop"] == 2 for e in events))


    def test_live_create_hover_false_negative_regression(self):
        import copy
        original = copy.deepcopy(self.config)
        original["targets"]["create"].pop("variants")
        baseline = MissionVision(original,self.base)
        roi = Image.open(ROOT/"tests/fixtures/open_room_list_failure_roi.png").convert("RGB")
        self.addCleanup(roi.close)
        capture = lambda box:roi.copy()
        old = baseline.match("create",capture)
        self.assertFalse(old.found)
        self.assertAlmostEqual(old.score,0.852845,places=5)
        self.assertAlmostEqual(baseline.match("room_list",capture).score,0.335026,places=5)
        fixed = self.vision.match("create",capture)
        self.assertTrue(fixed.found)
        self.assertGreater(fixed.score,.999)
        self.assertEqual(fixed.point,(1120,386))
        self.assertEqual(self.config["targets"]["create"]["threshold"],.88)

    def test_live_hover_frame_advances_through_create_without_reclicking_room_list(self):
        class Hover(FixtureSession):
            live_create = False
            def click_effect(self,label):
                super().click_effect(label)
                if label == "room_list":
                    self.live_create = True
                elif label == "create":
                    self.live_create = False
            def capture(self,box):
                if self.live_create and tuple(box) == tuple(self.config["targets"]["create"]["search_roi_client"]):
                    with Image.open(ROOT/"tests/fixtures/open_room_list_failure_roi.png") as image:
                        return image.convert("RGB")
                return super().capture(box)
        result,events,_ = self.run_case(Hover,"CREATE_ROOM")
        self.assertEqual(result.status,"SUCCESS")
        for label in ("talk","room_list","create"):
            self.assertEqual(sum(e["kind"] == "click" and e["label"] == label for e in events),1)
        self.assertEqual(sum(e["kind"] == "recording_start" for e in events),1)

    def test_create_variants_reject_other_resource_screens(self):
        for path in (ROOT/"resource").glob("*.png"):
            if path.name == "room list.png":
                continue
            with self.subTest(image=path.name),Image.open(path) as source:
                result = self.vision.match("create",lambda box:self.vision.crop_reference(source,box,[43,95]))
                self.assertFalse(result.found)
                self.assertLess(result.score,.88)

    def test_both_ui_transitions_wait_for_target_when_source_temporarily_absent(self):
        cases = [("OPEN_ROOM_LIST","room_list","room list.png"),("CREATE_ROOM","create","startmission.png")]
        directory = self.directory
        for state,source,target_frame in cases:
            with self.subTest(state=state):
                self.directory = directory/state
                class Transition(FixtureSession):
                    def enter(self,current,loop):
                        super().enter(current,loop)
                        self.entered = self.now()
                    def capture(self,box):
                        if self.state == state:
                            self.frame = "loading.png" if self.now()-self.entered < 1 else target_frame
                        return super().capture(box)
                result,events,_ = self.run_case(Transition,state)
                self.assertEqual(result.status,"SUCCESS")
                self.assertFalse(any(e["kind"] == "click" and e["state"] == state for e in events))
                retries = [e for e in events if e["kind"] == "retry" and e["state"] == state]
                self.assertTrue(any("no click sent" in e["reason"] for e in retries))

    def test_both_ui_transitions_recover_source_before_clicking(self):
        directory = self.directory
        for state,source in (("OPEN_ROOM_LIST","room_list"),("CREATE_ROOM","create")):
            with self.subTest(state=state):
                self.directory = directory/state
                class Recovery(FixtureSession):
                    def enter(self,current,loop):
                        super().enter(current,loop)
                        self.entered = self.now()
                        self.recovered = False
                        self.source_frame = self.frame
                    def capture(self,box):
                        if self.state == state and not self.recovered:
                            if self.now()-self.entered < 1:
                                self.frame = "loading.png"
                            else:
                                self.frame = self.source_frame
                                self.recovered = True
                        return super().capture(box)
                result,events,_ = self.run_case(Recovery,state)
                self.assertEqual(result.status,"SUCCESS")
                clicks = [e for e in events if e["kind"] == "click" and e["state"] == state]
                self.assertEqual([e["label"] for e in clicks],[source])
                result_row = next(e for e in events if e["kind"] == "state_result" and e["state"] == state)
                self.assertAlmostEqual(result_row["elapsed"],2.05)

    def test_both_ui_transitions_exhaust_unknown_state_budget_without_clicking(self):
        directory = self.directory
        for state in ("OPEN_ROOM_LIST","CREATE_ROOM"):
            with self.subTest(state=state):
                self.directory = directory/state
                class Unknown(FixtureSession):
                    def enter(self,current,loop):
                        super().enter(current,loop)
                        if current == state:
                            self.frame = "loading.png"
                result,events,_ = self.run_case(Unknown)
                self.assertEqual((result.status,result.state),("FAIL",state))
                self.assertIn("UI retry budget exhausted",result.reason)
                self.assertFalse(any(e["kind"] == "click" and e["state"] == state for e in events))
                rows = [e for e in events if e["kind"] == "retry" and e["state"] == state]
                self.assertEqual(len(rows),3)
                self.assertTrue(all("source_score" in e and "target_score" in e for e in rows))
                self.assertTrue(list(self.directory.glob("*_failure.png")))

    def test_ui_transition_timeout_while_source_missing_is_safe(self):
        self.config["timeouts"]["OPEN_ROOM_LIST"] = .5
        class Unknown(FixtureSession):
            def enter(self,current,loop):
                super().enter(current,loop)
                if current == "OPEN_ROOM_LIST":
                    self.frame = "loading.png"
        result,events,_ = self.run_case(Unknown)
        self.assertEqual((result.status,result.state),("FAIL","OPEN_ROOM_LIST"))
        self.assertIn("TimeoutError",result.reason)
        self.assertFalse(any(e["kind"] == "click" and e["state"] == "OPEN_ROOM_LIST" for e in events))

    def test_state_timing_is_aggregated_without_extra_recognition(self):
        class Counting(FixtureSession):
            captures = 0
            def capture(self,box):
                self.captures += 1
                return super().capture(box)
        result,events,session = self.run_case(Counting,"OPEN_ROOM_LIST")
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual(session.captures,7)
        summaries = [e for e in events if "timing" in e]
        self.assertEqual(len(summaries),4)
        self.assertTrue(all(e["kind"] == "state_result" for e in summaries))
        for summary in summaries:
            observations = sum(e["state"] == summary["state"] and e["kind"] in {"hud","recognition"} for e in events)
            self.assertEqual(summary["timing"].get("vision",{}).get("calls",0),observations)
            self.assertGreaterEqual(summary["wall_elapsed"],0)
        for row in events:
            if row["kind"] == "recognition":
                self.assertIn("score",row)
                self.assertIn("threshold",row)

    def test_live_log_absolute_path_is_announced_before_first_state(self):
        import tailstopia_mission
        class Window:
            f8 = iter([False,True,False])
            f9 = iter([False,True])
            def __init__(self,*args):
                pass
            def pressed(self,vk):
                return next(self.f8 if vk == 0x77 else self.f9)
            def activate(self):
                pass
        class FakeLive(FixtureSession):
            def __init__(self,config,directory,window):
                super().__init__(config,directory,ROOT/"resource")
        self.config["output"] = str(self.directory)
        output = io.StringIO()
        with patch("tailstopia_mission.load_mission_config",return_value=(self.config,self.base)), \
             patch("tailstopia_workflow.game_window.GameWindow",Window), \
             patch("tailstopia_workflow.mission_runtime.LiveSession",FakeLive), \
             patch("sys.argv",["tailstopia_mission.py","--through","OPEN_ROOM_LIST"]), \
             contextlib.redirect_stdout(output):
            self.assertEqual(tailstopia_mission.main(),0)
        logs = list(self.directory.glob("*/execution.jsonl"))
        self.assertEqual(len(logs),1)
        first = json.loads(logs[0].read_text().splitlines()[0])
        self.assertEqual(first["kind"],"run_start")
        self.assertEqual(first["log_path"],str(logs[0].resolve()))
        text = output.getvalue()
        self.assertLess(text.index(str(logs[0].resolve())),text.index("state_entry"))


    def test_keyboard_language_guard_blocks_start_before_any_input(self):
        class Thai(FixtureSession):
            def prepare_macro(self,name):
                raise RuntimeError("Switch to English; no macro input sent")
        result,events,_ = self.run_case(Thai)
        self.assertEqual((result.status,result.state),("FAIL","PLAY_START"))
        self.assertIn("Switch to English",result.reason)
        self.assertFalse(any(e["kind"] in {"action","click","recording_start"} for e in events))

    def test_keyboard_language_rechecked_before_mission_macro(self):
        class Switched(FixtureSession):
            def prepare_macro(self,name):
                if name == "mission":
                    raise RuntimeError("Switch to English; no macro input sent")
        result,events,_ = self.run_case(Switched)
        self.assertEqual((result.status,result.state),("FAIL","PLAY_MISSION"))
        self.assertFalse(any(e["state"] == "PLAY_MISSION" and e["kind"] in {"action","recording_start"} for e in events))

    def test_live_keyboard_guard_logs_thai_without_dispatch(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        api = SimpleNamespace(GetWindowThreadProcessId=Mock(return_value=321),
                              GetKeyboardLayout=Mock(return_value=0x041E041E))
        session = LiveSession.__new__(LiveSession)
        session.window = SimpleNamespace(api=api,hwnd=123)
        session.check = lambda:None
        session.paused_total,session.pause_started = 0,None
        Session.__init__(session,self.config,self.directory,SimulatedInput())
        with self.assertRaisesRegex(RuntimeError,"Switch to English"):
            session.prepare_macro("start")
        rows = [json.loads(row) for row in (self.directory/"execution.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual((rows[0]["kind"],rows[0]["status"]),("keyboard_layout","FAIL"))
        self.assertFalse(session.api.held_keys or session.api.held_buttons)

    def test_reward_spacing_uses_existing_config(self):
        self.config["reward"]["between_slots"] = 1.25
        result,events,_ = self.run_case(through="COLLECT_REWARDS")
        self.assertEqual(result.status,"SUCCESS")
        clicks = [e for e in events if e["kind"] == "click" and e["label"].startswith("reward_")]
        self.assertEqual(len(clicks),7)
        for before,after in zip(clicks,clicks[1:]):
            self.assertGreaterEqual(after["active_at"]-before["active_at"],1.25)
