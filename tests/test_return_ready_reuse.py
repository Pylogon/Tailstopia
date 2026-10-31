"""Reuse a recent verified return only at the immediately following loop."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.mission_config import load_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession, LiveSession, Session, SimulatedInput
from tailstopia_workflow.mission_vision import MissionVision

ROOT = Path(__file__).resolve().parents[1]


def observation(status):
    return SimpleNamespace(status=status,score=1.0 if status == "READY" else 0.0,
                           white_fraction=1.0 if status == "LOADING" else 0.0)


class FakeSession:
    def __init__(self):
        self.time,self.wall = 0.0,0.0
        self.loop,self.state = 1,"RETURN_CAMP"
        self.paused_total,self.pause_requested = 0.0,False
        self.capture_context = (123,43,95,1280,600)
        self.api = SimpleNamespace(held_keys=set(),held_buttons=set())
        self.observations,self.captures,self.events,self.clicks = [],[],[],[]
        self.on_capture = None
        self.error = None

    def now(self):
        return self.time

    def sleep(self,seconds):
        self.time += seconds
        self.wall += seconds

    def check(self):
        if self.error is not None:raise self.error

    def capture(self,box):
        self.captures.append((self.loop,self.state,self.time))
        if not self.observations:raise AssertionError("Unnecessary capture")
        value = observation(self.observations.pop(0))
        self.sleep(.05)
        if self.on_capture is not None:
            callback,self.on_capture = self.on_capture,None
            callback()
        return value

    def event(self,kind,**fields):
        self.events.append({"kind":kind,**fields})

    def click(self,point,label):
        self.clicks.append(label)
        self.sleep(.05)


class ReturnReuseTests(unittest.TestCase):
    def setUp(self):
        self.new_flow()
        clock = patch("tailstopia_workflow.mission_flow.time.perf_counter",side_effect=lambda:self.session.wall)
        clock.start()
        self.addCleanup(clock.stop)

    def new_flow(self):
        self.config,self.base = load_mission_config(ROOT/"examples/mission_flow.json")
        self.session = FakeSession()
        self.flow = MissionFlow.__new__(MissionFlow)
        self.flow.config,self.flow.session = self.config,self.session
        self.flow.timings = {}
        self.flow.reset_return_verification()
        self.flow.vision = SimpleNamespace(
            hud=lambda capture:capture(None),
            match=lambda name,capture:SimpleNamespace(found=True,point=(850,458),score=1.0),
            start=SimpleNamespace(config={"poll_seconds":3,"ready_confirmations":2}))

    def execute(self,state,values,loop=1):
        self.flow.state,self.session.state,self.session.loop = state,state,loop
        self.flow.deadline = self.session.now()+120
        self.session.observations = list(values)
        self.flow.timings = {}
        self.flow.execute_state()
        self.assertEqual(self.session.observations,[])

    def verified_return(self,loading=True,late_loading=False):
        self.execute("RETURN_CAMP",[])
        self.execute("WAIT_RETURN_LOADING",["LOADING"] if loading else ["READY","READY"])
        self.execute("VERIFY_RETURN_START",["LOADING","READY","READY"] if late_loading else ["READY","READY"])

    def next_start(self,values):
        before,at = len(self.session.captures),self.session.now()
        self.execute("VERIFY_START",values,2)
        return len(self.session.captures)-before,self.session.now()-at

    def test_first_loop_still_requires_two_ready_observations(self):
        self.execute("VERIFY_START",["READY","READY"])
        self.assertEqual(len(self.session.captures),2)
        self.assertAlmostEqual(self.session.now(),3.1)

    def test_verified_return_plus_one_fresh_ready_passes_without_wait(self):
        self.verified_return()
        captures,elapsed = self.next_start(["READY"])
        self.assertEqual(captures,1)
        self.assertAlmostEqual(elapsed,.05)
        self.assertEqual(self.session.clicks,["camp"])
        self.assertIsNone(self.flow._return_confirmation)
        self.assertEqual(sum(e["kind"] == "return_verification_reused" for e in self.session.events),1)

    def test_ready_without_observed_return_loading_cannot_be_reused(self):
        self.verified_return(loading=False)
        self.assertIsNone(self.flow._return_confirmation)
        self.assertEqual(self.next_start(["READY","READY"])[0],2)

    def test_loading_during_return_verification_is_acceptable_transition_evidence(self):
        self.verified_return(loading=False,late_loading=True)
        self.assertIsNotNone(self.flow._return_confirmation)
        self.assertEqual(self.next_start(["READY"])[0],1)

    def test_loading_without_verified_return_click_does_not_create_proof(self):
        self.execute("WAIT_RETURN_LOADING",["LOADING"])
        self.execute("VERIFY_RETURN_START",["READY","READY"])
        self.assertIsNone(self.flow._return_confirmation)
        self.assertEqual(self.next_start(["READY","READY"])[0],2)

    def test_expiry_uses_wall_time_even_when_active_clock_does_not_advance(self):
        self.verified_return()
        self.session.wall += 2
        captures,elapsed = self.next_start(["READY","READY"])
        self.assertEqual(captures,2)
        self.assertAlmostEqual(elapsed,3.1)

    def test_slow_fresh_capture_expires_proof_and_uses_normal_confirmation(self):
        self.verified_return()
        self.session.on_capture = lambda:self.session.sleep(1.2)
        self.assertEqual(self.next_start(["READY","READY"])[0],2)

    def test_intervening_state_discards_previous_return_confirmation(self):
        self.verified_return()
        self.execute("WAIT_RETURN_LOADING",["LOADING"])
        self.assertIsNone(self.flow._return_confirmation)
        self.assertEqual(self.next_start(["READY","READY"])[0],2)

    def test_pause_or_pending_pause_invalidates_previous_confirmation(self):
        for pending in (False,True):
            with self.subTest(pending=pending):
                self.new_flow();self.verified_return()
                self.session.pause_requested = pending
                if not pending:self.session.paused_total += .1
                self.assertEqual(self.next_start(["READY","READY"])[0],2)

    def test_pause_during_fresh_capture_falls_back_without_duplicate_capture(self):
        self.verified_return()
        self.session.on_capture = lambda:setattr(self.session,"paused_total",.1)
        self.assertEqual(self.next_start(["READY","READY"])[0],2)

    def test_window_geometry_change_during_capture_invalidates_proof(self):
        self.verified_return()
        self.session.on_capture = lambda:setattr(self.session,"capture_context",(123,50,95,1280,600))
        self.assertEqual(self.next_start(["READY","READY"])[0],2)

    def test_unknown_or_loading_fresh_frame_resets_count_without_immediate_recapture(self):
        for status in ("UNKNOWN","LOADING"):
            with self.subTest(status=status):
                self.new_flow();self.verified_return()
                before = len(self.session.captures)
                captures,elapsed = self.next_start([status,"READY","READY"])
                self.assertEqual(captures,3)
                self.assertAlmostEqual(elapsed,6.15)
                times=[r[2] for r in self.session.captures[before:]]
                self.assertAlmostEqual(times[1]-times[0],3.05)

    def test_weaker_return_policy_does_not_override_start_policy(self):
        for key,value in (("ready_confirmations",3),("poll_seconds",4)):
            with self.subTest(key=key):
                self.new_flow();self.verified_return()
                self.flow.vision.start.config[key]=value
                expected=3 if key == "ready_confirmations" else 2
                self.assertEqual(self.next_start(["READY"]*expected)[0],expected)

    def test_held_inputs_or_different_loop_disable_reuse(self):
        self.verified_return()
        self.session.api.held_keys.add("a")
        self.assertEqual(self.next_start(["READY","READY"])[0],2)
        self.new_flow();self.verified_return()
        self.execute("VERIFY_START",["READY","READY"],3)
        self.assertFalse(any(e["kind"] == "return_verification_reused" for e in self.session.events))

    def test_confirmation_is_single_use(self):
        self.verified_return()
        self.next_start(["READY"])
        self.assertEqual(self.next_start(["READY","READY"])[0],2)

    def test_f9_or_focus_loss_during_fresh_capture_does_not_pass_or_keep_proof(self):
        for error in (StopRequested("F9"),RuntimeError("Game lost focus")):
            with self.subTest(error=type(error).__name__):
                self.new_flow();self.verified_return()
                self.session.on_capture = lambda:setattr(self.session,"error",error)
                with self.assertRaises(type(error)):
                    self.next_start(["READY"])
                self.assertIsNone(self.flow._return_confirmation)
                self.assertFalse(any(e["kind"] == "return_verification_reused" for e in self.session.events))

    def test_pause_during_return_confirmation_prevents_proof_creation(self):
        self.execute("RETURN_CAMP",[])
        self.execute("WAIT_RETURN_LOADING",["LOADING"])
        self.session.on_capture = lambda:setattr(self.session,"paused_total",.1)
        self.execute("VERIFY_RETURN_START",["READY","READY"])
        self.assertIsNone(self.flow._return_confirmation)


class ReturnReuseIntegrationTests(unittest.TestCase):
    def test_two_loops_save_one_capture_and_three_seconds_without_changing_actions(self):
        config,base=load_mission_config(ROOT/"examples/mission_flow.json")
        config["max_loops"]=2
        vision=MissionVision(config,base)
        class Counting(FixtureSession):
            def __init__(self,*args):
                self.captures=[]
                super().__init__(*args)
            def capture(self,box):
                self.captures.append((self.loop,self.state))
                image=super().capture(box)
                self.virtual_time += .001  # Deterministic wall-time measurement in this test.
                return image
        results=[]
        for max_age in (0,1):
            with tempfile.TemporaryDirectory() as directory:
                session=Counting(config,directory,ROOT/"resource")
                with patch.object(MissionFlow,"RETURN_READY_MAX_AGE",max_age), \
                     patch("tailstopia_workflow.mission_flow.time.perf_counter",side_effect=session.now), \
                     session.log,contextlib.redirect_stdout(io.StringIO()):
                    flow=MissionFlow(config,base,session,vision)
                    result=flow.run()
                self.assertEqual((result.status,result.loops_completed),("SUCCESS",2))
                self.assertIsNone(flow._return_confirmation)
                rows=[json.loads(line) for line in (Path(directory)/"execution.jsonl").read_text().splitlines()]
                results.append((session.now(),session.captures,rows))
                if session.last_image is not None:session.last_image.close()
        before,after=results
        self.assertAlmostEqual(before[0]-after[0],3.001)
        self.assertEqual(len(before[1])-len(after[1]),1)
        self.assertEqual(after[1].count((1,"VERIFY_START")),2)
        self.assertEqual(after[1].count((2,"VERIFY_START")),1)
        for state in ("WAIT_RETURN_LOADING","VERIFY_RETURN_START"):
            self.assertEqual([r for r in before[1] if r[1] == state],[r for r in after[1] if r[1] == state])
        actions=lambda rows:[{k:v for k,v in row.items() if k not in ("time","active_at")} for row in rows if row["kind"] == "action"]
        self.assertEqual(actions(before[2]),actions(after[2]))
        self.assertEqual(sum(r["kind"] == "return_verification_reused" for r in after[2]),1)

    def test_capture_context_uses_geometry_query_that_already_exists(self):
        config,_=load_mission_config(ROOT/"examples/mission_flow.json")
        with tempfile.TemporaryDirectory() as directory:
            session=LiveSession.__new__(LiveSession)
            Session.__init__(session,config,directory,SimulatedInput())
            geometry=Mock(return_value=(43,95,1280,600))
            session.window=SimpleNamespace(hwnd=123,geometry=geometry)
            session.check=lambda:None
            with patch("tailstopia_workflow.mission_runtime.ImageGrab.grab",return_value=Image.new("RGB",(43,25))):
                with session.capture((346,0,389,25)):
                    self.assertEqual(session.capture_context,(123,43,95,1280,600))
            geometry.assert_called_once()
            session.last_image.close()


if __name__ == "__main__":
    unittest.main()
