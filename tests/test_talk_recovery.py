"""Bounded Talk recovery: stationary confirmation and release on every exit."""
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
from tailstopia_workflow.mission_runtime import FixtureSession, SimulatedInput
from tailstopia_workflow.mission_vision import MissionVision

ROOT = Path(__file__).resolve().parents[1]


class Backend(SimulatedInput):
    def __init__(self):
        self.calls = []
        self.fail_once = None

    def dispatch(self,method,value):
        self.calls.append((method,value))
        if self.fail_once == (method,value):
            self.fail_once = None
            raise RuntimeError("Input backend failed")

    def keyDown(self,key,**kwargs):self.dispatch("keyDown",key)
    def keyUp(self,key,**kwargs):self.dispatch("keyUp",key)
    def mouseDown(self,**kwargs):self.dispatch("mouseDown",kwargs["button"])
    def mouseUp(self,**kwargs):self.dispatch("mouseUp",kwargs["button"])


class RecoverySession(FixtureSession):
    def __init__(self,config,directory):
        super().__init__(config,directory,ROOT/"resource")
        self.api.backend = Backend()
        self.captures,self.holds,self.prepared = [],[],[]
        self.stop_at,self.stop_error = None,RuntimeError("Game lost focus")
        self.layout_error = None

    def check(self):
        if self.stop_at is not None and self.now() >= self.stop_at-1e-9:
            raise self.stop_error

    def sleep(self,seconds):
        self.holds.append((seconds,set(self.api.held_keys),set(self.api.held_buttons)))
        super().sleep(seconds)

    def capture(self,target):
        if self.api.held_keys or self.api.held_buttons:
            raise AssertionError("Recognition while recovery inputs held")
        self.captures.append((self.now(),target))

    def prepare_macro(self,name):
        self.prepared.append(name)
        if self.layout_error:raise self.layout_error


class TalkRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.config,self.base = load_mission_config(ROOT/"examples/mission_flow.json")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = RecoverySession(self.config,self.temp.name)
        self.talk_results = []
        self.room_found = lambda:sum(self.session.click_counts.values()) >= 2
        def match(name,capture):
            capture(name)
            found = self.room_found() if name == "room_list" else (self.talk_results.pop(0) if self.talk_results else False)
            return SimpleNamespace(found=found,score=.95 if found else .31,point=(478,321))
        self.flow = MissionFlow(self.config,self.base,self.session,SimpleNamespace(match=match))
        self.flow.state,self.flow.deadline = "OPEN_MISSION",35
        self.session.enter("OPEN_MISSION",1)

    def open(self):
        with self.session.log,contextlib.redirect_stdout(io.StringIO()):self.flow.open_mission()

    def rows(self):
        return [json.loads(line) for line in (Path(self.temp.name)/"execution.jsonl").read_text().splitlines()]

    def movements(self):
        return sum(call == ("mouseDown","right") for call in self.session.api.backend.calls)

    def assert_released(self):
        self.assertFalse(self.session.api.held_keys or self.session.api.held_buttons)
        self.assertTrue(self.session.api.FAILSAFE)

    def test_normal_fallback_adds_no_capture_wait_or_movement(self):
        self.talk_results = [True]
        self.open()
        self.assertEqual(self.movements(),0)
        self.assertEqual(len(self.session.captures),14)  # guard, 11 polls, Talk, fallback result.
        self.assertAlmostEqual(self.session.now(),15.6)
        self.assertEqual(self.session.prepared,[])

    def test_missing_then_present_on_confirmation_never_moves(self):
        self.talk_results = [False,True]
        self.open()
        self.assertEqual(self.movements(),0)
        self.assertEqual(self.session.click_counts['talk'],2)
        self.assertAlmostEqual(self.session.now(),16.1)

    def test_delayed_room_list_during_confirmation_prevents_walk_and_click(self):
        self.room_found = lambda:self.session.now() >= 10.5
        self.open()
        self.assertEqual(self.movements(),0)
        self.assertEqual(self.session.click_counts['talk'],1)
        self.assertEqual(sum(name=='talk' for _,name in self.session.captures),1)

    def test_one_burst_holds_right_s_a_together_then_releases_before_recognition(self):
        self.talk_results = [False,False,True]
        original = json.dumps(self.flow.macros,sort_keys=True)
        self.open()
        self.assertEqual(self.movements(),1)
        held = [(dt,keys,buttons) for dt,keys,buttons in self.session.holds if 'right' in buttons]
        self.assertTrue(held)
        self.assertTrue(all(keys == {'s','a'} and buttons == {'right'} for dt,keys,buttons in held))
        self.assertAlmostEqual(sum(dt for dt,_,_ in held),.25)
        self.assertEqual(self.session.prepared,['talk_recovery'])
        self.assertEqual(original,json.dumps(self.flow.macros,sort_keys=True))
        rows=self.rows()
        down=next(r for r in rows if r['kind']=='action' and r['method']=='mouseDown' and r['params']['button']=='right')
        anchor=next(a for a in self.flow.macros['start'][0] if a['type']=='mouse_down' and a['button']=='right')
        self.assertEqual((down['params']['x'],down['params']['y']),(anchor['x'],anchor['y']))
        self.assertEqual(self.session.click_counts['talk'],2)
        self.assert_released()

    def test_repeated_absence_uses_only_two_bursts_and_no_unverified_click(self):
        with self.assertRaisesRegex(RuntimeError,'budget exhausted; attempts=2'):self.open()
        self.assertEqual(self.movements(),2)
        self.assertEqual(self.session.click_counts['talk'],1)
        # First miss + confirmation + post-burst miss + confirmation per attempt.
        self.assertEqual(sum(name=='talk' for _,name in self.session.captures),6)
        self.assertAlmostEqual(sum(dt for dt,_,buttons in self.session.holds if 'right' in buttons),.5)
        self.assertEqual(sum(r['kind']=='talk_recovery_move' for r in self.rows()),2)
        self.assert_released()

    def test_room_list_after_movement_prevents_second_move_and_fallback_click(self):
        self.room_found = lambda:self.movements() >= 1
        self.open()
        self.assertEqual(self.movements(),1)
        self.assertEqual(self.session.click_counts['talk'],1)
        self.assert_released()

    def test_second_burst_can_find_talk_after_two_fresh_misses(self):
        self.talk_results = [False,False,False,False,True]
        self.open()
        self.assertEqual(self.movements(),2)
        self.assertEqual(self.session.click_counts['talk'],2)
        self.assert_released()

    def test_max_attempts_zero_disables_movement_but_still_confirms_absence(self):
        self.config['talk']['recovery']['max_attempts']=0
        with self.assertRaisesRegex(RuntimeError,'attempts=0'):self.open()
        self.assertEqual(self.movements(),0)
        self.assertEqual(sum(name=='talk' for _,name in self.session.captures),2)

    def test_stop_focus_loss_and_timeout_during_hold_release_all(self):
        for error in (StopRequested('F9'),RuntimeError('Game lost focus'),TimeoutError('deadline')):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                self.session.stop_at,self.session.stop_error=.1,error
                with self.assertRaises(type(error)):self.flow.move_toward_talk(.25)
                self.assertEqual(self.movements(),1)
                self.assertFalse(self.session.captures)
                self.assert_released()

    def test_real_state_deadline_interrupts_hold_and_releases_all(self):
        self.flow.deadline=.1
        with self.assertRaises(TimeoutError):self.flow.move_toward_talk(.25)
        self.assert_released()

    def test_partial_key_or_mouse_dispatch_failure_releases_every_tracked_input(self):
        for failing in (('mouseDown','right'),('keyDown','s'),('keyDown','a'),('keyUp','a'),('mouseUp','right')):
            with self.subTest(failing=failing):
                self.setUp()
                self.session.api.backend.fail_once=failing
                with self.assertRaisesRegex(RuntimeError,'backend failed'):self.flow.move_toward_talk(.25)
                # Up failures are retried by the existing run() Safe Stop cleanup.
                self.session.api.release_all()
                self.assert_released()
                if failing == ('keyUp','a'):
                    self.assertIn(('keyUp','s'),self.session.api.backend.calls)
                    self.assertIn(('mouseUp','right'),self.session.api.backend.calls)

    def test_non_english_layout_prevents_any_recovery_input(self):
        self.session.layout_error=RuntimeError('Switch keyboard to English')
        with self.assertRaisesRegex(RuntimeError,'English'):self.flow.move_toward_talk(.25)
        self.assertEqual(self.session.api.backend.calls,[])
        self.assert_released()

    def test_missing_recorded_mouse_anchor_never_sends_input(self):
        self.flow.macros['start']=([],0)
        with self.assertRaisesRegex(RuntimeError,'right-mouse anchor'):self.flow.move_toward_talk(.25)
        self.assertEqual(self.session.api.backend.calls,[])

    def test_existing_held_inputs_block_recovery(self):
        self.session.api.held_keys.add('w')
        with self.assertRaisesRegex(RuntimeError,'previous inputs released'):self.flow.move_toward_talk(.25)
        self.assertEqual(self.session.api.backend.calls,[])
        self.session.api.release_all()

    def test_configurable_short_burst_and_retry_budget(self):
        self.config['talk']['recovery'].update(max_attempts=1,hold_seconds=.2,confirm_wait=1,settle_seconds=.7)
        with self.assertRaisesRegex(RuntimeError,'attempts=1'):self.open()
        self.assertEqual(self.movements(),1)
        self.assertAlmostEqual(sum(dt for dt,_,buttons in self.session.holds if 'right' in buttons),.2)
        self.assert_released()

    def test_invalid_recovery_settings_rejected_before_input(self):
        for key,value in (('hold_seconds',0),('hold_seconds',.31),('hold_seconds',float('nan')),
                          ('hold_seconds',True),('max_attempts',-1),('max_attempts',1.5),('max_attempts',True),
                          ('confirm_wait',0),('settle_seconds',-1)):
            with self.subTest(key=key,value=value):
                config,_=load_mission_config(ROOT/'examples/mission_flow.json')
                config['talk']['recovery'][key]=value
                with self.assertRaises(ValueError):validate_mission_config(config)

    def test_old_config_without_recovery_remains_supported(self):
        del self.config['talk']['recovery']
        validate_mission_config(self.config)
        self.talk_results=[False,False,True]
        self.open()
        self.assertEqual(self.movements(),1)


class TalkRecoveryIntegrationTests(unittest.TestCase):
    def test_actual_talk_template_after_short_movement_opens_room_list(self):
        config,base=load_mission_config(ROOT/'examples/mission_flow.json')
        config['max_loops']=1
        class AppearingTalk(FixtureSession):
            recovered=False
            def enter(self,state,loop):
                super().enter(state,loop)
                if state=='OPEN_MISSION':self.frame='intiate.png'
            def sleep(self,seconds):
                if self.state=='OPEN_MISSION' and self.api.held_keys=={'s','a'} and self.api.held_buttons=={'right'}:
                    self.recovered=True
                super().sleep(seconds)
            def capture(self,box):
                if self.state=='OPEN_MISSION' and self.recovered and self.frame=='intiate.png':
                    if self.api.held_keys or self.api.held_buttons:
                        raise AssertionError('Captured during recovery hold')
                    self.frame='red panda2.png'
                return super().capture(box)
            def click_effect(self,label):
                if label!='talk' or self.recovered:super().click_effect(label)
        with tempfile.TemporaryDirectory() as directory:
            session=AppearingTalk(config,directory,ROOT/'resource')
            try:
                flow=MissionFlow(config,base,session,MissionVision(config,base))
                with session.log,contextlib.redirect_stdout(io.StringIO()):result=flow.run('OPEN_MISSION')
                rows=[json.loads(s) for s in (Path(directory)/'execution.jsonl').read_text().splitlines()]
                self.assertEqual((result.status,result.state),('SUCCESS','OPEN_MISSION'))
                self.assertEqual(sum(r['kind']=='talk_recovery_move' for r in rows),1)
                self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),2)
                self.assertEqual(sum(r['kind']=='recording_start' for r in rows),1)
                self.assertTrue(any(r['kind']=='recognition' and r['target']=='talk' and r['found'] for r in rows))
                self.assertFalse(session.api.held_keys or session.api.held_buttons)
            finally:
                if session.last_image is not None:session.last_image.close()

    def test_exhaustion_safe_stops_with_actual_frame_log_and_no_mission_macro(self):
        config,base=load_mission_config(ROOT/'examples/mission_flow.json')
        config['max_loops']=1
        class NoTalk(FixtureSession):
            def enter(self,state,loop):
                super().enter(state,loop)
                if state=='OPEN_MISSION':self.frame='intiate.png'
            def click_effect(self,label):
                if label!='talk':super().click_effect(label)
        with tempfile.TemporaryDirectory() as directory:
            session=NoTalk(config,directory,ROOT/'resource')
            try:
                flow=MissionFlow(config,base,session,MissionVision(config,base))
                with session.log,contextlib.redirect_stdout(io.StringIO()):result=flow.run()
                rows=[json.loads(s) for s in (Path(directory)/'execution.jsonl').read_text().splitlines()]
                self.assertEqual((result.status,result.state),('FAIL','OPEN_MISSION'))
                self.assertIn('attempts=2',result.reason)
                self.assertEqual(sum(r['kind']=='talk_recovery_move' for r in rows),2)
                self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),1)
                self.assertFalse(any(r['kind']=='recording_start' and r['recording']=='mission' for r in rows))
                self.assertTrue(list(Path(directory).glob('*_failure.png')))
                self.assertTrue(any(r['kind']=='state_result' and r['status']=='FAIL' and 'recovery_move' in r['timing'] for r in rows))
                self.assertFalse(session.api.held_keys or session.api.held_buttons)
            finally:
                if session.last_image is not None:session.last_image.close()


if __name__=='__main__':unittest.main()
