"""Opt-in Camp return trial: saved screenshots and virtual clocks, no OS input."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from PIL import Image

import test_return_ready_reuse as reuse
from offline.camp_checkpoint import HouseCheckpoint
from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.camp_detection import CampDetector
from tailstopia_workflow.mission_config import load_mission_config,validate_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT = Path(__file__).resolve().parents[1]


class CampVisionTests(unittest.TestCase):
    def test_promoted_detector_matches_frozen_offline_corpus_with_one_joint_capture(self):
        config,base = load_mission_config(ROOT/'examples/mission_95.json')
        config['return']['camp_checkpoint']['enabled'] = True
        vision = MissionVision(config,base)
        corpus = ROOT/'tests/fixtures/camp_checkpoint'
        with Image.open(corpus/'house_reference.png') as image:
            frozen = HouseCheckpoint(image)
        cases = json.loads((corpus/'manifest.json').read_text())['samples']
        for case in cases:
            with self.subTest(case=case['file']),Image.open(corpus/case['file']) as packed:
                with packed.crop((0,0,50,110)) as house:
                    expected = frozen.inspect(house)
                box = vision.camp.capture_box
                joint = Image.new('RGB',(box[2]-box[0],box[3]-box[1]))
                with packed.crop((0,0,50,110)) as house,packed.crop((50,0,93,25)) as time_roi:
                    joint.paste(house,(4-box[0],115-box[1]))
                    joint.paste(time_roi,(346-box[0],-box[1]))
                capture = Mock(return_value=joint)
                observed = vision.camp_hud(capture)
                capture.assert_called_once_with((4,0,389,225))
                self.assertEqual(observed.found,expected.found)
                self.assertAlmostEqual(observed.score,expected.score,places=6)
                self.assertEqual(observed.center,expected.center)

    def test_opt_out_does_not_load_template(self):
        config,base = load_mission_config(ROOT/'examples/mission_95.json')
        config['return']['camp_checkpoint']['enabled'] = False
        with patch('tailstopia_workflow.mission_vision.CampDetector',side_effect=AssertionError('Loaded while disabled')):
            self.assertIsNone(MissionVision(config,base).camp)

    def test_invalid_trial_settings_fail_before_game_interaction(self):
        config,_ = load_mission_config(ROOT/'examples/mission_95.json')
        for key,value in [('enabled',1),('threshold',.89),('poll',.3),('confirmation_interval',.3),
                          ('timeout',0),('threshold',float('nan')),('search_roi_client',[0,0,20,20])]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                cfg = copy.deepcopy(config);cfg['return']['camp_checkpoint'][key] = value
                validate_mission_config(cfg)


class CampReturnTests(unittest.TestCase):
    def setUp(self):
        self.config,self.base = load_mission_config(ROOT/'examples/mission_95.json')
        self.config['return']['camp_checkpoint']['enabled'] = True
        self.session = reuse.FakeSession()
        self.flow = MissionFlow.__new__(MissionFlow)
        self.flow.config,self.flow.session = self.config,self.session
        self.flow.timings = {}
        self.flow.reset_return_verification()
        def camp(capture):
            obs = capture(('joint',))
            return SimpleNamespace(found=obs.status == 'READY',score=.99,center=(27.5,146),hud=obs)
        self.flow.vision = SimpleNamespace(
            camp=SimpleNamespace(VERSION=CampDetector.VERSION),camp_hud=camp,
            hud=lambda capture:capture(('time',)),
            match=lambda name,capture:SimpleNamespace(found=True,point=(850,458),score=1),
            start=SimpleNamespace(config={'poll_seconds':3,'ready_confirmations':2}))
        self.flow.play = Mock(side_effect=AssertionError('Premature macro'))
        self.clock = patch('tailstopia_workflow.mission_flow.time.perf_counter',side_effect=lambda:self.session.wall)
        self.clock.start();self.addCleanup(self.clock.stop)

    def execute(self,state,values,loop=1,timeout=120):
        self.flow.state,self.session.state,self.session.loop = state,state,loop
        self.flow.deadline = self.session.now()+timeout
        self.session.observations = list(values)
        self.flow.execute_state()
        self.assertEqual(self.session.observations,[])

    def loading(self):
        self.execute('RETURN_CAMP',[])
        self.execute('WAIT_RETURN_LOADING',['LOADING'])

    def verified(self):
        self.loading()
        self.execute('VERIFY_RETURN_START',['READY','READY'])

    def test_fast_return_keeps_real_interval_and_reuses_one_fresh_joint_frame(self):
        self.loading();before = self.session.now()
        self.execute('VERIFY_RETURN_START',['READY','READY'])
        self.assertAlmostEqual(self.session.now()-before,.6)
        proof = self.flow._return_confirmation
        self.assertEqual((proof['kind'],proof['poll'],proof['confirmations']),('camp_checkpoint',.5,2))
        self.execute('VERIFY_START',['READY'],loop=2)
        self.assertIsNone(self.flow._return_confirmation)
        self.assertTrue(any(e['kind']=='return_verification_reused' and e['proof_kind']=='camp_checkpoint' for e in self.session.events))
        self.assertEqual(self.session.clicks,['camp']);self.flow.play.assert_not_called()

    def test_unstable_joint_observation_resets_streak(self):
        self.loading();before = self.session.now()
        self.execute('VERIFY_RETURN_START',['READY','UNKNOWN','READY','READY'])
        self.assertAlmostEqual(self.session.now()-before,1.7)
        self.assertEqual(self.flow._return_confirmation['confirmations'],2)

    def test_house_without_ready_time_does_not_confirm(self):
        self.loading()
        original = self.flow.vision.camp_hud
        def house_only(capture):
            sample = original(capture);sample.found = True
            return sample
        self.flow.vision.camp_hud = house_only
        self.execute('VERIFY_RETURN_START',['UNKNOWN','READY','READY'])
        self.assertEqual(self.flow._return_confirmation['confirmations'],2)
        self.assertEqual(sum(e['kind']=='camp_checkpoint' for e in self.session.events),3)

    def test_ready_without_house_is_not_fast_success_and_fallback_uses_last_hud(self):
        self.loading()
        self.config['return']['camp_checkpoint']['timeout'] = 1
        original = self.flow.vision.camp_hud
        def missing(capture):
            sample = original(capture);sample.found = False
            return sample
        self.flow.vision.camp_hud = missing
        self.execute('VERIFY_RETURN_START',['READY','READY','READY'])
        # Two trial captures, then last READY seeds legacy's 3s confirmation.
        self.assertEqual(self.flow._return_confirmation.get('kind'),None)
        self.assertAlmostEqual(self.flow._return_confirmation['poll'],3)
        self.assertEqual(sum(e['kind']=='camp_return_fallback' for e in self.session.events),1)

    def test_no_loading_evidence_uses_original_flow_without_joint_capture(self):
        self.execute('RETURN_CAMP',[])
        self.execute('WAIT_RETURN_LOADING',['UNKNOWN','READY','READY'])
        self.flow.vision.camp_hud = Mock(side_effect=AssertionError('Trial without LOADING'))
        self.execute('VERIFY_RETURN_START',['READY','READY'])
        self.assertIsNone(self.flow._return_confirmation)
        self.execute('VERIFY_START',['READY','READY'],loop=2)

    def test_missing_return_click_cannot_create_checkpoint_proof(self):
        self.execute('WAIT_RETURN_LOADING',['LOADING'])
        self.execute('VERIFY_RETURN_START',['READY','READY'])
        self.assertIsNone(self.flow._return_confirmation)

    def test_first_loop_still_uses_original_time_confirmation(self):
        self.flow.vision.camp_hud = Mock(side_effect=AssertionError('First loop changed'))
        self.execute('VERIFY_START',['READY','READY'])
        self.assertAlmostEqual(self.session.now(),3.1)

    def test_failed_fresh_house_uses_same_time_frame_then_legacy_wait(self):
        self.verified()
        original = self.flow.vision.camp_hud
        def missing(capture):
            sample = original(capture);sample.found = False
            return sample
        self.flow.vision.camp_hud = missing
        before = self.session.now()
        self.execute('VERIFY_START',['READY','READY'],loop=2)
        self.assertAlmostEqual(self.session.now()-before,3.1)
        self.assertTrue(any(e['kind']=='return_verification_fallback' for e in self.session.events))

    def test_expired_changed_context_pause_policy_or_held_input_invalidates_fast_proof(self):
        for change in ('expired','context','pause','policy','disabled','held','loop','start_policy'):
            with self.subTest(change=change):
                self.flow.reset_return_verification();self.session.api.held_keys.clear()
                self.session.paused_total=0;self.session.capture_context=(123,43,95,1280,600)
                self.config['return']['camp_checkpoint']['enabled']=True
                self.config['return']['camp_checkpoint']['threshold']=.90
                self.flow.vision.start.config['poll_seconds']=3
                self.verified()
                if change=='expired':self.session.wall += 2
                if change=='context':self.session.capture_context=(123,44,95,1280,600)
                if change=='pause':self.session.paused_total += 2
                if change=='policy':self.config['return']['camp_checkpoint']['threshold']=.91
                if change=='disabled':self.config['return']['camp_checkpoint']['enabled']=False
                if change=='held':self.session.api.held_keys.add('a')
                if change=='start_policy':self.flow.vision.start.config['poll_seconds']=4
                self.execute('VERIFY_START',['READY','READY'],loop=3 if change=='loop' else 2)
                self.assertIsNone(self.flow._return_confirmation)

    def test_pause_or_geometry_change_during_trial_discards_evidence(self):
        for change in ('pause','geometry'):
            with self.subTest(change=change):
                self.session.capture_context=(123,43,95,1280,600);self.session.paused_total=0
                self.flow.reset_return_verification();self.loading()
                def interrupted():
                    if change=='pause':self.session.paused_total += 5;self.session.wall += 5
                    else:self.session.capture_context=(123,44,95,1280,600)
                self.session.on_capture=interrupted
                self.execute('VERIFY_RETURN_START',['READY','READY','READY'])
                self.assertIsNone(self.flow._return_confirmation)

    def test_timeout_f9_and_focus_stop_without_next_capture_or_macro(self):
        for error in (None,StopRequested('F9 pressed'),RuntimeError('Game lost focus')):
            with self.subTest(error=error):
                self.session.error=None;self.flow.reset_return_verification();self.loading()
                if error:self.session.on_capture=lambda:setattr(self.session,'error',error)
                with self.assertRaises(type(error) if error else TimeoutError):
                    self.execute('VERIFY_RETURN_START',['READY'],timeout=.2)
                self.assertIsNone(self.flow._return_confirmation)
                self.flow.play.assert_not_called()

    def test_slow_fresh_capture_expires_fast_proof(self):
        self.verified()
        self.session.on_capture=lambda:self.session.sleep(1.2)
        self.execute('VERIFY_START',['READY','READY'],loop=2)
        self.assertTrue(any(e['kind']=='return_verification_fallback' for e in self.session.events))

    def test_f9_and_focus_loss_during_fresh_reuse_consume_proof_and_stop(self):
        for error in (StopRequested('F9 pressed'),RuntimeError('Game lost focus')):
            with self.subTest(error=error):
                self.session.error=None;self.flow.reset_return_verification();self.verified()
                self.session.on_capture=lambda:setattr(self.session,'error',error)
                with self.assertRaises(type(error)):
                    self.execute('VERIFY_START',['READY'],loop=2)
                self.assertIsNone(self.flow._return_confirmation)
                self.flow.play.assert_not_called()

    def test_fallback_does_not_extend_state_deadline(self):
        self.loading();self.config['return']['camp_checkpoint']['timeout']=.5
        original = self.flow.vision.camp_hud
        def missing(capture):
            sample=original(capture);sample.found=False
            return sample
        self.flow.vision.camp_hud=missing
        start=self.session.now()
        with self.assertRaises(TimeoutError):
            self.execute('VERIFY_RETURN_START',['READY'],timeout=1)
        self.assertAlmostEqual(self.session.now()-start,1)
        self.assertIsNone(self.flow._return_confirmation)


class CampReturnIntegrationTests(unittest.TestCase):
    def run_case(self,enabled,broken=False):
        config,base = load_mission_config(ROOT/'examples/mission_95.json')
        config['max_loops']=2;config['return']['camp_checkpoint']['enabled']=enabled
        class Frames(FixtureSession):
            def __init__(self,*args):
                super().__init__(*args);self.capture_context=(123,43,95,1280,600)
                self.return_at=None;self.captures=[]
            def enter(self,state,loop):
                super().enter(state,loop)
                if state=='VERIFY_RETURN_START':self.return_at=self.now()
            def capture(self,box):
                self.captures.append((self.loop,self.state,self.now(),box))
                if self.state=='VERIFY_RETURN_START':
                    self.frame='loading.png' if self.now()-self.return_at < 1.5 else '95/talk.png'
                    if broken and tuple(box)==(4,0,389,225):raise RuntimeError('Game lost focus')
                image=super().capture(box);self.sleep(.05)
                return image
        with tempfile.TemporaryDirectory() as directory:
            session=Frames(config,directory,ROOT/'resource')
            try:
                with patch('tailstopia_workflow.mission_flow.time.perf_counter',side_effect=session.now),session.log,contextlib.redirect_stdout(io.StringIO()):
                    result=MissionFlow(config,base,session,MissionVision(config,base)).run()
                rows=[json.loads(l) for l in (Path(directory)/'execution.jsonl').read_text().splitlines()]
                self.assertFalse(session.api.held_keys or session.api.held_buttons)
                return result,rows,session.captures,bool(list(Path(directory).glob('*.png')))
            finally:
                if session.last_image is not None:session.last_image.close()

    def test_two_loops_compare_waits_and_captures_with_identical_macro_inputs(self):
        old,old_rows,old_frames,_=self.run_case(False)
        new,new_rows,new_frames,_=self.run_case(True)
        self.assertEqual((old.status,new.status),('SUCCESS','SUCCESS'))
        actions=lambda rows:[{k:v for k,v in r.items() if k not in ('time','active_at')} for r in rows if r['kind']=='action']
        self.assertEqual(actions(old_rows),actions(new_rows))
        macros=lambda rows:[(r['recording'],r['action'],r['dispatched_seconds']) for r in rows if r['kind']=='macro_action']
        self.assertEqual(macros(old_rows),macros(new_rows))
        states=lambda rows:[r for r in rows if r['kind']=='state_result' and r['state']=='VERIFY_RETURN_START']
        for before,after in zip(states(old_rows),states(new_rows)):
            self.assertGreater(before['elapsed']-after['elapsed'],2.5)
        self.assertEqual(sum(r['kind']=='camp_return_verified' for r in new_rows),2)
        self.assertEqual(sum(r['kind']=='return_verification_reused' and r.get('proof_kind')=='camp_checkpoint' for r in new_rows),1)
        verified=next(r['active_at'] for r in new_rows if r['kind']=='camp_return_verified' and r['loop']==1)
        next_macro=next(r['active_at'] for r in new_rows if r['kind']=='recording_start' and r['loop']==2)
        self.assertGreaterEqual(next_macro,verified)
        self.assertEqual(sum(state=='VERIFY_START' and loop==2 for loop,state,_,_ in new_frames),1)

    def test_focus_loss_safe_stops_saves_debug_and_never_starts_next_loop(self):
        result,rows,_,debug=self.run_case(True,True)
        self.assertEqual((result.status,result.state),('FAIL','VERIFY_RETURN_START'))
        self.assertTrue(debug)
        self.assertFalse(any(r['kind']=='recording_start' and r['loop']==2 for r in rows))


if __name__ == '__main__':
    unittest.main()
