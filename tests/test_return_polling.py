"""Adaptive return polling: virtual time, recorded HUD traces, no OS input."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import test_return_ready_reuse as reuse
from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.mission_config import load_mission_config,validate_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT=Path(__file__).resolve().parents[1]


class AdaptiveReturnTests(unittest.TestCase):
    def setUp(self):
        self.config,self.base=load_mission_config(ROOT/'examples/mission_95.json')
        self.session=reuse.FakeSession()
        self.flow=MissionFlow.__new__(MissionFlow)
        self.flow.config,self.flow.session=self.config,self.session
        self.flow.timings={}
        self.flow.reset_return_verification()
        self.flow.vision=SimpleNamespace(hud=lambda capture:capture(None),
            match=lambda name,capture:SimpleNamespace(found=True,point=(850,458),score=1),
            start=SimpleNamespace(config={'poll_seconds':3,'ready_confirmations':2}))
        self.flow.play=Mock(side_effect=AssertionError('Unexpected macro'))
        self.waits=[]
        original_wait=self.flow.wait
        def wait(seconds):self.waits.append(seconds);return original_wait(seconds)
        self.flow.wait=wait
        clock=patch('tailstopia_workflow.mission_flow.time.perf_counter',side_effect=lambda:self.session.wall)
        clock.start();self.addCleanup(clock.stop)

    def execute(self,state,statuses,loop=1,timeout=120):
        self.flow.state,self.session.state,self.session.loop=state,state,loop
        self.flow.deadline=self.session.now()+timeout
        self.flow.timings={}
        self.session.observations=list(statuses)
        self.flow.execute_state()
        self.assertEqual(self.session.observations,[])

    def test_loading_unknown_and_failed_confirmation_choose_correct_interval(self):
        cases=[(['LOADING','READY','READY'],[1,3],4.15),
               (['UNKNOWN','READY','READY'],[1,3],4.15),
               (['READY','UNKNOWN','READY','READY'],[3,1,3],7.20),
               (['LOADING','READY','UNKNOWN','READY','READY'],[1,3,1,3],8.25)]
        for statuses,waits,elapsed in cases:
            self.session.time=self.session.wall=0
            self.session.captures.clear();self.waits.clear()
            with self.subTest(statuses=statuses):
                self.execute('VERIFY_RETURN_START',statuses)
                self.assertEqual(self.waits,waits)
                self.assertAlmostEqual(self.session.now(),elapsed)
                self.assertEqual(len(self.session.captures),len(statuses))
                self.assertFalse(self.session.clicks)
                self.flow.play.assert_not_called()

    def test_legacy_mode_is_unchanged_and_wait_loading_stays_half_second(self):
        self.config['return'].pop('not_ready_poll')
        self.execute('VERIFY_RETURN_START',['UNKNOWN','READY','READY'])
        self.assertEqual(self.waits,[3,3]);self.assertAlmostEqual(self.session.now(),6.15)
        self.waits.clear()
        self.config['return']['not_ready_poll']=1
        self.execute('WAIT_RETURN_LOADING',['UNKNOWN','LOADING'])
        self.assertEqual(self.waits,[.5])

    def test_valid_return_proof_keeps_three_second_policy_and_next_start_reuses(self):
        self.execute('RETURN_CAMP',[])
        self.execute('WAIT_RETURN_LOADING',['UNKNOWN','LOADING'])
        self.execute('VERIFY_RETURN_START',['LOADING','READY','READY'])
        proof=self.flow._return_confirmation
        self.assertEqual(proof['poll'],3)
        self.assertEqual(proof['confirmations'],2)
        self.assertEqual(proof['loop'],1)
        self.waits.clear();before=len(self.session.captures)
        self.execute('VERIFY_START',['READY'],loop=2)
        self.assertEqual(len(self.session.captures)-before,1)
        self.assertEqual(self.waits,[])
        self.assertEqual(self.session.clicks,['camp'])
        self.assertEqual(sum(e['kind']=='return_verification_reused' for e in self.session.events),1)
        self.flow.play.assert_not_called()

    def test_unknown_without_loading_cannot_create_proof_or_skip_start_confirmation(self):
        self.execute('RETURN_CAMP',[])
        self.execute('WAIT_RETURN_LOADING',['UNKNOWN','READY','READY'])
        self.execute('VERIFY_RETURN_START',['UNKNOWN','READY','READY'])
        self.assertIsNone(self.flow._return_confirmation)
        self.waits.clear()
        self.execute('VERIFY_START',['READY','READY'],loop=2)
        self.assertEqual(self.waits,[3])
        self.assertFalse(any(e['kind']=='return_verification_reused' for e in self.session.events))

    def test_unstable_ready_never_completes_before_original_deadline(self):
        with self.assertRaises(TimeoutError):
            self.execute('VERIFY_RETURN_START',['READY','UNKNOWN','READY','UNKNOWN'],timeout=8)
        self.assertGreaterEqual(self.session.now(),8)
        self.assertLessEqual(self.session.now(),8.05+1e-9)  # existing guard check granularity
        self.assertIsNone(self.flow._return_confirmation)
        self.assertEqual(self.waits,[3,1,3,1])
        self.assertFalse(self.session.clicks)
        self.flow.play.assert_not_called()

    def test_timeout_can_interrupt_three_second_confirmation(self):
        with self.assertRaises(TimeoutError):self.execute('VERIFY_RETURN_START',['READY'],timeout=1.2)
        self.assertAlmostEqual(self.session.now(),1.2)
        self.assertEqual(len(self.session.captures),1)
        self.flow.play.assert_not_called()

    def test_stop_and_focus_guards_interrupt_wait_before_another_capture(self):
        for error in (StopRequested('F9 pressed'),RuntimeError('Game lost focus or minimized')):
            with self.subTest(error=type(error).__name__):
                self.session.error=None
                before=len(self.session.captures)
                self.session.on_capture=lambda:setattr(self.session,'error',error)
                with self.assertRaises(type(error)):self.execute('VERIFY_RETURN_START',['READY'])
                self.assertEqual(len(self.session.captures)-before,1)
                self.assertIsNone(self.flow._return_confirmation)
                self.assertFalse(self.session.clicks)
        self.flow.play.assert_not_called()

    def test_pause_during_confirmation_preserves_active_wait_and_invalidates_proof(self):
        self.execute('RETURN_CAMP',[]);self.execute('WAIT_RETURN_LOADING',['LOADING'])
        before=self.session.now();wall=self.session.wall
        def pause():
            self.session.paused_total+=5
            self.session.wall+=5
        self.session.on_capture=pause
        self.execute('VERIFY_RETURN_START',['READY','READY'])
        self.assertAlmostEqual(self.session.now()-before,3.1)
        self.assertAlmostEqual(self.session.wall-wall,8.1)
        self.assertIsNone(self.flow._return_confirmation)

    def test_invalid_poll_is_rejected_and_original_profile_remains_opt_out(self):
        for value in [0,-1,True,float('nan'),float('inf'),'1',None]:
            cfg=copy.deepcopy(self.config);cfg['return']['not_ready_poll']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_mission_config(cfg)
        cfg,_=load_mission_config(ROOT/'examples/mission_flow.json')
        self.assertNotIn('not_ready_poll',cfg['return'])
        self.assertFalse(self.config['return']['camp_checkpoint']['enabled'])
        self.assertEqual({k:v for k,v in self.config['return'].items() if k != 'camp_checkpoint'},
                         dict(loading_poll=.5,not_ready_poll=1,time_poll=3,confirmations=2))


class AdaptiveReuseGuardTests(reuse.ReturnReuseTests):
    """Run every existing proof/expiry/focus/window guard with adaptive polling."""
    def new_flow(self):
        super().new_flow()
        self.config['return']['not_ready_poll']=1


class TimedCampSession(FixtureSession):
    ready_after=1.2
    stop_error=None

    def enter(self,state,loop):
        super().enter(state,loop)
        self.entered_at=self.now()

    def check(self):
        if self.state=='VERIFY_RETURN_START' and self.stop_error is not None and self.now()-self.entered_at>=1.2:
            raise self.stop_error

    def capture(self,box):
        if self.state=='VERIFY_RETURN_START':
            self.frame='loading.png' if self.now()-self.entered_at<self.ready_after else '95/talk.png'
        return super().capture(box)


class AdaptiveReturnIntegrationTests(unittest.TestCase):
    def run_case(self,adaptive=True,kind=TimedCampSession,timeout=120):
        config,base=load_mission_config(ROOT/'examples/mission_95.json')
        config['max_loops']=2;config['timeouts']['VERIFY_RETURN_START']=timeout
        if not adaptive:config['return'].pop('not_ready_poll')
        with tempfile.TemporaryDirectory() as directory:
            session=kind(config,directory,ROOT/'resource')
            try:
                with session.log,contextlib.redirect_stdout(io.StringIO()):
                    result=MissionFlow(config,base,session,MissionVision(config,base)).run()
                rows=[json.loads(l) for l in (Path(directory)/'execution.jsonl').read_text().splitlines()]
                self.assertFalse(session.api.held_keys or session.api.held_buttons)
                debug=list(Path(directory).glob('*.png'))
            finally:
                if session.last_image is not None:session.last_image.close()
        return result,rows,bool(debug)

    def test_two_loops_save_time_without_early_macro_or_extra_input(self):
        old,old_rows,_=self.run_case(False)
        new,new_rows,_=self.run_case(True)
        self.assertEqual((old.status,new.status),('SUCCESS','SUCCESS'))
        strip=lambda rows:[{k:v for k,v in r.items() if k not in ('time','active_at')} for r in rows if r['kind']=='action']
        self.assertEqual(strip(old_rows),strip(new_rows))
        states=lambda rows:[r for r in rows if r['kind']=='state_result' and r['state']=='VERIFY_RETURN_START']
        self.assertTrue(all(abs(a['elapsed']-b['elapsed']-1)<1e-6 for a,b in zip(states(old_rows),states(new_rows))))
        self.assertEqual(sum(r['kind']=='return_verification_reused' for r in new_rows),1)
        start=next(r for r in new_rows if r['kind']=='recording_start' and r['recording']=='start' and r['loop']==2)
        verified=next(r for r in new_rows if r['kind']=='state_result' and r['state']=='VERIFY_RETURN_START' and r['loop']==1)
        self.assertGreaterEqual(start['active_at'],verified['active_at'])
        ready=[r for r in new_rows if r['kind']=='hud' and r['state']=='VERIFY_RETURN_START' and r['status']=='READY' and r['loop']==1]
        self.assertAlmostEqual(ready[-1]['active_at']-ready[-2]['active_at'],3)

    def test_timeout_f9_and_focus_failure_safe_stop_save_debug_and_never_start_next_macro(self):
        for error in [None,StopRequested('F9 pressed'),RuntimeError('Game lost focus or minimized')]:
            class Broken(TimedCampSession):ready_after=999;stop_error=error
            with self.subTest(error=error):
                result,rows,debug=self.run_case(kind=Broken,timeout=4.5)
                self.assertEqual((result.status,result.state),('FAIL','VERIFY_RETURN_START'))
                self.assertTrue(debug)
                self.assertFalse(any(r['kind']=='recording_start' and r['loop']==2 for r in rows))
                self.assertEqual(sum(r['kind']=='click' and r['label']=='camp' for r in rows),1)


if __name__=='__main__':unittest.main()
