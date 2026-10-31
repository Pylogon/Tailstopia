"""Confirmed mission defeat is recoverable; automation failures are not defeats."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.mission_config import load_mission_config,validate_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision,Match

ROOT=Path(__file__).resolve().parents[1]


class ResultSession(FixtureSession):
    losses=frozenset()
    def enter(self,state,loop):
        super().enter(state,loop)
        if state in {'WAIT_RESULT','COLLECT_REWARDS','RETURN_CAMP'} and loop in self.losses:
            self.frame='95/fail.png'


class MissionOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.config,self.base=load_mission_config(ROOT/'examples/mission_95.json')
        self.config['max_loops']=1
        self.vision=MissionVision(self.config,self.base)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def run_case(self,kind=ResultSession,through=None,vision=None):
        session=kind(self.config,self.temp.name,ROOT/'resource')
        self.addCleanup(lambda:session.last_image.close() if session.last_image is not None else None)
        flow=MissionFlow(self.config,self.base,session,vision or self.vision)
        output=io.StringIO()
        with session.log,contextlib.redirect_stdout(output):result=flow.run(through)
        rows=[json.loads(line) for line in (Path(self.temp.name)/'execution.jsonl').read_text().splitlines()]
        self.assertFalse(session.api.held_keys or session.api.held_buttons)
        return result,rows,flow,output.getvalue()

    def summary(self,rows):return next(r for r in reversed(rows) if r['kind']=='run_result')

    def test_pass_and_fail_use_one_frame_each_and_cannot_cross_match(self):
        for filename,outcome in [('reward.png','pass'),('fail.png','fail')]:
            calls=[]
            with self.subTest(file=filename),Image.open(ROOT/'resource/95'/filename) as image:
                def capture(box):
                    calls.append(box)
                    return self.vision.crop_reference(image,box,[43,95])
                with patch('tailstopia_workflow.mission_vision.cv2.resize',side_effect=AssertionError('Runtime resize')):
                    observed=self.vision.observe_result(capture)
                self.assertEqual(len(calls),1)
                self.assertLessEqual((calls[0][2]-calls[0][0])*(calls[0][3]-calls[0][1]),260*130)
                self.assertTrue(observed['result'].found)
                self.assertEqual(observed['fail'].found,outcome=='fail')
                self.assertGreater(observed['fail'].score,.99) if outcome=='fail' else self.assertLess(observed['fail'].score,.3)
                self.assertEqual(set(observed),{'result','fail'})

    def test_other_reference_scenes_do_not_confirm_a_defeat(self):
        paths=list((ROOT/'resource').glob('*.png'))+[ROOT/'resource/95'/name for name in ('talk.png','create.png','vote.png','initiate.png') if (ROOT/'resource/95'/name).exists()]
        for path in paths:
            if path.name.startswith('reward'):continue
            with self.subTest(image=path.name),Image.open(path) as image:
                origin=(0,0) if image.size==(1280,600) else (43,95)
                observed=self.vision.observe_result(lambda box:self.vision.crop_reference(image,box,origin))
                self.assertFalse(observed['result'].found and observed['fail'].found)

    def test_loss_win_loss_returns_three_times_and_counts_outcomes_separately(self):
        self.config['max_loops']=3
        class Mixed(ResultSession):losses={1,3}
        result,rows,flow,output=self.run_case(Mixed)
        self.assertEqual((result.status,result.loops_completed),('SUCCESS',3))
        outcomes=[r for r in rows if r['kind']=='mission_outcome']
        self.assertEqual([r['outcome'] for r in outcomes],['FAIL','PASS_WITH_REWARD','FAIL'])
        self.assertEqual([(r['pass_with_reward'],r['mission_fail']) for r in outcomes],[(0,1),(1,1),(1,2)])
        summary=self.summary(rows)
        self.assertEqual((summary['pass_with_reward'],summary['mission_fail']),(1,2))
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':2,'no_reward':0})
        self.assertEqual(sum(r['kind']=='click' and r['label']=='camp' for r in rows),3)
        self.assertEqual(sum(r['kind']=='recording_start' and r['recording']=='mission' for r in rows),3)
        self.assertEqual(sum(r['kind']=='click' and r['label'].startswith('reward_') for r in rows),3)
        for loop in (1,3):
            self.assertFalse(any(r['loop']==loop and r['kind']=='click' and r['label'].startswith('reward_') for r in rows))
        self.assertIn('mission_outcome',output)
        self.assertIn('mission_fail',output)

    def test_all_losses_skip_reward_captures_clicks_and_continue_to_loop_limit(self):
        self.config['max_loops']=2
        class Lost(ResultSession):
            losses={1,2}
            reward_slot_captures=0
            def capture(session,box):
                if session.state=='COLLECT_REWARDS' and tuple(box)==tuple(session.config['reward']['received_box_client']):session.reward_slot_captures+=1
                return super().capture(box)
        result,rows,flow,_=self.run_case(Lost)
        self.assertEqual((result.status,result.loops_completed),('SUCCESS',2))
        self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':2,'no_reward':0})
        self.assertEqual(flow.session.reward_slot_captures,0)
        self.assertFalse(any(r['kind']=='click' and r['label'].startswith('reward_') for r in rows))
        self.assertFalse(any(r['kind']=='reward_verified' for r in rows))
        self.assertEqual(sum(r['kind']=='reward_skipped' for r in rows),2)
        self.assertEqual(sum(r['kind']=='click' and r['label']=='camp' for r in rows),2)

    def test_pass_is_counted_only_after_verified_pickup(self):
        result,rows,flow,_=self.run_case()
        self.assertEqual(result.status,'SUCCESS')
        verified=next(i for i,r in enumerate(rows) if r['kind']=='reward_verified')
        counted=next(i for i,r in enumerate(rows) if r['kind']=='mission_outcome')
        self.assertLess(verified,counted)
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':0,'no_reward':0})

    def test_missing_pickup_keeps_safe_stop_when_recovery_is_disabled(self):
        self.config['reward']['additional_retries']=1
        self.config['reward']['return_if_missing']=False
        class NoPickup(ResultSession):
            def click_effect(session,label):
                if not label.startswith('reward_'):super().click_effect(label)
        result,rows,flow,_=self.run_case(NoPickup)
        self.assertEqual((result.status,result.state),('FAIL','COLLECT_REWARDS'))
        self.assertIn('First pickup not detected',result.reason)
        self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':0,'no_reward':0})
        self.assertFalse(any(r['kind']=='click' and r['label']=='camp' for r in rows))

    def test_verified_pickup_remains_counted_when_return_camp_fails(self):
        class NoCamp(ResultSession):
            def capture(session,box):
                if session.state=='RETURN_CAMP':return Image.new('RGB',(box[2]-box[0],box[3]-box[1]))
                return super().capture(box)
        result,rows,flow,_=self.run_case(NoCamp)
        self.assertEqual((result.status,result.state,result.loops_completed),('FAIL','RETURN_CAMP',0))
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':0,'no_reward':0})
        self.assertEqual(self.summary(rows)['pass_with_reward'],1)

    def test_f9_and_focus_loss_during_grade_confirmation_do_not_count_defeat(self):
        for error in (StopRequested('F9'),RuntimeError('Game lost focus')):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                settle=self.config['reward']['settle']
                class Interrupted(ResultSession):
                    losses={1}
                    entered=None
                    def enter(session,state,loop):
                        super().enter(state,loop)
                        if state=='COLLECT_REWARDS':session.entered=session.now()
                    def check(session):
                        if session.state=='COLLECT_REWARDS' and session.now()-session.entered>=settle+.2-1e-8:raise error
                result,rows,flow,_=self.run_case(Interrupted)
                self.assertEqual(result.status,'FAIL')
                self.assertEqual(result.stopped,isinstance(error,StopRequested))
                self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':0,'no_reward':0})
                self.assertFalse(any(r['kind']=='click' and (r['label'].startswith('reward_') or r['label']=='camp') for r in rows))

    def test_missing_header_safe_stops_without_reward_or_camp(self):
        self.config['timeouts']['COLLECT_REWARDS']=self.config['reward']['settle']+1.2
        original=self.vision.observe_result
        def observe(capture):
            matches=original(capture)
            matches['result']=Match(False,.2,None);matches['fail']=Match(True,1.,(571,162))
            return matches
        self.vision.observe_result=observe
        result,rows,flow,_=self.run_case()
        self.assertEqual((result.status,result.state),('FAIL','COLLECT_REWARDS'))
        self.assertIn('TimeoutError',result.reason)
        self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':0,'no_reward':0})
        self.assertFalse(any(r['kind']=='click' and (r['label'].startswith('reward_') or r['label']=='camp') for r in rows))
        self.assertTrue(list(Path(self.temp.name).glob('*_failure.png')))

    def test_non_failed_grade_preserves_pickup_path_without_requiring_an_s(self):
        original=self.vision.observe_result
        def observe(capture):
            matches=original(capture)
            self.assertEqual(set(matches),{'result','fail'})
            matches['fail']=Match(False,.2,None)
            return matches
        self.vision.observe_result=observe
        result,rows,flow,_=self.run_case()
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':0,'no_reward':0})
        self.assertTrue(any(r['kind']=='reward_verified' for r in rows))

    def test_one_failed_grade_frame_then_two_passes_does_not_trigger_defeat(self):
        original=self.vision.observe_result;calls=[]
        def observe(capture):
            matches=original(capture);calls.append(1)
            if len(calls)==1:
                matches['fail']=Match(True,1.,(571,162))
            return matches
        self.vision.observe_result=observe
        result,rows,flow,_=self.run_case()
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(len(calls),3)
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':0,'no_reward':0})
        self.assertFalse(any(r['kind']=='reward_skipped' for r in rows))

    def test_duplicate_outcome_does_not_increment_and_conflict_is_rejected(self):
        session=ResultSession(self.config,self.temp.name,ROOT/'resource');session.enter('COLLECT_REWARDS',1)
        flow=MissionFlow(self.config,self.base,session,self.vision)
        with session.log,contextlib.redirect_stdout(io.StringIO()):
            flow.record_mission_outcome('FAIL','confirmed')
            flow.record_mission_outcome('FAIL','same result')
            with self.assertRaisesRegex(RuntimeError,'Conflicting'):flow.record_mission_outcome('PASS_WITH_REWARD','inconsistent')
        self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':1,'no_reward':0})
        flow.reset_mission_counts()
        self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':0,'no_reward':0})

    def test_invalid_outcome_config_is_rejected(self):
        for value in (None,{},'missing','result',True):
            cfg=copy.deepcopy(self.config);cfg['result']['failure_target']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_mission_config(cfg)

    def test_legacy_profile_counts_pickup_without_enabling_grade_capture(self):
        self.config,self.base=load_mission_config(ROOT/'examples/mission_flow.json');self.config['max_loops']=1
        self.vision=MissionVision(self.config,self.base)
        result,rows,flow,_=self.run_case(FixtureSession)
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':0,'no_reward':0})
        self.assertFalse(any(r['kind']=='recognition' and r['target']=='mission_failed' for r in rows))


if __name__=='__main__':unittest.main()
