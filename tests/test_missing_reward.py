"""Exhausted pickup retries can return safely without claiming a reward."""
import copy
import unittest

from PIL import Image

import test_mission_outcomes as outcomes
from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.mission_config import validate_mission_config


class NoPickup(outcomes.ResultSession):
    attempts=0
    last_click=None

    def click_effect(self,label):
        if label=='reward_1':
            self.attempts+=1
            self.last_click=self.now()
        if not label.startswith('reward_'):
            super().click_effect(label)


class MissingRewardTests(unittest.TestCase):
    setUp=outcomes.MissionOutcomeTests.setUp
    run_case=outcomes.MissionOutcomeTests.run_case
    summary=outcomes.MissionOutcomeTests.summary

    def test_three_retries_then_camp_and_next_loop_without_false_pass_or_defeat(self):
        self.assertEqual(self.config['reward']['additional_retries'],3)
        self.assertTrue(self.config['reward']['return_if_missing'])
        self.config['max_loops']=2
        result,rows,flow,output=self.run_case(NoPickup)
        self.assertEqual((result.status,result.loops_completed),('SUCCESS',2))
        self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':0,'no_reward':2})
        for loop in (1,2):
            rewards=[r for r in rows if r['loop']==loop and r['kind']=='click' and r['label'].startswith('reward_')]
            self.assertEqual([r['label'] for r in rewards],['reward_1']*4)
            self.assertEqual(sum(r['loop']==loop and r['kind']=='retry' and r.get('attempt') in (1,2,3) for r in rows),3)
            skipped=next(r for r in rows if r['loop']==loop and r['kind']=='reward_skipped')
            self.assertEqual((skipped['attempts'],skipped['retries']),(4,3))
        self.assertEqual(sum(r['kind']=='click' and r['label']=='camp' for r in rows),2)
        self.assertFalse(any(r['kind']=='reward_verified' for r in rows))
        self.assertEqual(self.summary(rows)['no_reward'],2)
        self.assertIn('no_reward',output)

    def test_defeat_missing_and_received_rewards_have_separate_counts(self):
        self.config['max_loops']=3
        class Mixed(outcomes.ResultSession):
            losses={1}
            def click_effect(session,label):
                if session.loop!=2 or not label.startswith('reward_'):
                    super().click_effect(label)
        result,rows,flow,_=self.run_case(Mixed)
        self.assertEqual((result.status,result.loops_completed),('SUCCESS',3))
        self.assertEqual([r['outcome'] for r in rows if r['kind']=='mission_outcome'],['FAIL','NO_REWARD','PASS_WITH_REWARD'])
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':1,'no_reward':1})
        self.assertEqual(sum(r['kind']=='click' and r['label']=='camp' for r in rows),3)

    def test_late_first_pickup_avoids_an_unnecessary_retry_click(self):
        class Late(NoPickup):
            def capture(session,box):
                if session.last_click is not None and session.now()-session.last_click>=.8:
                    session.received=True
                return super().capture(box)
        result,rows,flow,_=self.run_case(Late)
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(sum(r['kind']=='click' and r['label']=='reward_1' for r in rows),1)
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':0,'no_reward':0})

    def test_last_retry_delayed_pickup_is_counted_before_returning(self):
        class Late(NoPickup):
            def capture(session,box):
                if session.attempts==4 and session.now()-session.last_click>=.8:
                    session.received=True
                return super().capture(box)
        result,rows,flow,_=self.run_case(Late)
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual(sum(r['kind']=='click' and r['label']=='reward_1' for r in rows),4)
        self.assertTrue(any(r['kind']=='reward_verified' for r in rows))
        self.assertFalse(any(r['kind']=='reward_skipped' for r in rows))
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':0,'no_reward':0})

    def test_result_disappearing_after_last_retry_safe_stops(self):
        result_roi=tuple(self.config['targets']['result']['search_roi_client'])
        class Disappeared(NoPickup):
            def capture(session,box):
                if session.attempts==4 and session.now()-session.last_click>=.8 and tuple(box)==result_roi:
                    return Image.new('RGB',(box[2]-box[0],box[3]-box[1]))
                return super().capture(box)
        result,rows,flow,_=self.run_case(Disappeared)
        self.assertEqual((result.status,result.state),('FAIL','COLLECT_REWARDS'))
        self.assertIn('Mission Result disappeared after reward retry budget',result.reason)
        self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':0,'no_reward':0})
        self.assertFalse(any(r['kind']=='click' and r['label']=='camp' for r in rows))

    def test_focus_loss_or_f9_during_final_wait_never_returns_blindly(self):
        for error in (RuntimeError('Game lost focus'),StopRequested('F9')):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                class Interrupted(NoPickup):
                    def check(session):
                        if session.attempts==4 and session.now()-session.last_click>=.5:
                            raise error
                result,rows,flow,_=self.run_case(Interrupted)
                self.assertEqual(result.status,'FAIL')
                self.assertEqual(result.stopped,isinstance(error,StopRequested))
                self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':0,'no_reward':0})
                self.assertFalse(any(r['kind']=='click' and r['label']=='camp' for r in rows))

    def test_missing_camp_safe_stops_after_unconfirmed_reward(self):
        self.config['max_loops']=2
        class MissingCamp(NoPickup):
            def capture(session,box):
                if session.state=='RETURN_CAMP':
                    return Image.new('RGB',(box[2]-box[0],box[3]-box[1]))
                return super().capture(box)
        result,rows,flow,_=self.run_case(MissingCamp)
        self.assertEqual((result.status,result.state,result.loops_completed),('FAIL','RETURN_CAMP',0))
        self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':0,'no_reward':1})
        self.assertFalse(any(r['kind']=='click' and r['label']=='camp' for r in rows))
        self.assertFalse(any(r['loop']==2 for r in rows))

    def test_return_if_missing_requires_a_boolean(self):
        for value in (None,0,1,'true',[],{}):
            cfg=copy.deepcopy(self.config);cfg['reward']['return_if_missing']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_mission_config(cfg)


if __name__=='__main__':unittest.main()
