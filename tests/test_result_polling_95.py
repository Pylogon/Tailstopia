"""Stage 95 polls after the initial delay until a real Result appears."""
import math
from pathlib import Path
import unittest

from PIL import Image

import test_mission_outcomes as outcomes


class DelayedResult(outcomes.ResultSession):
    result_at=5
    mission_at=None

    def event(self,kind,**fields):
        if kind=='recording_start' and fields.get('recording')=='mission':
            self.mission_at=self.now()
        super().event(kind,**fields)

    def capture(self,box):
        if self.state=='WAIT_RESULT':
            if self.api.held_keys or self.api.held_buttons:
                raise AssertionError('Result polling overlaps held macro input')
            if self.now()-self.mission_at<self.result_at:
                return Image.new('RGB',(box[2]-box[0],box[3]-box[1]))
        return super().capture(box)


class ResultPolling95Tests(unittest.TestCase):
    setUp=outcomes.MissionOutcomeTests.setUp
    run_case=outcomes.MissionOutcomeTests.run_case

    def test_early_win_polls_after_macro_without_an_additional_five_second_wait(self):
        self.assertEqual(self.config['result']['after_mission_start'],5)
        result,rows,flow,_=self.run_case(DelayedResult)
        self.assertEqual((result.status,result.loops_completed),('SUCCESS',1))
        mission=next(r for r in rows if r['kind']=='recording_start' and r['recording']=='mission')
        first=next(r for r in rows if r['state']=='WAIT_RESULT' and r['kind']=='recognition')
        expected=max(mission['duration'],self.config['result']['after_mission_start'])
        self.assertAlmostEqual(first['active_at']-mission['active_at'],expected)
        timer=next(r for r in rows if r['kind']=='result_timer')
        self.assertAlmostEqual(timer['remaining'],0)
        end=next(r for r in rows if r['kind']=='state_result' and r['state']=='WAIT_RESULT')
        self.assertAlmostEqual(end['elapsed'],self.config['result']['poll'])
        self.assertEqual(flow.mission_counts,{'pass_with_reward':1,'mission_fail':0,'no_reward':0})
        self.assertEqual(sum(r['kind']=='macro_action' and r['recording']=='mission' for r in rows),14)

    def test_defeat_at_twenty_or_twenty_five_seconds_is_polled_and_returns_to_camp(self):
        for ready_at in (20,25):
            with self.subTest(result_at=ready_at):
                self.setUp()
                class Defeated(DelayedResult):
                    result_at=ready_at
                    losses={1}
                result,rows,flow,_=self.run_case(Defeated)
                self.assertEqual((result.status,result.loops_completed),('SUCCESS',1))
                mission=next(r for r in rows if r['kind']=='recording_start' and r['recording']=='mission')
                checks=[r for r in rows if r['state']=='WAIT_RESULT' and r['kind']=='recognition']
                poll=self.config['result']['poll']
                first_at=max(mission['duration'],self.config['result']['after_mission_start'])
                expected=first_at+math.ceil((ready_at-first_at)/poll)*poll
                found=next(r for r in checks if r['found'])
                self.assertAlmostEqual(found['active_at']-mission['active_at'],expected)
                self.assertTrue(all(not r['found'] for r in checks if r['active_at']-mission['active_at']<ready_at))
                self.assertTrue(all(abs(b['active_at']-a['active_at']-poll)<1e-7 for a,b in zip(checks,checks[1:])))
                self.assertFalse(any(r['kind']=='click' and r['label'].startswith('reward_') for r in rows))
                self.assertEqual(sum(r['kind']=='click' and r['label']=='camp' for r in rows),1)
                self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':1,'no_reward':0})

    def test_no_result_after_twenty_five_seconds_does_not_infer_defeat_or_click_camp(self):
        self.config['timeouts']['WAIT_RESULT']=30
        class Absent(DelayedResult):result_at=100
        result,rows,flow,_=self.run_case(Absent)
        self.assertEqual((result.status,result.state),('FAIL','WAIT_RESULT'))
        self.assertIn('TimeoutError',result.reason)
        checks=[r for r in rows if r['state']=='WAIT_RESULT' and r['kind']=='recognition']
        self.assertFalse(any(r['found'] for r in checks))
        self.assertGreater(checks[-1]['active_at']-checks[0]['active_at'],25)
        self.assertFalse(any(r['kind']=='click' and (r['label']=='camp' or r['label'].startswith('reward_')) for r in rows))
        self.assertEqual(flow.mission_counts,{'pass_with_reward':0,'mission_fail':0,'no_reward':0})
        self.assertTrue(list(Path(self.temp.name).glob('*_WAIT_RESULT_failure.png')))


if __name__=='__main__':unittest.main()
