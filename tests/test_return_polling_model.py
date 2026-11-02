"""Time-indexed estimates must preserve recorded evidence and proof decisions."""
import unittest

from offline.benchmark_return_polling import ROOT,scenario,simulate
from tailstopia_workflow.mission_config import load_mission_config


class ReturnPollingModelTests(unittest.TestCase):
    def setUp(self):
        self.config,_=load_mission_config(ROOT/'examples/mission_95.json')
        self.case=dict(loop=1,camp_seconds=.1,costs={s:.05 for s in
            ('RETURN_CAMP','WAIT_RETURN_LOADING','VERIFY_RETURN_START','VERIFY_START')},
            hud={'WAIT_RETURN_LOADING':['UNKNOWN','LOADING'],
                 'VERIFY_RETURN_START':['LOADING','READY','READY'],'VERIFY_START':['READY']})

    def test_early_scenario_saves_two_seconds_with_same_capture_count(self):
        before,samples=simulate(self.config,self.case,False)
        signal=scenario(samples,True)
        checked,_=simulate(self.config,self.case,False,signal)
        after,_=simulate(self.config,self.case,True,signal)
        self.assertAlmostEqual(checked['seconds'],before['seconds'])
        self.assertAlmostEqual(before['seconds'],6.9)
        self.assertAlmostEqual(after['seconds'],4.9)
        self.assertEqual((before['captures'],after['captures']),(7,7))
        self.assertTrue(before['reused'] and after['reused'])

    def test_late_scenario_can_cost_two_captures_instead_of_saving_time(self):
        before,samples=simulate(self.config,self.case,False)
        after,_=simulate(self.config,self.case,True,scenario(samples,False))
        self.assertAlmostEqual(after['seconds']-before['seconds'],.1)
        self.assertEqual(after['captures']-before['captures'],2)
        self.assertAlmostEqual(after['wait_seconds'],before['wait_seconds'])

    def test_no_loading_trace_never_gets_reuse_under_either_scenario(self):
        self.case['hud']={'WAIT_RETURN_LOADING':['UNKNOWN','READY','READY'],
                          'VERIFY_RETURN_START':['READY','READY'],'VERIFY_START':['READY','READY']}
        before,samples=simulate(self.config,self.case,False)
        for early in (False,True):
            with self.subTest(early=early):
                after,_=simulate(self.config,self.case,True,scenario(samples,early))
                self.assertFalse(before['reused'] or after['reused'])
                self.assertEqual(after['states']['VERIFY_START']['captures'],2)
                self.assertEqual(after['states']['VERIFY_START']['wait_seconds'],3)


if __name__=='__main__':unittest.main()
