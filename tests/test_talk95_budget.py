"""Exercise the entire stage-95 recovery budget with realistic vision delays."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tailstopia_workflow.mission_config import load_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from test_talk_recovery import RecoverySession

ROOT=Path(__file__).resolve().parents[1]


class Talk95BudgetTests(unittest.TestCase):
    def setUp(self):
        self.config,self.base=load_mission_config(ROOT/'examples/mission_95.json')
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session=RecoverySession(self.config,self.temp.name)
        self.talk_after=None
        self.menu_after=None
        def match(name,capture):
            capture(name)  # Asserts no recognition while inputs are held.
            self.session.sleep(1.2 if name=='talk' else .05)
            moves=self.movements()
            if name=='talk':found=self.talk_after is not None and moves>=self.talk_after
            else:
                found=(self.session.click_counts.get('talk',0)>=2 or
                       self.menu_after is not None and moves>=self.menu_after)
            return SimpleNamespace(found=found,score=.99 if found else .4,point=(615,308))
        self.flow=MissionFlow(self.config,self.base,self.session,SimpleNamespace(match=match))
        self.flow.state='OPEN_MISSION'
        self.flow.deadline=self.config['timeouts']['OPEN_MISSION']
        self.session.enter('OPEN_MISSION',1)

    def movements(self):
        return sum(call==('mouseDown','right') for call in self.session.api.backend.calls)

    def open(self):
        with self.session.log,contextlib.redirect_stdout(io.StringIO()):self.flow.open_mission()

    def assert_released(self):
        self.assertFalse(self.session.api.held_keys or self.session.api.held_buttons)

    def test_talk_found_after_fifteenth_burst_has_time_to_open_menu(self):
        self.assertEqual(self.config['talk']['recovery']['max_attempts'],15)
        self.assertEqual(self.config['timeouts']['OPEN_MISSION'],120)
        self.talk_after=15
        self.open()
        self.assertEqual(self.movements(),15)
        self.assertEqual(self.session.click_counts['talk'],2)
        self.assertLess(self.session.now(),self.flow.deadline)
        held=[(seconds,keys,buttons) for seconds,keys,buttons in self.session.holds if 'right' in buttons]
        self.assertAlmostEqual(sum(seconds for seconds,_,_ in held),3.75)
        self.assertTrue(all(keys=={'w','a'} and buttons=={'right'} for _,keys,buttons in held))
        rows=[json.loads(line) for line in (Path(self.temp.name)/'execution.jsonl').read_text().splitlines()]
        self.assertEqual([r['attempt'] for r in rows if r['kind']=='talk_recovery_move'],list(range(1,16)))
        self.assert_released()

    def test_persistent_absence_stops_after_fifteen_without_sixteenth_move(self):
        with self.assertRaisesRegex(RuntimeError,'budget exhausted; attempts=15'):self.open()
        self.assertEqual(self.movements(),15)
        self.assertEqual(self.session.click_counts['talk'],1)
        self.assertLess(self.session.now(),self.flow.deadline)
        self.assert_released()

    def test_delayed_menu_stops_recovery_before_remaining_budget(self):
        self.menu_after=3
        self.open()
        self.assertEqual(self.movements(),3)
        self.assertEqual(self.session.click_counts['talk'],1)
        self.assert_released()

    def test_state_timeout_remains_a_hard_stop(self):
        self.flow.deadline=14
        with self.assertRaises(TimeoutError):self.open()
        self.assertLess(self.movements(),15)
        self.assertEqual(self.session.click_counts['talk'],1)
        self.assert_released()


if __name__=='__main__':unittest.main()
