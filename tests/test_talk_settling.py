"""Recorded Talk pair + log-derived virtual timing; no screen capture or OS input.

Known-case regression, not a generalization/accuracy study. Both word crops
preserve the original detector scores. The missing earlier images are represented
by failed observations, not invented screenshots. CPU costs are simulated from
the failure log; actual rendering/readiness must still be checked in Live runs.
"""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.mission_config import load_mission_config,validate_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_vision import MissionVision,Match
from test_talk_recovery import RecoverySession

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT/'tests/fixtures/talk_settling'


class TalkSettlingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config,base = load_mission_config(ROOT/'examples/mission_95.json')
        vision = MissionVision(config,base)
        cls.recorded = []
        manifest = json.loads((CORPUS/'manifest.json').read_text())
        for case in manifest['sources']:
            path = CORPUS/case['image']
            if hashlib.sha256(path.read_bytes()).hexdigest() != case['image_sha256']:
                raise AssertionError('Fixture changed')
            with Image.new('RGB',(1280,600)) as client,Image.open(path) as word:
                client.paste(word,(310,350))
                cls.recorded.append(vision.match('talk',lambda box:client.crop(box)))

    def test_recorded_last_frame_stays_rejected_and_later_frame_passes(self):
        missing,ready = self.recorded
        self.assertFalse(missing.found)
        self.assertAlmostEqual(missing.score,.7810168266,places=6)
        self.assertTrue(ready.found)
        self.assertAlmostEqual(ready.score,.8446527719,places=6)
        self.assertEqual(ready.point,(352,376))

    def scenario(self,timeout=45,grace=10,settles=True,error=None,late=False):
        config,base = load_mission_config(ROOT/'examples/mission_95.json')
        config['timeouts']['OPEN_MISSION'] = timeout
        config['talk']['recovery'].update(max_attempts=15,stationary_grace_seconds=grace)
        validate_mission_config(config)
        directory = tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        session = RecoverySession(config,directory.name)
        session.enter('OPEN_MISSION',23)
        if error is not None:session.stop_at,session.stop_error = 37,error
        movement_starts = []
        original_prepare = session.prepare_macro
        def prepare(name):
            if name=='talk_recovery':movement_starts.append(session.now())
            original_prepare(name)
        session.prepare_macro = prepare
        captures_after_six = 0
        def match(name,capture):
            nonlocal captures_after_six
            # 50ms capture + 1.30s Talk processing approximates the observed log.
            # This is a controller simulation, not a measured detector speedup.
            session.sleep(.05)
            if name != 'talk':
                return Match(session.click_counts.get('talk',0)>=2,1,(352,376))
            session.sleep(1.30)
            if len(movement_starts)<6:
                return Match(False,.35,None)
            captures_after_six += 1
            if not settles or captures_after_six==1:
                return self.recorded[0]
            if late:session.sleep(4)
            return self.recorded[1]
        vision = SimpleNamespace(match=match)
        flow = MissionFlow(config,base,session,vision)
        flow.state,flow.deadline = 'OPEN_MISSION',timeout
        error_result = None
        with session.log,contextlib.redirect_stdout(io.StringIO()):
            try:flow.execute_state()
            except (TimeoutError,RuntimeError,StopRequested) as exc:error_result=exc
        self.assertFalse(session.api.held_keys or session.api.held_buttons)
        rows=[json.loads(l) for l in (Path(directory.name)/'execution.jsonl').read_text().splitlines()]
        return error_result,session.now(),movement_starts,rows

    def test_old_timeout_fails_but_bounded_extension_finishes_same_six_moves(self):
        old,_,old_moves,old_rows = self.scenario(timeout=35,grace=0)
        new,elapsed,new_moves,new_rows = self.scenario()
        self.assertIsInstance(old,TimeoutError)
        self.assertIsNone(new)
        self.assertLess(elapsed,45)
        self.assertEqual(old_moves,new_moves)
        self.assertEqual(len(new_moves),6)
        self.assertTrue(all(at+.25<=35+1e-8 for at in new_moves))
        clicks=lambda rows:[r for r in rows if r['kind']=='click']
        self.assertEqual(len(clicks(old_rows)),1)
        self.assertEqual([r['client_point'] for r in clicks(new_rows)],[[615,308],[352,376]])
        keys=lambda rows:[(r['method'],r.get('key'),r.get('params')) for r in rows
                          if r['kind']=='action' and r['method'] in ('keyDown','keyUp')]
        self.assertEqual(keys(old_rows),keys(new_rows))

    def test_never_settled_word_times_out_without_spending_grace_on_more_walking(self):
        error,_,moves,rows = self.scenario(settles=False)
        self.assertIsInstance(error,TimeoutError)
        self.assertEqual(len(moves),6)
        self.assertTrue(all(at+.25<=35+1e-8 for at in moves))
        self.assertTrue(any(r['kind']=='retry' and 'stationary checks only' in r.get('reason','') for r in rows))
        self.assertEqual(sum(r['kind']=='click' for r in rows),1)

    def test_guard_interrupts_stationary_checks_without_extra_movement_or_click(self):
        for cause in (StopRequested('F9 pressed'),RuntimeError('Game lost focus')):
            with self.subTest(cause=cause):
                error,_,moves,rows = self.scenario(error=cause)
                self.assertIsInstance(error,type(cause))
                self.assertEqual(len(moves),6)
                self.assertEqual(sum(r['kind']=='click' for r in rows),1)

    def test_verified_word_too_late_for_menu_wait_is_not_clicked(self):
        error,_,moves,rows = self.scenario(late=True)
        self.assertIsInstance(error,TimeoutError)
        self.assertIn('insufficient time',str(error))
        self.assertEqual(len(moves),6)
        self.assertEqual(sum(r['kind']=='click' for r in rows),1)

    def test_grace_validation_and_legacy_opt_out(self):
        config,_ = load_mission_config(ROOT/'examples/mission_95.json')
        for value in (-1,True,float('nan'),float('inf'),config['timeouts']['OPEN_MISSION']):
            with self.subTest(value=value),self.assertRaises(ValueError):
                cfg=copy.deepcopy(config);cfg['talk']['recovery']['stationary_grace_seconds']=value
                validate_mission_config(cfg)
        legacy,_ = load_mission_config(ROOT/'examples/mission_flow.json')
        self.assertNotIn('stationary_grace_seconds',legacy['talk']['recovery'])
        self.assertEqual(legacy['timeouts']['OPEN_MISSION'],35)


if __name__ == '__main__':
    unittest.main()
