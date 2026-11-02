"""Live detector/flow exercised with saved frames and simulated inputs only."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from offline.benchmark_talk import CORPUS,load_client
from offline.talk_detection import TalkExperiment
import test_mission_outcomes as outcomes
from tailstopia_workflow.mission_config import load_mission_config,validate_mission_config
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT=outcomes.ROOT
CASES={c['id']:c for c in json.loads((CORPUS/'manifest.json').read_text(encoding='utf-8'))['cases']}
ROI=(280,170,1000,520)


class NormalizedTalkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.base=load_mission_config(ROOT/'examples/mission_95.json')
        cls.vision=MissionVision(cls.config,cls.base)
        legacy=copy.deepcopy(cls.config)
        legacy['targets']['talk'].pop('normalized_fallback')
        cls.reference=TalkExperiment(legacy,cls.base)

    def test_promoted_normalizer_preserves_frozen_results_for_all_32_recordings(self):
        self.assertEqual(len(CASES),32)
        for case in CASES.values():
            with self.subTest(case=case['id']),load_client(case) as client:
                rgb=np.array(client.crop(ROI))
                expected=self.reference.normalize(rgb)
                actual=self.vision.talk_normalizer.normalize(rgb)
                self.assertEqual(actual.found,expected.found)
                self.assertEqual(actual.point,expected.point)
                self.assertAlmostEqual(actual.score,expected.score,places=6)
                self.assertEqual(actual.candidates,expected.candidates)

    def test_fast_path_keeps_original_score_point_and_skips_normalization(self):
        calls=[]
        with load_client(CASES['reference_95']) as client:
            expected=self.reference.baseline(client)
            def capture(box):calls.append(tuple(box));return client.crop(box)
            with patch.object(self.vision.talk_normalizer,'normalize',side_effect=AssertionError('Fallback called')), \
                 patch('tailstopia_workflow.mission_vision.cv2.resize',side_effect=AssertionError('Runtime resize')), \
                 patch('tailstopia_workflow.mission_vision.cv2.inRange',side_effect=AssertionError('Runtime HSV')):
                actual=self.vision.match('talk',capture)
        self.assertEqual(calls,[ROI])
        self.assertEqual(actual.method,'template')
        self.assertEqual(actual.point,expected.point)
        self.assertEqual(actual.score,expected.score)

    def test_recorded_failures_normalize_from_one_capture_at_original_threshold(self):
        for case_id,point in [('loop30',(491,372)),('holdout_large',(727,482))]:
            calls=[]
            with self.subTest(case=case_id),load_client(CASES[case_id]) as client:
                def capture(box):calls.append(tuple(box));return client.crop(box)
                actual=self.vision.match('talk',capture)
            self.assertEqual(calls,[ROI])
            self.assertTrue(actual.found)
            self.assertEqual(actual.method,'normalize')
            self.assertEqual(actual.point,point)
            self.assertLess(actual.fast_score,.82)
            self.assertGreaterEqual(actual.score,.82)
        self.assertEqual(self.config['targets']['talk']['threshold'],.82)
        self.assertEqual(self.config['talk']['recovery']['max_attempts'],2)
        self.assertEqual(self.config['talk']['recovery']['hold_seconds'],.25)

    def test_known_unseen_font_miss_stays_unverified(self):
        with load_client(CASES['holdout_regular']) as client:
            result=self.vision.match('talk',lambda box:client.crop(box))
        self.assertFalse(result.found)
        self.assertIsNone(result.point)
        self.assertEqual(result.method,'normalize')
        self.assertIn('no verified Talk',result.reason)

    def test_dense_components_exhaust_cpu_budget_without_click(self):
        rgb=np.zeros((350,720,3),np.uint8)
        for y in range(0,340,16):
            for x in range(0,710,16):rgb[y:y+8,x:x+8]=(120,220,150)
        result=self.vision.talk_normalizer.normalize(rgb)
        self.assertFalse(result.found)
        self.assertIsNone(result.point)
        self.assertIn('budget',result.reason)

    def test_opt_in_config_validation_and_other_profile_stays_legacy(self):
        for policy in [True,{}, {'search_roi_client':[280,170,1000]},
                       {'search_roi_client':[280,170,1000,601]},
                       {'search_roi_client':[281,170,1000,520]},
                       {'search_roi_client':[280,170,1000,520],'threshold':.7}]:
            cfg=copy.deepcopy(self.config)
            cfg['targets']['talk']['normalized_fallback']=policy
            with self.subTest(policy=policy),self.assertRaises(ValueError):validate_mission_config(cfg)
        for key,value in [('template_scales',[.9,1.1]),('glyph_hsv_lower',None)]:
            cfg=copy.deepcopy(self.config)
            if value is None:cfg['targets']['talk'].pop(key)
            else:cfg['targets']['talk'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_mission_config(cfg)
        cfg,base=load_mission_config(ROOT/'examples/mission_flow.json')
        self.assertIsNone(MissionVision(cfg,base).talk_normalizer)


class RecordedTalk(FixtureSession):
    case_id='loop30'
    duplicate=False
    lose_focus=False
    lost=False

    def check(self):
        if self.lost:raise RuntimeError('Game lost focus or minimized')

    def click_effect(self,label):
        if label=='talk' and self.click_counts.get('talk',0)==0:
            self.click_counts['talk']=1
            return
        super().click_effect(label)

    def capture(self,box):
        if self.state=='OPEN_MISSION' and self.frame!='95/create.png' and tuple(box)==ROI:
            if self.api.held_keys or self.api.held_buttons:raise AssertionError('Recognition during movement')
            with load_client(CASES[self.case_id]) as client:
                if self.duplicate:
                    with client.crop((467,364,515,380)) as word:client.paste(word,(700,400))
                image=client.crop(box)
            if self.last_image is not None:self.last_image.close()
            self.last_image,self.last_box=image.copy(),list(box)
            if self.lose_focus:self.lost=True
            return image
        return super().capture(box)


class NormalizedTalkFlowTests(unittest.TestCase):
    setUp=outcomes.MissionOutcomeTests.setUp
    run_case=outcomes.MissionOutcomeTests.run_case

    def test_verified_normalized_talk_clicks_detected_point_without_walking(self):
        result,rows,_,_=self.run_case(RecordedTalk,through='OPEN_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        self.assertEqual([r['client_point'] for r in rows if r['kind']=='click' and r['label']=='talk'],
                         [[615,308],[491,372]])
        self.assertFalse(any(r['kind']=='talk_recovery_move' for r in rows))
        events=[r for r in rows if r['kind']=='recognition' and r.get('detector')=='normalize']
        self.assertEqual(len(events),1)
        self.assertGreaterEqual(events[0]['score'],.82)
        self.assertAlmostEqual(events[0]['fast_score'],.5851224,places=5)
        self.assertIn('verified',events[0]['reason'])

    def test_still_missing_talk_uses_original_two_moves_then_safe_stops(self):
        class Missing(RecordedTalk):case_id='holdout_regular'
        result,rows,_,_=self.run_case(Missing)
        self.assertEqual((result.status,result.state),('FAIL','OPEN_MISSION'))
        self.assertIn('budget exhausted',result.reason)
        moves=[r for r in rows if r['kind']=='talk_recovery_move']
        self.assertEqual(len(moves),2)
        self.assertAlmostEqual(sum(r['duration'] for r in moves),.5)
        self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),1)
        self.assertFalse(any(r['kind']=='recording_start' and r['recording']=='mission' for r in rows))
        self.assertTrue(any(r['kind']=='debug_image' for r in rows))

    def test_ambiguous_words_safe_stop_without_recovery_or_second_click(self):
        class Ambiguous(RecordedTalk):duplicate=True
        result,rows,_,_=self.run_case(Ambiguous)
        self.assertEqual((result.status,result.state),('FAIL','OPEN_MISSION'))
        self.assertIn('ambiguous',result.reason)
        self.assertFalse(any(r['kind']=='talk_recovery_move' for r in rows))
        self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),1)
        self.assertTrue(any(r['kind']=='debug_image' for r in rows))

    def test_focus_loss_during_recognition_prevents_normalized_click(self):
        class FocusLost(RecordedTalk):lose_focus=True
        result,rows,_,_=self.run_case(FocusLost)
        self.assertEqual(result.status,'FAIL')
        self.assertIn('lost focus',result.reason)
        self.assertFalse(any(r['kind']=='talk_recovery_move' for r in rows))
        self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),1)


if __name__=='__main__':unittest.main()
