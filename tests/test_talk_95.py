"""Stage 95 Talk regression from live loop 16 and bounded W+A recovery."""
import copy
import unittest
from unittest.mock import patch

from PIL import Image

import test_mission_outcomes as outcomes
from test_talk_detection import recorded_talk_capture
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT=outcomes.ROOT
ROI=(280,170,1000,470)
LOOP16_ROI=(420,210,860,410)
FIXTURE=ROOT/'tests/fixtures/talk_95_loop16_roi.png'
LOOP81_FIXTURE=ROOT/'tests/fixtures/talk_95_loop81_roi.png'


class MissedFixedClick(FixtureSession):
    def click_effect(self,label):
        if label=='talk' and self.click_counts.get('talk',0)==0:
            self.click_counts['talk']=1
            return
        super().click_effect(label)

    def capture(self,box):
        if self.state=='OPEN_MISSION' and self.frame!='95/create.png' and tuple(box)==ROI:
            if self.api.held_keys or self.api.held_buttons:
                raise AssertionError('Recognition during movement')
            return recorded_talk_capture(FIXTURE,box,LOOP16_ROI)
        return super().capture(box)


class NeedsWalk(FixtureSession):
    recovered=False
    movement_holds=0

    def enter(self,state,loop):
        super().enter(state,loop)
        if state=='OPEN_MISSION':self.frame='intiate.png'

    def sleep(self,seconds):
        if self.state=='OPEN_MISSION' and 'right' in self.api.held_buttons:
            if self.api.held_buttons!={'right'} or self.api.held_keys!={'w','a'}:
                raise AssertionError('Wrong recovery combination')
            self.recovered=True
            self.movement_holds+=seconds
        super().sleep(seconds)

    def capture(self,box):
        if self.state=='OPEN_MISSION':
            if self.api.held_keys or self.api.held_buttons:
                raise AssertionError('Recognition during movement')
            if self.recovered and self.frame=='intiate.png':self.frame='95/talk.png'
        return super().capture(box)

    def click_effect(self,label):
        if label!='talk' or self.recovered:super().click_effect(label)


class Talk95Tests(unittest.TestCase):
    setUp=outcomes.MissionOutcomeTests.setUp
    run_case=outcomes.MissionOutcomeTests.run_case

    def test_loop16_requires_complete_roi_and_measured_glyph_variant(self):
        for wide,variant in [(False,False),(True,False),(False,True),(True,True)]:
            cfg=copy.deepcopy(self.config)
            if not wide:cfg['targets']['talk']['search_roi_client']=[540,250,710,360]
            if not variant:
                cfg['targets']['talk']['variants']=[v for v in cfg['targets']['talk']['variants']
                    if v.get('template_image')!='../resource/templates/talk_95_live.png']
            vision=MissionVision(cfg,self.base)
            with self.subTest(wide=wide,variant=variant):
                match=vision.match('talk',lambda box:recorded_talk_capture(FIXTURE,box,LOOP16_ROI))
                self.assertEqual(match.found,wide and variant)
                if wide and variant:
                    self.assertGreater(match.score,.99)
                    self.assertEqual(match.point,(708,303))
                if wide and not variant:self.assertAlmostEqual(match.score,.728119,places=5)
        self.assertEqual(self.config['targets']['talk']['threshold'],.82)
        self.assertEqual(self.config['talk']['point_client'],[615,308])

    def test_loop81_requires_large_glyph_variant_even_inside_previous_roi(self):
        for wide,variant in [(False,False),(True,False),(False,True),(True,True)]:
            cfg=copy.deepcopy(self.config)
            if not wide:cfg['targets']['talk']['search_roi_client']=[420,210,860,410]
            if not variant:
                cfg['targets']['talk']['variants']=[v for v in cfg['targets']['talk']['variants']
                    if v.get('template_image')!='../resource/templates/talk_95_live_large.png']
            vision=MissionVision(cfg,self.base)
            with self.subTest(wide=wide,variant=variant):
                match=vision.match('talk',lambda box:recorded_talk_capture(LOOP81_FIXTURE,box,ROI))
                self.assertEqual(match.found,variant)
                if variant:
                    self.assertGreater(match.score,.99)
                    self.assertEqual(match.point,(469,361))
        self.assertEqual(self.config['targets']['talk']['threshold'],.82)

    def test_loop81_fallback_clicks_actual_talk_before_any_recovery_movement(self):
        class Loop81(MissedFixedClick):
            def capture(session,box):
                if session.state=='OPEN_MISSION' and session.frame!='95/create.png' and tuple(box)==ROI:
                    if session.api.held_keys or session.api.held_buttons:raise AssertionError('Capture during movement')
                    return recorded_talk_capture(LOOP81_FIXTURE,box,ROI)
                return super().capture(box)
        result,rows,_,_=self.run_case(Loop81,through='OPEN_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        clicks=[r for r in rows if r['kind']=='click' and r['label']=='talk']
        self.assertEqual([r['client_point'] for r in clicks],[[615,308],[469,361]])
        self.assertFalse(any(r['kind']=='talk_recovery_move' for r in rows))
        self.assertEqual(sum(r['kind']=='recognition' and r['target']=='talk' for r in rows),1)

    def test_wider_roi_rejects_non_talk_reference_scenes(self):
        paths=[p for p in (ROOT/'resource').glob('*.png')
               if not p.name.startswith('red panda') and p.name!='new redpanda.png']
        paths += [p for p in (ROOT/'resource/95').glob('*.png') if p.name!='talk.png']
        for path in paths:
            with self.subTest(image=str(path)),Image.open(path) as image:
                origin=(0,0) if image.size==(1280,600) else (43,95)
                self.assertFalse(self.vision.match('talk',lambda box:self.vision.crop_reference(image,box,origin)).found)
        for color in (0,128,255):
            with Image.new('RGB',(ROI[2]-ROI[0],ROI[3]-ROI[1]),(color,color,color)) as image:
                self.assertFalse(self.vision.match('talk',lambda box:image.copy()).found)

    def test_one_capture_with_precomputed_masks_and_scales(self):
        calls=[]
        def capture(box):
            calls.append(tuple(box))
            return recorded_talk_capture(FIXTURE,box,LOOP16_ROI)
        with patch('tailstopia_workflow.mission_vision.cv2.resize',side_effect=AssertionError('Runtime resize')), \
             patch('tailstopia_workflow.mission_vision.cv2.inRange',side_effect=AssertionError('Runtime mask')):
            self.assertTrue(self.vision.match('talk',capture).found)
        self.assertEqual(calls,[ROI])

    def test_actual_live_roi_fallback_clicks_talk_without_movement(self):
        result,rows,_,_=self.run_case(MissedFixedClick,through='OPEN_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        clicks=[r for r in rows if r['kind']=='click' and r['label']=='talk']
        self.assertEqual([r['client_point'] for r in clicks],[[615,308],[708,303]])
        self.assertFalse(any(r['kind']=='talk_recovery_move' for r in rows))
        self.assertEqual(sum(r['kind']=='recognition' and r['target']=='talk' for r in rows),1)
        self.assertEqual(sum(r['kind']=='recording_start' for r in rows),1)

    def test_w_a_burst_then_real_talk_opens_mission_list(self):
        self.assertEqual(self.config['talk']['recovery']['keys'],['w','a'])
        self.assertEqual(self.config['talk']['recovery']['max_attempts'],15)
        result,rows,flow,_=self.run_case(NeedsWalk,through='OPEN_MISSION')
        self.assertEqual(result.status,'SUCCESS')
        self.assertAlmostEqual(flow.session.movement_holds,.25)
        moves=[r for r in rows if r['kind']=='talk_recovery_move']
        self.assertEqual(len(moves),1)
        self.assertEqual(moves[0]['keys'],['w','a'])
        downs=[r['key'] for r in rows if r['state']=='OPEN_MISSION' and r['kind']=='action' and r['method']=='keyDown']
        self.assertEqual(downs,['w','a'])
        self.assertTrue(any(r['kind']=='recognition' and r['target']=='mission_list' and r['found'] for r in rows))
        self.assertEqual(sum(r['kind']=='recording_start' for r in rows),1)

    def test_two_w_a_bursts_then_safe_stop_without_mission_macro(self):
        self.config['talk']['recovery']['max_attempts']=2
        class NoTalk(NeedsWalk):
            def capture(session,box):
                if session.state=='OPEN_MISSION':
                    if session.api.held_keys or session.api.held_buttons:raise AssertionError('Capture while held')
                    session.recovered=False
                return super().capture(box)
        result,rows,flow,_=self.run_case(NoTalk)
        self.assertEqual((result.status,result.state),('FAIL','OPEN_MISSION'))
        self.assertIn('attempts=2',result.reason)
        self.assertAlmostEqual(flow.session.movement_holds,.5)
        self.assertEqual(sum(r['kind']=='talk_recovery_move' for r in rows),2)
        self.assertEqual(sum(r['kind']=='click' and r['label']=='talk' for r in rows),1)
        self.assertFalse(any(r['kind']=='recording_start' and r['recording']=='mission' for r in rows))
        self.assertTrue(any(r['kind']=='debug_image' for r in rows))


if __name__=='__main__':unittest.main()
