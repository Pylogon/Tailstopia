"""New Loading artwork: saved UI crops + fixture sessions; no OS input.

One reference image only. Dimming/background variants are synthetic robustness
checks, not independent positive samples or a live detection-rate measurement.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image

from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.mission_config import load_mission_config,validate_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT = Path(__file__).resolve().parents[1]


def loading_frame():
    image = Image.new('RGB',(1280,600))
    with Image.open(ROOT/'resource/templates/loading_95.png') as ui:
        image.paste(ui,(474,521))
    with Image.open(ROOT/'tests/fixtures/loading_95/time_roi.png') as time_roi:
        image.paste(time_roi,(346,0))
    return image


class Loading95Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,cls.base = load_mission_config(ROOT/'examples/mission_95.json')
        cls.vision = MissionVision(cls.config,cls.base)

    def test_actual_time_roi_is_unknown_but_new_ui_recognizes_loading(self):
        with loading_frame() as client:
            old = self.vision.hud(lambda box:client.crop(box))
            capture = Mock(side_effect=lambda box:client.crop(box))
            new = self.vision.transition_hud(capture)
        self.assertEqual(old.status,'UNKNOWN')
        self.assertAlmostEqual(old.score,.1785775572,places=6)
        self.assertEqual(new.status,'LOADING')
        self.assertEqual(new.loading_source,'loading_ui_95')
        self.assertGreater(new.loading_score,.99)
        self.assertEqual(len(new.loading_parts),2)
        capture.assert_called_once_with((346,0,554,580))
        self.assertEqual(self.vision.start.roi,(346,0,389,25))

    def test_old_white_loading_and_real_camp_mission_still_keep_their_status(self):
        for name in ('loading.png','95/talk.png','screenshot_18.png','95/vote.png'):
            with self.subTest(name=name),Image.open(ROOT/'resource'/name) as image:
                origin=(0,0) if image.size==(1280,600) else (43,95)
                def capture(box):
                    return image.crop((box[0]+origin[0],box[1]+origin[1],box[2]+origin[0],box[3]+origin[1]))
                expected = self.vision.hud(capture)
                observed = self.vision.transition_hud(capture)
                self.assertEqual(observed.status,expected.status)
                self.assertFalse(hasattr(observed,'loading_source'))

    def test_new_signature_has_no_hits_on_other_reference_screens(self):
        paths=list((ROOT/'resource').glob('*.png'))+list((ROOT/'resource/95').glob('*.png'))
        for path in paths:
            if path.name=='new loading.png':continue  # Source/reference, not an independent test.
            with self.subTest(name=path.name),Image.open(path) as image:
                if image.size not in ((1280,600),(1366,768)):continue
                origin=(0,0) if image.size==(1280,600) else (43,95)
                x,y,right,bottom=self.vision.loading.roi
                with image.crop((x+origin[0],y+origin[1],right+origin[0],bottom+origin[1])) as roi:
                    found,_,_=self.vision.loading.inspect(roi)
                self.assertFalse(found)

    def test_label_and_ornament_are_both_required(self):
        for erased in ((18,2,65,21),(2,22,38,55)):
            with self.subTest(erased=erased),Image.open(ROOT/'resource/templates/loading_95.png') as reference:
                image=reference.convert('RGB');image.paste('#151515',erased)
                self.assertFalse(self.vision.loading.inspect(image)[0]);image.close()

    def test_masks_ignore_fill_and_background_and_tolerate_uniform_dimming(self):
        with Image.open(ROOT/'resource/templates/loading_95.png') as image:
            rgb=np.array(image.convert('RGB'))
        preserve=np.zeros(rgb.shape[:2],bool)
        for (x,y,right,bottom),_,mask in self.vision.loading.features:
            preserve[y:bottom,x:right] |= mask!=0
        altered=np.random.default_rng(7).integers(0,256,rgb.shape,dtype=np.uint8)
        altered[preserve]=rgb[preserve]
        self.assertTrue(self.vision.loading.inspect(Image.fromarray(altered))[0])
        for factor in (.4,.7,1.0):
            with self.subTest(factor=factor):
                dimmed=np.clip(rgb.astype(float)*factor,0,255).astype('uint8')
                self.assertTrue(self.vision.loading.inspect(Image.fromarray(dimmed))[0])

    def test_blank_frames_are_not_new_loading(self):
        for color in ('black','white','#826f43'):
            with Image.new('RGB',(80,59),color) as image:
                self.assertFalse(self.vision.loading.inspect(image)[0])

    def test_invalid_configuration_is_rejected_and_other_profile_is_opt_out(self):
        for key,value in [('threshold',.94),('threshold',True),('threshold',float('nan')),
                          ('search_roi_client',[474,521,553,580]),('template_image','')]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                cfg=copy.deepcopy(self.config);cfg['return']['loading_screen'][key]=value
                validate_mission_config(cfg)
        config,base=load_mission_config(ROOT/'examples/mission_flow.json')
        with patch('tailstopia_workflow.mission_vision.LoadingDetector',side_effect=AssertionError('Opt-out loaded')):
            self.assertIsNone(MissionVision(config,base).loading)


class Loading95IntegrationTests(unittest.TestCase):
    def run_case(self,enabled=True,missed=False,error=None):
        config,base=load_mission_config(ROOT/'examples/mission_95.json')
        config['max_loops']=2
        if not enabled:config['return'].pop('loading_screen')
        class Frames(FixtureSession):
            def __init__(self,*args):
                super().__init__(*args);self.capture_context=(123,43,95,1280,600)
                self.seen=0;self.captures=[];self.lost=None
            def enter(self,state,loop):
                super().enter(state,loop)
                if state=='WAIT_RETURN_LOADING':self.seen=0
            def check(self):
                if self.lost is not None:raise self.lost
            def capture(self,box):
                self.captures.append((self.loop,self.state,self.now(),box))
                if self.state=='WAIT_RETURN_LOADING':
                    if self.seen==0 and not missed:
                        with loading_frame() as client:image=client.crop(box)
                    else:
                        self.frame='95/talk.png';image=super().capture(box)
                    self.seen+=1
                    if self.last_image is not None:self.last_image.close()
                    self.last_image,self.last_box=image.copy(),list(box)
                    self.lost=error
                else:image=super().capture(box)
                self.sleep(.05)
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

    def test_one_brief_loading_frame_restores_proof_and_keeps_all_macro_inputs(self):
        old,old_rows,_,_=self.run_case(False)
        new,rows,captures,_=self.run_case(True)
        self.assertEqual((old.status,new.status),('SUCCESS','SUCCESS'))
        self.assertEqual(sum(r['kind']=='camp_return_verified' for r in rows),2)
        self.assertEqual(sum(r['kind']=='return_verification_reused' for r in rows),1)
        loading=[r for r in rows if r['kind']=='hud' and r.get('loading_source')=='loading_ui_95']
        self.assertEqual(len(loading),2)
        self.assertTrue(all(r['status']=='LOADING' and r['loading_score']>=.95 for r in loading))
        self.assertEqual(sum(state=='WAIT_RETURN_LOADING' for _,state,_,_ in captures),2)
        actions=lambda rs:[{k:v for k,v in r.items() if k not in ('time','active_at')} for r in rs if r['kind']=='action']
        self.assertEqual(actions(old_rows),actions(rows))

    def test_loading_missed_between_polls_does_not_invent_transition_proof(self):
        result,rows,_,_=self.run_case(missed=True)
        self.assertEqual(result.status,'SUCCESS')
        self.assertFalse(any(r['kind']=='camp_return_verified' for r in rows))
        self.assertFalse(any(r['kind']=='return_verification_reused' for r in rows))
        self.assertEqual(sum(r['kind']=='camp_return_fallback' for r in rows),2)

    def test_f9_or_focus_loss_during_new_capture_safe_stops_without_next_loop(self):
        for error in (StopRequested('F9 pressed'),RuntimeError('Game lost focus')):
            with self.subTest(error=error):
                result,rows,_,debug=self.run_case(error=error)
                self.assertEqual((result.status,result.state),('FAIL','WAIT_RETURN_LOADING'))
                self.assertTrue(debug)
                self.assertFalse(any(r['kind']=='recording_start' and r['loop']==2 for r in rows))
                self.assertFalse(any(r['kind']=='camp_return_verified' for r in rows))


if __name__=='__main__':
    unittest.main()
