"""Offline-only safety and recorded regression tests; no game input/capture."""
import ast
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from offline.benchmark_talk import CORPUS, load_client
from offline.talk_detection import TalkExperiment
from tailstopia_workflow.mission_config import load_mission_config

ROOT=Path(__file__).resolve().parents[1]


class DynamicTalkOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config,base=load_mission_config(ROOT/'examples/mission_95.json')
        cls.config['targets']['talk'].pop('normalized_fallback',None)
        cls.detector=TalkExperiment(cls.config,base)
        cls.manifest=json.loads((CORPUS/'manifest.json').read_text(encoding='utf-8'))
        cls.cases={c['id']:c for c in cls.manifest['cases']}

    def test_split_by_source_run_and_no_duplicate_frame_leakage(self):
        groups={};hashes=set();crop_hashes=set()
        for case in self.cases.values():
            self.assertNotIn(case['source_sha256'],hashes)
            hashes.add(case['source_sha256'])
            self.assertNotIn(case['image_sha256'],crop_hashes)
            crop_hashes.add(case['image_sha256'])
            self.assertEqual(groups.setdefault(case['group'],case['split']),case['split'])
            self.assertEqual(hashlib.sha256((CORPUS/case['image']).read_bytes()).hexdigest(),case['image_sha256'])
        self.assertEqual(sum(c['talk_box'] is not None and c['split']=='holdout' for c in self.cases.values()),1)
        self.assertEqual(self.cases['holdout_large']['split'],'challenge_regression')
        self.assertEqual(self.cases['loop30']['split'],'development')
        self.assertEqual(self.cases['loop16']['split'],'template_regression')
        self.assertEqual(self.cases['loop81']['split'],'template_regression')

    def test_recorded_loop30_false_negative_and_normalized_location(self):
        with load_client(self.cases['loop30']) as image:
            before=self.detector.baseline(image)
            expanded=self.detector.baseline(image,True)
            self.assertFalse(before.found)
            self.assertAlmostEqual(before.score,.5851224,places=5)
            self.assertFalse(expanded.found)  # larger search area alone cannot fix scale
            rgb=self.detector._frame(image)
            for method in ('normalize','coarse_to_fine'):
                result=getattr(self.detector,method)(rgb)
                with self.subTest(method=method):
                    self.assertTrue(result.found)
                    self.assertGreaterEqual(result.score,.82)
                    self.assertEqual(result.point,(491,372))
        self.assertEqual(self.detector.threshold,self.config['targets']['talk']['threshold'])

    def test_current_fast_path_returns_unchanged_without_running_fallback(self):
        with load_client(self.cases['reference_95']) as image:
            before=self.detector.baseline(image)
            with patch.object(self.detector,'normalize',side_effect=AssertionError('Unneeded fallback')), \
                 patch.object(self.detector,'coarse_to_fine',side_effect=AssertionError('Unneeded fallback')):
                self.assertEqual(self.detector.detect(image,'normalize'),before)
                self.assertEqual(self.detector.detect(image,'coarse_to_fine'),before)

    def test_large_recorded_label_below_live_roi_is_localized_offline(self):
        case=self.cases['holdout_large']
        with load_client(case) as image:
            self.assertFalse(self.detector.baseline(image).found)
            self.assertFalse(self.detector.baseline(image,True).found)
            rgb=self.detector._frame(image)
            for method in ('normalize','coarse_to_fine'):
                result=getattr(self.detector,method)(rgb)
                with self.subTest(method=method):
                    self.assertTrue(result.found)
                    self.assertGreaterEqual(result.score,.82)
                    x,y=result.point;box=case['talk_box']
                    self.assertTrue(box[0]<=x<box[2] and box[1]<=y<box[3])
                    self.assertGreater(y,self.config['targets']['talk']['search_roi_client'][3])

    def test_known_unseen_font_failure_abstains_instead_of_lowering_threshold(self):
        # This observed miss remains a limitation, not a passing accuracy claim.
        with load_client(self.cases['holdout_regular']) as image:
            rgb=self.detector._frame(image)
            for method in ('normalize','coarse_to_fine'):
                result=getattr(self.detector,method)(rgb)
                with self.subTest(method=method):
                    self.assertFalse(result.found)
                    self.assertLess(result.score,.82)
                    self.assertIsNone(result.point)
                    self.assertIn('safe_stop',result.reason)

    def test_independent_negative_recordings_return_no_click_coordinates(self):
        for case in self.cases.values():
            if case['split']!='holdout' or case['talk_box'] is not None:continue
            with load_client(case) as image:
                rgb=self.detector._frame(image)
                for method in ('normalize','coarse_to_fine'):
                    with self.subTest(image=case['id'],method=method):
                        result=getattr(self.detector,method)(rgb)
                        self.assertFalse(result.found)
                        self.assertIsNone(result.point)

    def test_flat_images_never_produce_click_coordinates(self):
        for value in (0,128,255):
            rgb=np.full((350,720,3),value,np.uint8)
            for method in ('normalize','coarse_to_fine'):
                with self.subTest(value=value,method=method):
                    result=getattr(self.detector,method)(rgb)
                    self.assertFalse(result.found)
                    self.assertIsNone(result.point)
                    self.assertIn('safe_stop',result.reason)

    def test_distinct_passing_labels_are_ambiguous(self):
        # Exercise the decision rule independently from segmentation quality.
        for method in ('normalize','coarse_to_fine'):
            result=self.detector._finish([(.95,(10,10,50,25)),(.90,(200,90,240,105))],.95,method,2)
            self.assertFalse(result.found)
            self.assertIsNone(result.point)
            self.assertIn('ambiguous',result.reason)

    def test_proposal_budget_exhaustion_cannot_return_a_click(self):
        with patch.object(self.detector,'proposals',return_value=None):
            result=self.detector.normalize(np.zeros((350,720,3),np.uint8))
        self.assertFalse(result.found)
        self.assertIsNone(result.point)
        self.assertIn('budget',result.reason)

    def test_invalid_frame_or_method_rejected(self):
        for image in (Image.new('RGB',(720,350)),Image.new('L',(1280,600))):
            with image,self.assertRaises(ValueError):self.detector.detect(image)
        with Image.new('RGB',(1280,600)) as image,self.assertRaises(ValueError):
            self.detector.detect(image,'unknown')

    def test_no_live_flow_imports_offline_prototype(self):
        for file in (ROOT/'tailstopia_workflow').glob('*.py'):
            tree=ast.parse(file.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):names=[alias.name for alias in node.names]
                elif isinstance(node,ast.ImportFrom):names=[node.module or '']
                else:continue
                self.assertFalse(any(name=='offline' or name.startswith('offline.') for name in names),str(file))
        # There is deliberately no input adapter/session passed to TalkExperiment.
        tree=ast.parse((ROOT/'offline/talk_detection.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names=[alias.name for alias in node.names]
            elif isinstance(node,ast.ImportFrom):names=[node.module or '']
            else:continue
            self.assertFalse(any(any(token in n for token in ('pyautogui','pynput','mission_runtime','mission_flow')) for n in names))


if __name__=='__main__':unittest.main()
