"""Offline-only regression cases; never instantiate a live input session."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from offline.camp_checkpoint import (
    HOUSE_BOX, HOUSE_ROI, HouseCheckpoint, FrameEvidence, OfflinePairCheck,
)
from tailstopia_workflow.start_detection import StartDetector

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT/'tests/fixtures/camp_checkpoint'


class HouseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Image.open(CORPUS/'house_reference.png') as ref:
            cls.detector = HouseCheckpoint(ref)
        cls.hud = StartDetector(ROOT/'examples/step1.json')
        cls.cases = json.loads((CORPUS/'manifest.json').read_text())['samples']

    def test_frozen_saved_image_corpus(self):
        # Regression assertions, not evidence of unseen-scene generalization.
        for case in self.cases:
            with self.subTest(source=case['source']):
                with Image.open(CORPUS/case['file']) as image:
                    house = self.detector.inspect(image.crop((0, 0, 50, 110)))
                    hud = self.hud.inspect(image.crop((50, 0, 93, 25)))
                if case['scene'] == 'camp_open':
                    self.assertTrue(house.found)
                    self.assertEqual(hud.status, 'READY')
                elif not case['scene'].startswith('camp_'):
                    self.assertFalse(house.found)
                else:
                    self.assertFalse(house.found and hud.status == 'READY')

    def test_fixture_integrity_and_recording_split(self):
        development_sources = {c['source_sha256'] for c in self.cases if c['split'] != 'evaluation'}
        development_runs = {c['source'].split('/')[2] for c in self.cases
                            if c['source'].startswith('workflow_runs/') and c['split'] != 'evaluation'}
        for case in self.cases:
            self.assertEqual(hashlib.sha256((CORPUS/case['file']).read_bytes()).hexdigest(),
                             case['fixture_file_sha256'])
            if case['split'] == 'evaluation':
                self.assertNotIn(case['source_sha256'], development_sources)
                self.assertNotIn(case['source'].split('/')[2], development_runs)

    def test_background_outside_disk_has_no_effect_at_reference_position(self):
        case = next(c for c in self.cases if c['split'] == 'reference')
        with Image.open(CORPUS/case['file']) as packed:
            original = np.array(packed.crop((0, 0, 50, 110)).convert('RGB'))
        x = HOUSE_BOX[0]-HOUSE_ROI[0]; y = HOUSE_BOX[1]-HOUSE_ROI[1]
        disk = np.zeros((110, 50), bool)
        disk[y:y+34, x:x+33] = self.detector.mask != 0
        rng = np.random.default_rng(42)
        altered = rng.integers(0, 256, original.shape, dtype=np.uint8)
        altered[disk] = original[disk]
        result = self.detector.inspect(Image.fromarray(altered))
        self.assertTrue(result.found)
        self.assertGreater(result.score, .999)

    def test_constant_frames_rejected(self):
        for color in ('black', 'white', '#897647'):
            self.assertFalse(self.detector.inspect(Image.new('RGB', (50,110), color)).found)

    def test_icon_occlusion_is_not_assumed_safe(self):
        case = next(c for c in self.cases if c['split'] == 'reference')
        with Image.open(CORPUS/case['file']) as image:
            roi = image.crop((0, 0, 50, 110)).convert('RGB')
        roi.paste('black', (7,14,40,48))
        self.assertFalse(self.detector.inspect(roi).found)

    def test_wrong_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.inspect(Image.new('RGB', (50,111)))


class PairTests(unittest.TestCase):
    def setUp(self):
        self.gate = OfflinePairCheck(.5)
        self.first = FrameEvidence('first', 1, True, 'READY', 'return-loop-1')

    def test_two_fresh_joint_frames(self):
        self.assertFalse(self.gate.feed(self.first))
        self.assertTrue(self.gate.feed(replace(self.first, frame_id='second', at=1.5)))

    def test_house_or_time_alone_cannot_confirm(self):
        for frame in (replace(self.first, house=False), replace(self.first, hud_status='UNKNOWN'),
                      replace(self.first, hud_status='LOADING')):
            self.gate.feed(self.first)
            self.assertFalse(self.gate.feed(replace(frame, frame_id='bad', at=1.5)))
            self.assertFalse(self.gate.feed(replace(self.first, frame_id='new', at=2)))

    def test_same_frame_or_short_gap_rejected(self):
        self.gate.feed(self.first)
        self.assertFalse(self.gate.feed(replace(self.first, at=1.5)))
        self.assertFalse(self.gate.feed(replace(self.first, frame_id='early', at=1.4)))

    def test_pause_focus_timeout_and_held_input_reset_evidence(self):
        # Abstract guard flag: actual production safeguards remain untouched.
        for guard in ('pause', 'focus_loss', 'timeout', 'held_input'):
            with self.subTest(guard=guard):
                self.gate.feed(self.first)
                self.assertFalse(self.gate.feed(replace(self.first, guarded=False, at=1.5)))
                self.assertFalse(self.gate.feed(replace(self.first, frame_id='new', at=2)))

    def test_context_change_and_time_reversal_reset(self):
        for changed in (replace(self.first, context='other-loop', at=1.5),
                        replace(self.first, at=.5)):
            self.gate.feed(self.first)
            self.assertFalse(self.gate.feed(changed))

    def test_positive_gap_required(self):
        with self.assertRaises(ValueError):
            OfflinePairCheck(0)


if __name__ == '__main__':
    unittest.main()
