"""Talk recognition with fixed reference glyph masks and a measured text size."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tailstopia_workflow.mission_config import load_mission_config, validate_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/"tests/fixtures/talk_live_background_roi.png"
SECOND_FIXTURE = ROOT/"tests/fixtures/talk_second_live_background_roi.png"


class TalkDetectionTests(unittest.TestCase):
    def setUp(self):
        self.config,self.base = load_mission_config(ROOT/"examples/mission_flow.json")
        self.vision = MissionVision(self.config,self.base)

    def fixture_match(self,vision):
        with Image.open(FIXTURE) as image:
            return vision.match("talk",lambda box:image.copy())

    def test_live_failure_old_score_and_fixed_score_at_same_threshold(self):
        old = copy.deepcopy(self.config)
        for key in ("glyph_hsv_lower","glyph_hsv_upper","template_scales"):
            old["targets"]["talk"].pop(key)
        old["targets"]["talk"]["variants"] = [v for v in old["targets"]["talk"]["variants"] if "template_image" not in v]
        before = self.fixture_match(MissionVision(old,self.base))
        self.assertFalse(before.found)
        self.assertAlmostEqual(before.score,.5451096,places=5)
        after = self.fixture_match(self.vision)
        self.assertTrue(after.found)
        self.assertGreater(after.score,.87)
        self.assertEqual(after.point,(575,293))
        self.assertEqual(self.config["targets"]["talk"]["threshold"],.82)

    def test_mask_alone_does_not_fix_measured_font_size_difference(self):
        config = copy.deepcopy(self.config)
        config["targets"]["talk"]["template_scales"] = [1.0]
        config["targets"]["talk"]["variants"] = [v for v in config["targets"]["talk"]["variants"] if "template_image" not in v]
        self.assertFalse(self.fixture_match(MissionVision(config,self.base)).found)

    def test_second_live_failure_passes_with_measured_variant_at_same_threshold(self):
        before_config = copy.deepcopy(self.config)
        before_config["targets"]["talk"]["variants"] = [v for v in before_config["targets"]["talk"]["variants"] if "template_image" not in v]
        before_vision = MissionVision(before_config,self.base)
        with Image.open(SECOND_FIXTURE) as image:
            before = before_vision.match("talk",lambda box:image.copy())
            after = self.vision.match("talk",lambda box:image.copy())
        self.assertFalse(before.found)
        self.assertAlmostEqual(before.score,.7576707,places=5)
        self.assertTrue(after.found)
        self.assertGreater(after.score,.99)
        self.assertEqual(after.point,(562,297))
        self.assertEqual(self.config["targets"]["talk"]["threshold"],.82)

    def test_original_reference_views_still_pass(self):
        for filename in ("red panda1.png","red panda2.png"):
            with self.subTest(image=filename),Image.open(ROOT/"resource"/filename) as image:
                match = self.vision.match("talk",lambda box:self.vision.crop_reference(image,box,[43,95]))
                self.assertTrue(match.found)
                self.assertGreater(match.score,.99)

    def test_other_reference_scenes_do_not_match_talk(self):
        for path in (ROOT/"resource").glob("*.png"):
            if path.name.startswith("red panda") or path.name == "new redpanda.png":
                continue
            with self.subTest(image=path.name),Image.open(path) as image:
                origin = (0,0) if image.size == (1280,600) else (43,95)
                match = self.vision.match("talk",lambda box:self.vision.crop_reference(image,box,origin))
                self.assertFalse(match.found)

    def test_glyph_match_survives_synthetic_background_changes(self):
        templates,roi,_ = self.vision.targets["talk"]
        template = templates[-1]
        selected = self.vision.target_masks["talk"][-1] != 0
        height,width = roi[3]-roi[1],roi[2]-roi[0]
        for color in ((0,0,0),(255,255,255),(180,130,60),(25,60,200),(90,190,130)):
            rgb = np.full((height,width,3),color,dtype=np.uint8)
            patch_rgb = rgb[30:30+template.shape[0],20:20+template.shape[1]]
            patch_rgb[selected] = np.repeat(template[selected,None],3,axis=1)
            with self.subTest(background=color),Image.fromarray(rgb) as image:
                match = self.vision.match("talk",lambda box:image.copy())
                self.assertTrue(match.found)
                self.assertGreater(match.score,.99)

    def test_flat_and_random_images_are_rejected(self):
        _,roi,_ = self.vision.targets["talk"]
        shape = (roi[3]-roi[1],roi[2]-roi[0],3)
        rng = np.random.default_rng(221030)
        for array in [np.full(shape,c,dtype=np.uint8) for c in (0,128,255)]+[rng.integers(0,256,shape,dtype=np.uint8) for _ in range(5)]:
            with Image.fromarray(array) as image:
                self.assertFalse(self.vision.match("talk",lambda box:image.copy()).found)

    def test_one_capture_without_runtime_color_filtering_or_resizing(self):
        calls = []
        def capture(box):
            calls.append(tuple(box))
            return Image.open(FIXTURE).convert("RGB")
        with patch("tailstopia_workflow.mission_vision.cv2.resize",side_effect=AssertionError("Runtime resize")), \
             patch("tailstopia_workflow.mission_vision.cv2.inRange",side_effect=AssertionError("Runtime color filtering")):
            self.assertTrue(self.vision.match("talk",capture).found)
        self.assertEqual(calls,[tuple(self.config["targets"]["talk"]["search_roi_client"])])

    def test_invalid_glyph_config_rejected(self):
        for key,value in (("template_scales",[]),("template_scales",[0]),("template_scales",[float("nan")]),
                          ("glyph_hsv_lower",[40,15]),("glyph_hsv_upper",[180,255,255]),
                          ("glyph_hsv_lower",[-1,0,0])):
            with self.subTest(key=key,value=value):
                config = copy.deepcopy(self.config)
                config["targets"]["talk"][key] = value
                with self.assertRaises(ValueError):
                    validate_mission_config(config)

    def test_live_background_fallback_follows_full_polling_window(self):
        self.run_live_fallback(FIXTURE,[575,293])

    def test_second_live_background_uses_existing_fallback_without_extra_retry(self):
        self.run_live_fallback(SECOND_FIXTURE,[562,297])

    def test_new_start_reference_matches_at_updated_fixed_talk_position(self):
        point = self.config["talk"]["point_client"]
        self.assertEqual(point,[478,321])
        self.assertEqual(self.config["targets"]["talk"]["search_roi_client"],[432,250,597,355])
        self.assertEqual(self.config["targets"]["talk"]["threshold"],.82)
        self.assertEqual(self.config["talk"]["first_wait"],5)
        with Image.open(ROOT/"resource/new redpanda.png") as image:
            match = self.vision.match("talk",lambda box:self.vision.crop_reference(image,box,[43,95]))
        self.assertTrue(match.found)
        self.assertGreater(match.score,.99)
        self.assertEqual(match.point,tuple(point))

    def test_new_start_recording_then_fixed_talk_click_keeps_existing_flow(self):
        from tailstopia_macro_replay import recording
        actions,duration = recording(self.base/self.config["start_recording"])
        class NewStart(FixtureSession):
            def enter(session,state,loop):
                super().enter(state,loop)
                if state == "OPEN_MISSION":
                    session.frame = "new redpanda.png"
        self.config["max_loops"] = 1
        with tempfile.TemporaryDirectory() as directory:
            session = NewStart(self.config,directory,ROOT/"resource")
            with session.log,contextlib.redirect_stdout(io.StringIO()):
                result = MissionFlow(self.config,self.base,session,self.vision).run("OPEN_MISSION")
            rows = [json.loads(line) for line in (Path(directory)/"execution.jsonl").read_text().splitlines()]
            if session.last_image is not None:
                session.last_image.close()
        self.assertEqual(result.status,"SUCCESS")
        clicks = [row for row in rows if row["kind"] == "click" and row["label"] == "talk"]
        self.assertEqual(len(clicks),1)
        self.assertEqual(clicks[0]["client_point"],[478,321])
        self.assertEqual(clicks[0]["desktop_point"],[521,416])
        play = next(row for row in rows if row["kind"] == "state_result" and row["state"] == "PLAY_START")
        self.assertAlmostEqual(play["elapsed"],duration)
        opening = next(row for row in rows if row["kind"] == "state_result" and row["state"] == "OPEN_MISSION")
        self.assertAlmostEqual(opening["elapsed"],self.config["talk"]["first_wait"]+self.config["ui"]["click_hold"])
        self.assertEqual(sum(row["kind"] == "macro_action" for row in rows),len(actions))
        self.assertFalse(any(row["kind"] == "recognition" and row["target"] == "talk" for row in rows))

    def run_live_fallback(self,fixture,expected_point):
        class LiveBackground(FixtureSession):
            def click_effect(session,label):
                if label == "talk" and session.click_counts.get(label,0) < 1:
                    session.click_counts[label] = session.click_counts.get(label,0)+1
                else:
                    super().click_effect(label)

            def capture(session,box):
                if session.state == "OPEN_MISSION" and session.frame != "mission.png" and tuple(box) == tuple(session.config["targets"]["talk"]["search_roi_client"]):
                    with Image.open(fixture) as image:
                        return image.convert("RGB")
                return super().capture(box)

        self.config["max_loops"] = 1
        with tempfile.TemporaryDirectory() as directory:
            session = LiveBackground(self.config,directory,ROOT/"resource")
            with contextlib.redirect_stdout(io.StringIO()):
                result = MissionFlow(self.config,self.base,session,self.vision).run("OPEN_MISSION")
            events = [json.loads(row) for row in (Path(directory)/"execution.jsonl").read_text().splitlines()]
        self.assertEqual(result.status,"SUCCESS")
        clicks = [e for e in events if e["kind"] == "click" and e["label"] == "talk"]
        self.assertEqual(clicks[0]["client_point"],self.config["talk"]["point_client"])
        self.assertEqual(len(clicks),2)
        self.assertEqual(clicks[1]["client_point"],expected_point)  # Existing template fallback.
        state = next(e for e in events if e["kind"] == "state_result" and e["state"] == "OPEN_MISSION")
        policy = self.config["talk"]
        expected = policy["first_wait"]+policy["poll_timeout"]+policy["fallback_wait"]+2*self.config["ui"]["click_hold"]
        self.assertAlmostEqual(state["elapsed"],expected)
        self.assertEqual(policy["first_wait"],5)
        self.assertFalse(session.api.held_keys or session.api.held_buttons)


if __name__ == "__main__":
    unittest.main()
