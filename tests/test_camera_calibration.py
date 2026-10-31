"""Recorded camera angles and safe calibration state-machine regression tests."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tailstopia_macro_replay import StopRequested
from tailstopia_workflow.camera_detection import CameraDetector
from tailstopia_workflow.mission_config import STATES, load_mission_config, validate_mission_config
from tailstopia_workflow.mission_flow import MissionFlow
from tailstopia_workflow.mission_runtime import FixtureSession
from tailstopia_workflow.mission_vision import MissionVision

ROOT = Path(__file__).resolve().parents[1]


class CameraDetectorTests(unittest.TestCase):
    def setUp(self):
        self.config,self.base = load_mission_config(ROOT/"examples/mission_flow.json")
        self.detector = CameraDetector(self.config["camera"],self.base)

    def inspect_file(self,path,origin):
        r = self.detector.roi
        with Image.open(path) as image:
            return self.detector.inspect(image.crop((r[0]+origin[0],r[1]+origin[1],r[2]+origin[0],r[3]+origin[1])))

    def test_only_reference_angle_has_blue_in_narrow_roi(self):
        for n,expected in ((16,False),(17,False),(18,True)):
            with self.subTest(screenshot=n):
                observation = self.inspect_file(ROOT/f"resource/screenshot_{n}.png",(0,0))
                self.assertEqual(observation.aligned,expected)
                if expected:
                    self.assertGreaterEqual(observation.blob_pixels,300)
                else:
                    self.assertEqual(observation.blob_pixels,0)

    def test_all_other_recorded_scenes_reject_blue_checkpoint(self):
        for path in (ROOT/"resource").glob("*.png"):
            if path.name.startswith("screenshot_"):
                continue
            with self.subTest(image=path.name):
                observation = self.inspect_file(path,self.config["reference_client_origin"])
                self.assertFalse(observation.aligned)
                self.assertLess(observation.blob_pixels,300)

    def test_live_stopped_roi_has_no_tent(self):
        # Real last observation from the 20261030_072338_269151_live run.
        with Image.open(ROOT/"tests/fixtures/camera_live_no_tent_roi.png") as image:
            result = self.detector.inspect(image)
        self.assertFalse(result.aligned)
        self.assertEqual(result.blob_pixels,0)

    def test_tent_shift_beyond_previous_twenty_pixel_limit_now_passes(self):
        r = self.detector.roi
        with Image.open(ROOT/"resource/screenshot_18.png") as image:
            roi = image.crop(r)
        for dx in (-65,-33,45):
            with self.subTest(shift=dx),Image.new("RGB",roi.size,(150,100,60)) as shifted:
                shifted.paste(roi,(dx,0))
                self.assertTrue(self.detector.inspect(shifted).aligned)
        roi.close()

    def test_only_small_partial_blue_patch_is_enough(self):
        r = self.detector.roi
        with Image.new("RGB",(r[2]-r[0],r[3]-r[1]),(180,130,70)) as image:
            image.paste((25,65,130),(0,50,20,70))
            observation = self.detector.inspect(image)
        self.assertTrue(observation.aligned)
        self.assertEqual(observation.blob_pixels,400)
        self.assertLess(observation.blue_pixels/((r[2]-r[0])*(r[3]-r[1])),.03)

    def test_speckles_and_bright_sky_do_not_pass(self):
        r = self.detector.roi
        shape = (r[3]-r[1],r[2]-r[0],3)
        value = np.full(shape,(180,130,70),dtype=np.uint8)
        value[::4,::4] = (25,65,130)
        with Image.fromarray(value) as image:
            observation = self.detector.inspect(image)
        self.assertGreater(observation.blue_pixels,300)
        self.assertEqual(observation.blob_pixels,1)
        self.assertFalse(observation.aligned)
        for color in ((0,0,0),(255,255,255),(90,165,240)):
            with Image.new("RGB",(shape[1],shape[0]),color) as image:
                self.assertFalse(self.detector.inspect(image).aligned)

    def test_random_noise_without_coherent_patch_does_not_pass(self):
        r = self.detector.roi
        rng = np.random.default_rng(21030)
        for _ in range(5):
            rgb = rng.integers(0,256,(r[3]-r[1],r[2]-r[0],3),dtype=np.uint8)
            with Image.fromarray(rgb) as image:
                self.assertFalse(self.detector.inspect(image).aligned)

    def test_minimum_patch_area_is_configurable(self):
        self.detector.config["min_blue_blob_pixels"] = 5000
        self.assertFalse(self.inspect_file(ROOT/"resource/screenshot_18.png",(0,0)).aligned)

    def test_wrong_roi_size_fails(self):
        with Image.new("RGB",(43,25)) as image:
            with self.assertRaisesRegex(ValueError,"dimensions"):
                self.detector.inspect(image)

    def test_recorded_left_right_and_center_are_guidance_only(self):
        for n,side,aligned in ((16,"right",False),(17,"left",False),(18,"center",True)):
            with self.subTest(screenshot=n),Image.open(ROOT/f"resource/screenshot_{n}.png") as image:
                result = self.detector.inspect(image.crop(self.detector.direction_roi),self.detector.direction_roi)
                self.assertEqual((result.side,result.aligned),(side,aligned))
                if not aligned:
                    self.assertGreaterEqual(result.side_blob_pixels,300)

    def test_direction_observation_takes_one_capture_and_keeps_center_result_identical(self):
        boxes = []
        with Image.open(ROOT/"resource/screenshot_18.png") as source:
            def capture(box):
                boxes.append(tuple(box))
                return source.crop(box)
            wide = self.detector.observe(capture)
            narrow = self.detector.observe(capture,scan_sides=False)
        self.assertEqual(boxes,[self.detector.direction_roi,self.detector.roi])
        self.assertEqual(wide,narrow)

    def test_both_sides_are_ambiguous_and_cannot_pass_center(self):
        r = self.detector.direction_roi
        with Image.new("RGB",(r[2]-r[0],r[3]-r[1]),(180,130,70)) as image:
            image.paste((25,65,130),(20,20,50,50))
            image.paste((25,65,130),(900,20,930,50))
            result = self.detector.inspect(image,r)
        self.assertEqual(result.side,"ambiguous")
        self.assertFalse(result.aligned)

    def test_invalid_config_rejected_before_inputs(self):
        for key,value in (("min_blue_blob_pixels",0),("min_blue_blob_pixels",100000),
                          ("rotation_seconds",0),("settle_seconds",0),("max_rotations",-1),
                          ("blue_hsv_lower",[-1,0,0]),("blue_hsv_upper",[180,255,255]),
                          ("blue_hsv_lower",[0,0]),("rotation_key","home"),("left_rotation_key","home"),("confirmations",1),
                          ("direction_roi_client",[700,35,850,130]),("direction_roi_client",[160,0,1280,130]),
                          ("search_roi_client",[0,0,0,10])):
            with self.subTest(key=key,value=value):
                config = copy.deepcopy(self.config)
                config["camera"][key] = value
                with self.assertRaises(ValueError):
                    validate_mission_config(config)


class CameraFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config,self.base = load_mission_config(ROOT/"examples/mission_flow.json")
        self.config["max_loops"] = 1
        self.vision = MissionVision(self.config,self.base)

    def run_case(self,frames,through="CALIBRATE_CAMERA",failure=None):
        class Scripted(FixtureSession):
            def enter(session,state,loop):
                super().enter(state,loop)
                if state == "CALIBRATE_CAMERA":
                    session.camera_captures = []
                    session.camera_entered = session.now()

            def capture(session,box):
                if session.state != "CALIBRATE_CAMERA":
                    return super().capture(box)
                if session.api.held_keys or session.api.held_buttons:
                    raise AssertionError("Capture while rotating")
                index = min(len(session.camera_captures),len(frames)-1)
                session.camera_captures.append(session.now())
                if frames[index] == "empty":
                    roi = Image.new("RGB",(box[2]-box[0],box[3]-box[1]),(180,130,70))
                else:
                    with Image.open(ROOT/f"resource/screenshot_{frames[index]}.png") as image:
                        roi = image.crop(box).convert("RGB")
                if session.last_image is not None:
                    session.last_image.close()
                session.last_image,session.last_box = roi.copy(),list(box)
                return roi

            def check(session):
                if session.state == "CALIBRATE_CAMERA":
                    if failure in ("focus","stop") and session.api.held_keys and session.now()-session.camera_entered >= .1:
                        if failure == "stop":
                            raise StopRequested("F9")
                        raise RuntimeError("Game lost focus or minimized")
                    if failure == "confirm_focus" and len(session.camera_captures) >= 2:
                        raise RuntimeError("Game lost focus or minimized")

            def prepare_macro(session,name):
                if failure == "thai" and name == "camera_calibration":
                    raise RuntimeError("Switch to English; no camera input sent")

        session = Scripted(self.config,self.directory,ROOT/"resource")
        flow = MissionFlow(self.config,self.base,session,self.vision)
        with contextlib.redirect_stdout(io.StringIO()):
            result = flow.run(through)
        events = [json.loads(row) for row in (self.directory/"execution.jsonl").read_text().splitlines()]
        self.assertFalse(session.api.held_keys or session.api.held_buttons)
        return result,events,session

    def camera_events(self,events,kind):
        return [e for e in events if e["state"] == "CALIBRATE_CAMERA" and e["kind"] == kind]

    def assert_safe_failure(self,result,events):
        self.assertEqual((result.status,result.state),("FAIL","CALIBRATE_CAMERA"))
        self.assertFalse(any(e["kind"] == "recording_start" and e["recording"] == "mission" for e in events))
        self.assertTrue(list(self.directory.glob("*_CALIBRATE_CAMERA_*.png")))

    def test_state_order_and_aligned_camera_does_not_rotate(self):
        i = STATES.index("CALIBRATE_CAMERA")
        self.assertEqual(STATES[i-1:i+2],("VERIFY_STAGE","CALIBRATE_CAMERA","PLAY_MISSION"))
        result,events,session = self.run_case([18,18])
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual(len(session.camera_captures),2)
        self.assertAlmostEqual(session.camera_captures[1]-session.camera_captures[0],.5)
        self.assertFalse(self.camera_events(events,"action"))
        self.assertEqual(len(self.camera_events(events,"camera_calibrated")),1)

    def test_tent_on_right_does_not_pass_narrow_checkpoint(self):
        self.config["camera"]["max_rotations"] = 0
        result,events,session = self.run_case([16,16])
        self.assert_safe_failure(result,events)
        self.assertEqual(len(session.camera_captures),1)
        self.assertFalse(self.camera_events(events,"action"))
        self.assertFalse(self.camera_events(events,"camera_calibrated"))

    def test_rotate_stop_check_then_macro_only_after_two_confirmations(self):
        result,events,session = self.run_case([17,17,18,18],through="PLAY_MISSION")
        self.assertEqual(result.status,"SUCCESS")
        observations = self.camera_events(events,"camera_observation")
        self.assertEqual([e["confirmations"] for e in observations],[0,0,1,2])
        actions = self.camera_events(events,"action")
        self.assertEqual([(e["method"],e["key"]) for e in actions],[("keyDown","c"),("keyUp","c")]*2)
        for down,up in zip(actions[::2],actions[1::2]):
            self.assertAlmostEqual(up["active_at"]-down["active_at"],.2)
        for a,b in zip(session.camera_captures,session.camera_captures[1:]):
            self.assertGreaterEqual(b-a,.5-1e-8)
        calibrated = self.camera_events(events,"camera_calibrated")[0]
        mission = next(e for e in events if e["kind"] == "recording_start" and e["recording"] == "mission")
        self.assertGreaterEqual(mission["active_at"],calibrated["active_at"])
        summary = self.camera_events(events,"state_result")[0]
        self.assertEqual(summary["timing"]["vision"]["calls"],4)
        self.assertEqual(summary["timing"]["rotation"]["calls"],2)

    def test_lost_match_resets_confirmation_without_rotating_after_first_positive(self):
        result,events,_ = self.run_case([18,17,18,18])
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual([e["confirmations"] for e in self.camera_events(events,"camera_observation")],[1,0,1,2])
        self.assertEqual(len(self.camera_events(events,"action")),2)

    def test_retry_budget_exhaustion_saves_screenshot_and_blocks_macro(self):
        self.config["camera"]["max_rotations"] = 2
        result,events,session = self.run_case([17])
        self.assert_safe_failure(result,events)
        self.assertIn("retry budget exhausted",result.reason)
        self.assertEqual(len(session.camera_captures),3)
        self.assertEqual(len(self.camera_events(events,"action")),4)

    def test_right_tent_uses_z_then_confirms_center_without_more_rotation(self):
        result,events,session = self.run_case([16,18,18])
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual([(e["method"],e["key"]) for e in self.camera_events(events,"action")],
                         [("keyDown","z"),("keyUp","z")])
        self.assertEqual([e["side"] for e in self.camera_events(events,"camera_observation")],
                         ["right","center","center"])
        self.assertEqual(len(session.camera_captures),3)

    def test_left_tent_uses_c(self):
        result,events,_ = self.run_case([17,18,18])
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual([e["key"] for e in self.camera_events(events,"action")],["c","c"])

    def test_unseen_tent_keeps_last_direction(self):
        result,events,_ = self.run_case([16,"empty",18,18])
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual([e["key"] for e in self.camera_events(events,"action")],["z"]*4)
        self.assertEqual([e["side"] for e in self.camera_events(events,"camera_observation")],
                         ["right","unseen","center","center"])

    def test_initial_unseen_tent_uses_configured_fallback(self):
        self.config["camera"]["rotation_key"] = "z"
        result,events,_ = self.run_case(["empty",18,18])
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual([e["key"] for e in self.camera_events(events,"action")],["z","z"])

    def test_side_can_change_direction_without_passing_center(self):
        result,events,_ = self.run_case([16,17,18,18])
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual([e["key"] for e in self.camera_events(events,"action")],["z","z","c","c"])
        self.assertEqual([e["confirmations"] for e in self.camera_events(events,"camera_observation")],[0,0,1,2])

    def test_last_rotation_can_still_receive_two_positive_confirmations(self):
        self.config["camera"]["max_rotations"] = 1
        result,events,_ = self.run_case([17,18,18])
        self.assertEqual(result.status,"SUCCESS")
        self.assertEqual(len(self.camera_events(events,"camera_observation")),3)

    def test_timeout_mid_rotation_releases_key_and_blocks_macro(self):
        self.config["timeouts"]["CALIBRATE_CAMERA"] = .1
        result,events,_ = self.run_case([17])
        self.assert_safe_failure(result,events)
        self.assertIn("TimeoutError",result.reason)
        self.assertEqual([e["method"] for e in self.camera_events(events,"action")],["keyDown","keyUp"])

    def test_focus_loss_mid_rotation_releases_key(self):
        result,events,_ = self.run_case([17],failure="focus")
        self.assert_safe_failure(result,events)
        self.assertIn("lost focus",result.reason)
        self.assertEqual([e["method"] for e in self.camera_events(events,"action")],["keyDown","keyUp"])

    def test_f9_mid_rotation_releases_key(self):
        result,events,_ = self.run_case([17],failure="stop")
        self.assert_safe_failure(result,events)
        self.assertTrue(result.stopped)
        self.assertEqual([e["method"] for e in self.camera_events(events,"action")],["keyDown","keyUp"])

    def test_focus_loss_before_second_confirmation_blocks_macro(self):
        result,events,_ = self.run_case([18,18],failure="confirm_focus")
        self.assert_safe_failure(result,events)
        self.assertFalse(self.camera_events(events,"camera_calibrated"))
        self.assertFalse(self.camera_events(events,"action"))

    def test_keyboard_guard_blocks_rotation(self):
        result,events,_ = self.run_case([17],failure="thai")
        self.assert_safe_failure(result,events)
        self.assertIn("Switch to English",result.reason)
        self.assertFalse(self.camera_events(events,"action"))

    def test_z_and_rotation_settle_durations_are_configurable(self):
        self.config["camera"].update(left_rotation_key="z",rotation_seconds=.25,settle_seconds=.4)
        result,events,session = self.run_case([17,18,18])
        self.assertEqual(result.status,"SUCCESS")
        down,up = self.camera_events(events,"action")
        self.assertEqual((down["key"],up["key"]),("z","z"))
        self.assertAlmostEqual(up["active_at"]-down["active_at"],.25)
        self.assertAlmostEqual(session.camera_captures[1]-session.camera_captures[0],.65)
        self.assertAlmostEqual(session.camera_captures[2]-session.camera_captures[1],.4)


if __name__ == "__main__":
    unittest.main()
