"""Loading is optional; stable Time HUD passes under the user-selected policy."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from PIL import Image
try:
    from tailstopia_workflow.mission_config import load_mission_config
    from tailstopia_workflow.mission_flow import MissionFlow
    from tailstopia_workflow.start_detection import StartDetector, Observation
    AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name not in {"cv2","numpy"}:
        raise
    AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]


class Session:
    def __init__(self,observations):
        self.observations = list(observations)
        self.time = 0.0
        self.captures = 0
        self.events = []
        self.lose_focus_after_capture = None

    def now(self):
        return self.time

    def sleep(self,seconds):
        self.time += seconds

    def check(self):
        if self.lose_focus_after_capture is not None and self.captures >= self.lose_focus_after_capture:
            raise RuntimeError("Game lost focus or minimized")

    def capture(self,box):
        value = self.observations[min(self.captures,len(self.observations)-1)]
        self.captures += 1
        return value

    def event(self,kind,**fields):
        self.events.append({"kind":kind,**fields})

    def click(self,*args,**kwargs):
        raise AssertionError("Loading verification must not click or replay an action")


def obs(status,score=0):
    return Observation(status,score,1.0 if status == "LOADING" else 0.0,None)


@unittest.skipUnless(AVAILABLE,"Use .venv Python with OpenCV/NumPy")
class LoadingFlowTests(unittest.TestCase):
    def flow(self,destination,observations,timeout=None):
        config,_ = load_mission_config(ROOT/"examples/mission_flow.json")
        flow = MissionFlow.__new__(MissionFlow)
        flow.config = config
        flow.state = "WAIT_STAGE_LOADING" if destination == "Stage" else "WAIT_RETURN_LOADING"
        flow.stage_clicked = -config["stage"]["min_entry_wait"]
        flow.timings = {}
        flow.session = Session(observations)
        flow.vision = SimpleNamespace(hud=lambda capture:capture(None))
        flow.deadline = timeout if timeout is not None else config["timeouts"][flow.state]
        return flow

    def poll(self,flow):
        return flow.config["stage" if flow.state == "WAIT_STAGE_LOADING" else "return"]["loading_poll"]

    def test_live_unknown_then_ready_trace_passes_without_loading(self):
        # execution.jsonl from 20261029_201558_180222_live contains exactly
        # 20 UNKNOWN followed by 46 READY at this score, without any LOADING.
        flow = self.flow("Stage",[obs("UNKNOWN")]*20+[obs("READY",.9645951986312866)]*46)
        flow.wait_loading(self.poll(flow))
        self.assertEqual(flow.session.captures,22)
        self.assertEqual(sum(e["kind"] == "loading_skipped" for e in flow.session.events),1)
        self.assertAlmostEqual(flow.session.time,10.5)
        self.assertTrue(all("white_fraction" in e for e in flow.session.events if e["kind"] == "hud"))

    def test_unknown_then_ready_camp_passes_without_loading(self):
        flow = self.flow("Camp",[obs("UNKNOWN"),obs("READY",1),obs("READY",1)])
        flow.wait_loading(self.poll(flow))
        self.assertEqual(flow.session.captures,3)
        self.assertAlmostEqual(flow.session.time,1)

    def test_ready_alone_passes_both_directions_after_two_confirmations(self):
        for destination in ("Stage","Camp"):
            with self.subTest(destination=destination):
                flow = self.flow(destination,[obs("READY",1)],timeout=1.1)
                flow.wait_loading(self.poll(flow))
                self.assertEqual(flow.session.captures,2)
                self.assertAlmostEqual(flow.session.time,.5)
                self.assertTrue(any(e["kind"] == "loading_skipped" for e in flow.session.events))

    def test_stage_ready_cannot_pass_before_existing_ten_second_minimum(self):
        flow = self.flow("Stage",[obs("READY",1)])
        flow.stage_clicked = 0
        flow.wait_loading(self.poll(flow))
        self.assertAlmostEqual(flow.session.time,10.5)
        self.assertEqual(flow.session.captures,22)
        self.assertEqual(flow.config["stage"]["min_entry_wait"],10)

    def test_ready_interrupted_by_unknown_requires_two_new_confirmations(self):
        for destination in ("Stage","Camp"):
            with self.subTest(destination=destination):
                flow = self.flow(destination,[obs("READY",1),obs("UNKNOWN"),obs("READY",1),obs("READY",1)])
                flow.wait_loading(self.poll(flow))
                self.assertEqual(flow.session.captures,4)
                self.assertAlmostEqual(flow.session.time,1.5)

    def test_single_ready_then_unknown_times_out_without_passing(self):
        flow = self.flow("Stage",[obs("READY",1),obs("UNKNOWN")],timeout=1.1)
        with self.assertRaises(TimeoutError):
            flow.wait_loading(self.poll(flow))
        self.assertFalse(any(e["kind"] == "loading_skipped" for e in flow.session.events))

    def test_latest_live_failure_roi_now_passes_without_loading(self):
        detector = StartDetector(ROOT/"examples/step1.json")
        with Image.open(ROOT/"tests/fixtures/stage_missed_loading_ready_roi.png") as image:
            ready = detector.inspect(image)
        self.assertEqual(ready.status,"READY")
        self.assertGreater(ready.score,.99)
        flow = self.flow("Stage",[obs("UNKNOWN")]*21+[ready,ready])
        flow.wait_loading(self.poll(flow))
        self.assertEqual(flow.session.captures,23)
        self.assertTrue(any(e["kind"] == "loading_skipped" for e in flow.session.events))

    def test_unknown_alone_still_uses_original_timeout(self):
        for destination,timeout in (("Stage",60),("Camp",120)):
            with self.subTest(destination=destination):
                flow = self.flow(destination,[obs("UNKNOWN")])
                self.assertEqual(flow.deadline,timeout)
                with self.assertRaises(TimeoutError):
                    flow.wait_loading(self.poll(flow))
                self.assertAlmostEqual(flow.session.time,timeout)

    def test_visible_loading_keeps_existing_path_in_both_directions(self):
        for destination in ("Stage","Camp"):
            with self.subTest(destination=destination):
                flow = self.flow(destination,[obs("UNKNOWN"),obs("LOADING")])
                flow.wait_loading(self.poll(flow))
                self.assertEqual(flow.session.captures,2)
                self.assertAlmostEqual(flow.session.time,.5)
                self.assertFalse(any(e["kind"] == "transition_unverified" for e in flow.session.events))

    def test_single_ready_is_not_a_confirmed_return(self):
        flow = self.flow("Stage",[obs("UNKNOWN"),obs("READY",.96),obs("UNKNOWN"),obs("LOADING")])
        flow.wait_loading(self.poll(flow))
        self.assertEqual(flow.session.captures,4)
        self.assertFalse(any(e["kind"] == "transition_unverified" for e in flow.session.events))

    def test_focus_loss_before_capture_stops_without_observations(self):
        for destination in ("Stage","Camp"):
            with self.subTest(destination=destination):
                flow = self.flow(destination,[obs("READY",1)])
                flow.session.lose_focus_after_capture = 0
                with self.assertRaisesRegex(RuntimeError,"Game lost focus"):
                    flow.wait_loading(self.poll(flow))
                self.assertEqual(flow.session.captures,0)

    def test_focus_loss_before_ready_confirmation_is_not_swallowed(self):
        for destination in ("Stage","Camp"):
            with self.subTest(destination=destination):
                flow = self.flow(destination,[obs("UNKNOWN"),obs("READY",.96),obs("READY",.96)])
                flow.session.lose_focus_after_capture = 3
                with self.assertRaisesRegex(RuntimeError,"Game lost focus"):
                    flow.wait_loading(self.poll(flow))
                self.assertFalse(any(e["kind"] == "transition_unverified" for e in flow.session.events))

    def test_real_ready_roi_and_camp_both_match_the_same_time_label(self):
        detector = StartDetector(ROOT/"examples/step1.json")
        with Image.open(ROOT/"tests/fixtures/stage_loading_live_ready_roi.png") as image:
            stage = detector.inspect(image)
        with Image.open(ROOT/"resource/intiate.png") as image:
            camp = detector.inspect(detector.crop_screenshot(image))
        self.assertEqual((stage.status,camp.status),("READY","READY"))
        self.assertGreater(stage.score,.99)  # Glyph mask ignores changing scenery.
        self.assertEqual(tuple(detector.roi),(346,0,389,25))


    def test_verify_stage_accepts_unchanged_glyph_on_gold_background_at_existing_poll(self):
        import numpy as np
        detector = StartDetector(ROOT/"examples/step1.json")
        config = detector.config
        origin = config["reference_client_origin"]
        x1,y1,x2,y2 = config["template_box_client"]
        with Image.open(ROOT/"resource/intiate.png") as image:
            glyph = np.array(image.crop((x1+origin[0],y1+origin[1],x2+origin[0],y2+origin[1])).convert("RGB"))
        # Synthetic camera/scenery change only, keeping the real glyph intact.
        rgb = np.full((25,43,3),(190,140,50),dtype=np.uint8)
        x,y = x1-detector.roi[0],y1-detector.roi[1]
        patch = rgb[y:y+glyph.shape[0],x:x+glyph.shape[1]]
        selected = detector.match_mask != 0
        patch[selected] = glyph[selected]
        frame = Image.fromarray(rgb)
        self.addCleanup(frame.close)
        flow = self.flow("Stage",[])
        flow.state = "VERIFY_STAGE"
        flow.stage_clicked = -flow.config["stage"]["min_entry_wait"]
        flow.deadline = flow.config["timeouts"]["VERIFY_STAGE"]
        def capture(box):
            self.assertEqual(tuple(box),tuple(detector.roi))
            flow.session.captures += 1
            return frame.copy()
        def hud(capture):
            with capture(detector.roi) as image:
                return detector.inspect(image)
        flow.session.capture = capture
        flow.vision = SimpleNamespace(hud=hud)
        flow.execute_state()
        self.assertEqual(flow.session.captures,2)
        self.assertAlmostEqual(flow.session.time,flow.config["stage"]["time_poll"])
        self.assertEqual([e["status"] for e in flow.session.events if e["kind"] == "hud"],["READY","READY"])
