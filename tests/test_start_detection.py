from dataclasses import replace
from pathlib import Path
import unittest

from PIL import Image
try:
    from tailstopia_workflow.start_detection import StartDetector, wait_for_start
    AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name not in {"cv2", "numpy"}:
        raise
    AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]


class FakeTime:
    def __init__(self):
        self.now = 0.0
    def clock(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


@unittest.skipUnless(AVAILABLE, "Use .venv Python with OpenCV/NumPy installed")
class StartDetectionTests(unittest.TestCase):
    def setUp(self):
        self.detector = StartDetector(ROOT/"examples/step1.json")

    def roi(self, filename):
        with Image.open(ROOT/"resource"/filename) as image:
            return self.detector.crop_screenshot(image)

    def test_ready_across_different_scenery(self):
        for name in ("intiate.png","red panda1.png","red panda2.png","startinagme .png"):
            with self.subTest(name=name):
                result = self.detector.inspect(self.roi(name))
                self.assertEqual(result.status,"READY")
                self.assertGreaterEqual(result.score,0.85)

    def test_loading_reference_is_not_ready(self):
        result = self.detector.inspect(self.roi("loading.png"))
        self.assertEqual(result.status,"LOADING")

    def test_dialog_and_reward_screens_are_not_ready(self):
        for name in ("mission.png","room list.png","startmission.png",
                     "reward1.png","reward2.png","reward3.png","reward4.png"):
            with self.subTest(name=name):
                self.assertNotEqual(self.detector.inspect(self.roi(name)).status,"READY")

    def test_blank_and_dark_frames(self):
        size = self.roi("loading.png").size
        self.assertEqual(self.detector.inspect(Image.new("RGB",size,"white")).status,"LOADING")
        self.assertEqual(self.detector.inspect(Image.new("RGB",size,"black")).status,"UNKNOWN")

    def test_wrong_roi_size_rejected(self):
        with self.assertRaises(ValueError):
            self.detector.inspect(Image.new("RGB",(2,2)))

    def test_client_origin_translation(self):
        with Image.open(ROOT/"resource/intiate.png") as image:
            moved = Image.new("RGB",(image.width+80,image.height+60))
            moved.paste(image,(80,60))
        self.assertEqual(self.detector.inspect(self.detector.crop_screenshot(moved,(123,155))).status,"READY")

    def test_loading_then_two_ready_confirmations(self):
        timer = FakeTime()
        frames = iter([self.roi("loading.png"),self.roi("intiate.png"),self.roi("red panda2.png")])
        observations = []
        result = wait_for_start(self.detector,lambda:next(frames),lambda:None,
                                lambda value,count,elapsed:observations.append((value.status,count,elapsed)),
                                timer.clock,timer.sleep)
        self.assertEqual(result.status,"READY")
        self.assertEqual([r[:2] for r in observations],[("LOADING",0),("READY",1),("READY",2)])
        self.assertAlmostEqual(timer.now,6)

    def test_unknown_resets_confirmation(self):
        timer = FakeTime()
        frames = iter([self.roi(n) for n in ("intiate.png","mission.png","intiate.png","intiate.png")])
        counts = []
        wait_for_start(self.detector,lambda:next(frames),lambda:None,
                       lambda value,count,elapsed:counts.append(count),timer.clock,timer.sleep)
        self.assertEqual(counts,[1,0,1,2])
        self.assertAlmostEqual(timer.now,9)

    def test_timeout_never_returns_ready(self):
        timer = FakeTime()
        self.detector.config["timeout_seconds"] = 5
        frame = self.roi("loading.png")
        with self.assertRaises(TimeoutError):
            wait_for_start(self.detector,lambda:frame,lambda:None,clock=timer.clock,sleep=timer.sleep)
        self.assertAlmostEqual(timer.now,5)

    def test_stop_during_polling(self):
        timer = FakeTime()
        def check():
            if timer.now >= 0.2:
                raise RuntimeError("F9 requested")
        with self.assertRaisesRegex(RuntimeError,"F9"):
            wait_for_start(self.detector,lambda:self.roi("loading.png"),check,clock=timer.clock,sleep=timer.sleep)
        self.assertLess(timer.now,0.26)

    def test_slow_capture_cannot_pass_timeout(self):
        timer = FakeTime()
        self.detector.config["timeout_seconds"] = 1
        def capture():
            timer.sleep(2)
            return self.roi("intiate.png")
        with self.assertRaises(TimeoutError):
            wait_for_start(self.detector,capture,lambda:None,clock=timer.clock,sleep=timer.sleep)


    def test_live_camera_background_regression(self):
        import cv2
        import numpy as np
        with Image.open(ROOT/"tests/fixtures/time_hud_camera_roi.png") as image:
            actual = self.detector.inspect(image)
            binary = self.detector.text_mask(np.array(image.convert("RGB")))
        old = cv2.minMaxLoc(cv2.matchTemplate(binary,self.detector.match_mask,cv2.TM_CCOEFF_NORMED))[1]
        self.assertAlmostEqual(old,.853089,places=5)
        self.assertEqual(actual.status,"READY")
        self.assertGreater(actual.score,.99)
        self.assertEqual(self.detector.config["ready_threshold"],.85)

    def test_time_glyph_survives_synthetic_camera_background_changes(self):
        import cv2
        import numpy as np
        config = self.detector.config
        origin = config["reference_client_origin"]
        x1,y1,x2,y2 = config["template_box_client"]
        with Image.open(ROOT/"resource/intiate.png") as image:
            reference = np.array(image.crop((x1+origin[0],y1+origin[1],x2+origin[0],y2+origin[1])).convert("RGB"))
        roi = self.detector.roi
        x,y = x1-roi[0],y1-roi[1]
        foreground = self.detector.match_mask != 0
        historical_false_negatives = 0
        for background in [(40,90,130),(190,140,50),(200,150,80),(20,100,20),(180,75,40),(0,0,0)]:
            with self.subTest(background=background):
                # Preserve the exact real glyph; change only its surroundings.
                array = np.full((roi[3]-roi[1],roi[2]-roi[0],3),background,dtype=np.uint8)
                patch = array[y:y+reference.shape[0],x:x+reference.shape[1]]
                patch[foreground] = reference[foreground]
                old = cv2.minMaxLoc(cv2.matchTemplate(self.detector.text_mask(array),
                                                     self.detector.match_mask,cv2.TM_CCOEFF_NORMED))[1]
                historical_false_negatives += old < config["ready_threshold"]
                result = self.detector.inspect(Image.fromarray(array))
                self.assertEqual(result.status,"READY")
                self.assertGreater(result.score,.99)
        self.assertEqual(historical_false_negatives,2)

    def test_solid_gold_near_constant_and_noise_without_text_are_not_ready(self):
        import numpy as np
        from math import isfinite
        colors = [(190,140,50),(200,150,80),(80,80,80),(0,0,0),(255,255,255)]
        for color in colors:
            with self.subTest(color=color):
                result = self.detector.inspect(Image.new("RGB",(43,25),color))
                self.assertNotEqual(result.status,"READY")
                self.assertTrue(isfinite(result.score))
        rng = np.random.default_rng(20261029)
        for _ in range(20):
            array = rng.integers(0,256,(25,43,3),dtype=np.uint8)
            self.assertNotEqual(self.detector.inspect(Image.fromarray(array)).status,"READY")
        nearly_flat = np.full((25,43,3),(190,140,50),dtype=np.uint8)
        nearly_flat[8,12] += 1
        self.assertNotEqual(self.detector.inspect(Image.fromarray(nearly_flat)).status,"READY")

    def test_other_hud_words_do_not_match_time(self):
        for label,box in [("Mana",(218,99,249,114)),("Score",(307,99,338,114)),("Ping",(488,99,519,114))]:
            with self.subTest(label=label),Image.open(ROOT/"resource/intiate.png") as image:
                roi = Image.new("RGB",(43,25),(40,90,130))
                roi.paste(image.crop(box),(6,4))
                self.assertNotEqual(self.detector.inspect(roi).status,"READY")
