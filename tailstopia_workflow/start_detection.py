"""Step 1: recognize the static Time label, without OCR or input actions."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import time

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Observation:
    status: str
    score: float
    white_fraction: float
    match_xy: tuple | None


class StartDetector:
    def __init__(self, config_path):
        cv2.setNumThreads(1)  # Tiny ROI: avoid a worker pool on the 4GB laptop.
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        self.roi = self._box(self.config["search_roi_client"])
        origin = self.config["reference_client_origin"]
        with Image.open(self.config_path.parent / self.config["reference_image"]) as image:
            x1,y1,x2,y2 = self._box(self.config["template_box_client"])
            box = [x1+origin[0],y1+origin[1],x2+origin[0],y2+origin[1]]
            if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
                raise ValueError("Template box outside reference screenshot")
            reference = np.array(image.crop(box).convert("RGB"))
            # Select actual glyph pixels once, instead of classifying the entire
            # live background by gold HSV (sand/tents can share that color).
            self.match_mask = self.text_mask(reference)
            self.template = cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY)
        glyph = self.template[self.match_mask != 0]
        if not glyph.size or float(glyph.std()) < 1 or float(self.match_mask.std()) < 1:
            raise ValueError("Template must contain a visible Time label")
        if self.template.shape[1] > self.roi[2]-self.roi[0] or self.template.shape[0] > self.roi[3]-self.roi[1]:
            raise ValueError("Search ROI smaller than template")
        for key in ("ready_threshold", "loading_white_fraction"):
            if not 0 < self.config[key] <= 1:
                raise ValueError(f"{key} must be in (0,1]")
        for key in ("poll_seconds", "timeout_seconds"):
            value = self.config[key]
            if not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}")
        confirmations = self.config["ready_confirmations"]
        if isinstance(confirmations, bool) or not isinstance(confirmations, int) or confirmations < 1:
            raise ValueError("ready_confirmations must be a positive integer")

    def text_mask(self, rgb):
        # Used on the reference only to build a fixed spatial glyph mask.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return cv2.inRange(hsv, np.array(self.config["text_hsv_lower"], dtype=np.uint8),
                           np.array(self.config["text_hsv_upper"], dtype=np.uint8))

    @staticmethod
    def _box(box):
        if len(box) != 4 or any(isinstance(v,bool) or not isinstance(v,int) or v < 0 for v in box):
            raise ValueError("Invalid ROI box")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Empty ROI box")
        return tuple(box)

    def crop_screenshot(self, image, origin=None):
        origin = origin if origin is not None else self.config["reference_client_origin"]
        x1,y1,x2,y2 = self.roi
        box = (x1+origin[0],y1+origin[1],x2+origin[0],y2+origin[1])
        if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
            raise ValueError("Search ROI outside screenshot")
        return image.crop(box)

    def inspect(self, roi_image):
        rgb = np.array(roi_image.convert("RGB"))
        expected = (self.roi[3]-self.roi[1], self.roi[2]-self.roi[0])
        if rgb.shape[:2] != expected:
            raise ValueError("ROI image dimensions do not match config")
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        scores = cv2.matchTemplate(gray,self.template,cv2.TM_CCOEFF_NORMED,mask=self.match_mask)
        # Constant candidate regions have zero masked variance. OpenCV may
        # return NaN/Inf (or invalid out-of-range roundoff); never accept those.
        scores[~np.isfinite(scores) | (np.abs(scores) > 1.0001)] = -1.0
        _, maximum, _, point = cv2.minMaxLoc(scores)
        score = max(0.0,min(1.0,float(maximum)))
        white = float(np.mean(np.all(rgb >= self.config["loading_min_channel"], axis=2)))
        # A white loading region never passes READY, regardless of correlation.
        if white >= self.config["loading_white_fraction"]:
            return Observation("LOADING", score, white, None)
        if score >= self.config["ready_threshold"]:
            return Observation("READY", score, white, (point[0]+self.roi[0], point[1]+self.roi[1]))
        return Observation("UNKNOWN", score, white, None)


def wait_for_start(detector, capture, check, on_observation=lambda *args: None,
                   clock=time.monotonic, sleep=time.sleep):
    """Retry only capture/recognition. This function cannot dispatch input."""
    config = detector.config
    started = clock()
    deadline = started + config["timeout_seconds"]
    ready_count = 0
    last = None
    while True:
        check()
        if clock() >= deadline:
            reason = last.status if last is not None else "no frame"
            raise TimeoutError(f"Start not verified after {config['timeout_seconds']}s; last={reason}")
        roi = capture()
        check()
        if clock() >= deadline:
            raise TimeoutError("Start capture exceeded deadline")
        last = detector.inspect(roi)
        check()
        if clock() >= deadline:
            raise TimeoutError("Start recognition exceeded deadline")
        ready_count = ready_count + 1 if last.status == "READY" else 0
        on_observation(last, ready_count, clock()-started)
        if ready_count >= config["ready_confirmations"]:
            return last
        target = min(deadline, clock() + config["poll_seconds"])
        while clock() < target:
            check()
            sleep(min(0.05, target-clock()))
