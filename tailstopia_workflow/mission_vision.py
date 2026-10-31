"""Small-ROI OpenCV templates for note.txt's UI, plus reward change detection."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .start_detection import StartDetector
from .camera_detection import CameraDetector


@dataclass(frozen=True)
class Match:
    found: bool
    score: float
    point: tuple | None


class MissionVision:
    def __init__(self, config, base):
        cv2.setNumThreads(1)
        self.config = config
        self.base = Path(base)
        self.start = StartDetector(self.base / config["start_detector"])
        self.camera = CameraDetector(config["camera"],self.base)
        self.targets = {}
        self.target_masks = {}
        origin = config["reference_client_origin"]
        for name,spec in config["targets"].items():
            roi = StartDetector._box(spec["search_roi_client"])
            arrays,masks = [],[]
            for variant in [spec]+spec.get("variants",[]):
                if "template_image" in variant:
                    # A measured UI state (e.g. hover) may be stored as a tight crop.
                    with Image.open(self.base / variant["template_image"]) as image:
                        template = image.convert("RGB")
                else:
                    box = StartDetector._box(variant["template_box_client"])
                    with Image.open(self.base / variant["image"]) as image:
                        template = self.crop_reference(image,box,origin).convert("RGB")
                rgb = np.array(template)
                for scale in spec.get("template_scales",[1.0]):
                    if scale == 1:
                        scaled = rgb
                        array = np.array(template.convert("L"))
                    else:
                        scaled = cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
                        array = cv2.cvtColor(scaled,cv2.COLOR_RGB2GRAY)
                    if array.std() < 1 or array.shape[1] > roi[2]-roi[0] or array.shape[0] > roi[3]-roi[1]:
                        raise ValueError(f"Invalid template: {name}")
                    mask = None
                    if "glyph_hsv_lower" in spec:
                        # Reference-only glyph selection. Do not color-filter
                        # the live background or perform OCR on each poll.
                        mask = cv2.inRange(cv2.cvtColor(scaled,cv2.COLOR_RGB2HSV),
                                           np.array(spec["glyph_hsv_lower"],dtype=np.uint8),
                                           np.array(spec["glyph_hsv_upper"],dtype=np.uint8))
                        selected = array[mask != 0]
                        if not selected.size or selected.std() < 1:
                            raise ValueError(f"Invalid glyph mask: {name}")
                    arrays.append(array)
                    masks.append(mask)
            self.targets[name] = (arrays,roi,spec["threshold"])
            self.target_masks[name] = masks
        self.reward_box = StartDetector._box(config["reward"]["received_box_client"])
        with Image.open(self.base / config["reward"]["empty_image"]) as image:
            self.empty = self.crop_reference(image,self.reward_box,origin).convert("RGB")

    @staticmethod
    def crop_reference(image,box,origin):
        area = (box[0]+origin[0],box[1]+origin[1],box[2]+origin[0],box[3]+origin[1])
        if not (0 <= area[0] < area[2] <= image.width and 0 <= area[1] < area[3] <= image.height):
            raise ValueError("ROI outside reference image")
        return image.crop(area)

    def match(self,name,capture):
        templates,roi,threshold = self.targets[name]
        with capture(roi) as image:
            gray = np.array(image.convert("L"))
        if gray.shape != (roi[3]-roi[1],roi[2]-roi[0]):
            raise ValueError("Capture dimensions differ from ROI")
        best = (-1,None,None)
        for template,mask in zip(templates,self.target_masks[name]):
            if mask is None:
                scores = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
            else:
                scores = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED,mask=mask)
                # Constant masked candidates can yield NaN/Inf in OpenCV.
                scores[~np.isfinite(scores) | (np.abs(scores) > 1.0001)] = -1.0
            _,score,_,point = cv2.minMaxLoc(scores)
            if np.isfinite(score) and score > best[0]:
                best = (score,point,template)
        score,point,template = best
        if point is None:
            return Match(False,0.0,None)
        found = bool(np.isfinite(score) and score >= threshold)
        center = (roi[0]+point[0]+template.shape[1]//2,roi[1]+point[1]+template.shape[0]//2)
        return Match(found,float(score),center if found else None)

    def hud(self,capture):
        with capture(self.start.roi) as image:
            return self.start.inspect(image)

    def change(self,before,after):
        a,b = np.array(before.convert("RGB"),dtype=np.int16),np.array(after.convert("RGB"),dtype=np.int16)
        if a.shape != b.shape:
            raise ValueError("Reward slot image dimensions differ")
        difference = np.max(np.abs(a-b),axis=2)
        fraction = float(np.mean(difference >= self.config["reward"]["change_pixel_delta"]))
        return fraction >= self.config["reward"]["change_fraction"],fraction

    def empty_slot(self,image):
        a,b = np.array(self.empty,dtype=np.int16),np.array(image.convert("RGB"),dtype=np.int16)
        return bool(a.shape == b.shape and np.mean(np.abs(a-b)) <= self.config["reward"]["empty_max_mean_error"])
