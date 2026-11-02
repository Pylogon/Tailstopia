"""Small, reference-masked Camp UI checkpoint, promoted from offline fixtures.

Only the opaque inner house disk is matched. A hit describes visible Camp UI;
the controller additionally requires return/loading evidence and fresh Time HUD.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .start_detection import StartDetector


@dataclass(frozen=True)
class CampObservation:
    found: bool
    score: float
    center: tuple
    hud: object


class CampDetector:
    VERSION = "camp_house_disk_v1"

    def __init__(self, policy, base, start):
        self.start = start
        self.roi = StartDetector._box(policy["search_roi_client"])
        self.threshold = policy["threshold"]
        with Image.open(Path(base)/policy["template_image"]) as image:
            rgb = np.array(image.convert("RGB"))
        if rgb.shape != (34,33,3):
            raise ValueError("Camp house reference must be 33x34 pixels")
        self.template = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        yy,xx = np.ogrid[:34,:33]
        self.mask = (((xx-16)**2+(yy-16.5)**2) <= 14**2).astype("uint8")*255
        if self.template[self.mask != 0].std() < 1:
            raise ValueError("Camp house reference is blank")
        a,b = self.roi,start.roi
        self.capture_box = (min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))

    def inspect_house(self, image):
        if image.size != (self.roi[2]-self.roi[0],self.roi[3]-self.roi[1]):
            raise ValueError("Camp capture dimensions differ from ROI")
        gray = cv2.cvtColor(np.array(image.convert("RGB")),cv2.COLOR_RGB2GRAY)
        scores = cv2.matchTemplate(gray,self.template,cv2.TM_CCOEFF_NORMED,mask=self.mask)
        scores[~np.isfinite(scores) | (np.abs(scores) > 1.0001)] = -1
        _,score,_,point = cv2.minMaxLoc(scores)
        center = (point[0]+self.roi[0]+16.5,point[1]+self.roi[1]+17)
        # Frozen offline position gate; no scale bank, background or OCR search.
        found = bool(score >= self.threshold and abs(center[0]-27.5) <= 6)
        return found,float(score),center

    def observe(self, capture):
        box = self.capture_box
        with capture(box) as image:
            if image.size != (box[2]-box[0],box[3]-box[1]):
                raise ValueError("Camp joint capture dimensions differ from ROI")
            def relative(roi):
                return (roi[0]-box[0],roi[1]-box[1],roi[2]-box[0],roi[3]-box[1])
            with image.crop(relative(self.roi)) as house:
                found,score,center = self.inspect_house(house)
            with image.crop(relative(self.start.roi)) as time_roi:
                hud = self.start.inspect(time_roi)
        return CampObservation(found,score,center,hud)
