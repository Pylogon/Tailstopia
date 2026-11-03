"""Recognize the new stage-95 loading UI from two small, static features.

No OCR, capture or input API. Ignore progress fill, percentage and animated dots.
Both the loading label and gold ornament must match in one stationary image.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .start_detection import Observation,StartDetector


@dataclass(frozen=True)
class LoadingObservation(Observation):
    loading_source: str = "loading_ui_95"
    loading_score: float = 0.0
    loading_parts: tuple = ()


class LoadingDetector:
    def __init__(self,policy,base,time_roi):
        self.roi = StartDetector._box(policy["search_roi_client"])
        self.threshold = policy["threshold"]
        with Image.open(Path(base)/policy["template_image"]) as image:
            rgb = np.array(image.convert("RGB"))
        if rgb.shape != (59,80,3):
            raise ValueError("Loading reference must be 80x59 pixels")
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        # Reference-only masks: text brightness and gold hue, never classify
        # arbitrary live scenery by color. Boxes are relative to the tight crop.
        label = cv2.inRange(hsv,np.array([0,0,90],np.uint8),np.array([179,255,255],np.uint8))
        gold = cv2.inRange(hsv,np.array([10,60,100],np.uint8),np.array([35,255,255],np.uint8))
        self.features = []
        for box,mask in [((20,4,63,19),label),((4,24,36,53),gold)]:
            x,y,right,bottom = box
            template = gray[y:bottom,x:right].copy();mask = mask[y:bottom,x:right].copy()
            selected = template[mask != 0]
            if selected.size < 10 or selected.std() < 1:
                raise ValueError("Loading feature mask is empty or blank")
            self.features.append((box,template,mask))
        a,b = self.roi,time_roi
        self.capture_box = (min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))

    def inspect(self,image):
        if image.size != (80,59):
            raise ValueError("Loading UI capture dimensions differ from ROI")
        gray = np.array(image.convert("L"))
        values,shifts = [],[]
        for (x,y,right,bottom),template,mask in self.features:
            area = gray[y-2:bottom+2,x-2:right+2]
            scores = cv2.matchTemplate(area,template,cv2.TM_CCOEFF_NORMED,mask=mask)
            scores[~np.isfinite(scores) | (np.abs(scores) > 1.0001)] = -1
            _,score,_,point = cv2.minMaxLoc(scores)
            values.append(float(score));shifts.append(point)
        consistent = max(abs(shifts[0][i]-shifts[1][i]) for i in (0,1)) <= 1
        score = min(values)
        return bool(score >= self.threshold and consistent),score,tuple(values)
