"""Detect part of the blue tent in a bounded checkpoint ROI, not its exact pose."""
from dataclasses import dataclass

import cv2
import numpy as np

from .start_detection import StartDetector


@dataclass(frozen=True)
class CameraObservation:
    aligned: bool
    blue_pixels: int
    blob_pixels: int
    point: tuple | None
    side: str = "unseen"
    side_blob_pixels: int = 0
    side_point: tuple | None = None


class CameraDetector:
    def __init__(self, config, base):
        cv2.setNumThreads(1)
        self.config = config
        self.roi = StartDetector._box(config["search_roi_client"])
        self.direction_roi = StartDetector._box(config["direction_roi_client"])
        self.lower = np.array(config["blue_hsv_lower"],dtype=np.uint8)
        self.upper = np.array(config["blue_hsv_upper"],dtype=np.uint8)

    @staticmethod
    def blob(mask,origin):
        blue_pixels = int(cv2.countNonZero(mask))
        # A small connected patch is enough; scattered pixels cannot align.
        count,_,stats,centers = cv2.connectedComponentsWithStats(mask,connectivity=8)
        if count <= 1:
            return blue_pixels,0,None
        index = 1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        point = (origin[0]+int(round(centers[index][0])),
                 origin[1]+int(round(centers[index][1])))
        return blue_pixels,int(stats[index,cv2.CC_STAT_AREA]),point

    def inspect(self, image, box=None):
        box = self.roi if box is None else tuple(box)
        if box not in (self.roi,self.direction_roi):
            raise ValueError("Unexpected camera capture ROI")
        rgb = np.array(image.convert("RGB"))
        if rgb.shape[:2] != (box[3]-box[1],box[2]-box[0]):
            raise ValueError("Camera capture dimensions differ from ROI")
        mask = cv2.inRange(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV),self.lower,self.upper)
        x1,y1,x2,y2 = self.roi
        # The center always uses the identical pixels/threshold, including when
        # it is cropped from the single wider direction capture.
        center = mask[y1-box[1]:y2-box[1],x1-box[0]:x2-box[0]]
        blue_pixels,blob_pixels,point = self.blob(center,(x1,y1))
        minimum = self.config["min_blue_blob_pixels"]
        if blob_pixels >= minimum:
            return CameraObservation(True,blue_pixels,blob_pixels,point,"center")
        if box == self.roi:
            return CameraObservation(False,blue_pixels,blob_pixels,point)
        # Use sides only to steer, never to pass the center checkpoint. Both
        # sides being blue is ambiguous: keep the previous search direction.
        _,left,left_point = self.blob(mask[:,:x1-box[0]],(box[0],box[1]))
        _,right,right_point = self.blob(mask[:,x2-box[0]:],(x2,box[1]))
        if left >= minimum and right >= minimum:
            return CameraObservation(False,blue_pixels,blob_pixels,point,"ambiguous")
        if left >= minimum:
            return CameraObservation(False,blue_pixels,blob_pixels,point,"left",left,left_point)
        if right >= minimum:
            return CameraObservation(False,blue_pixels,blob_pixels,point,"right",right,right_point)
        return CameraObservation(False,blue_pixels,blob_pixels,point)

    def observe(self,capture,scan_sides=True):
        # One capture, one HSV conversion. Use only the narrow ROI when
        # confirming an already aligned camera; no recognition during rotation.
        box = self.direction_roi if scan_sides else self.roi
        with capture(box) as image:
            return self.inspect(image,box)
