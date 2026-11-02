"""Saved-image experiment only. Never imported by the mission runner.

House pixels describe a visible UI checkpoint, NOT input readiness. Parameters
are frozen before evaluating the November 2 recording split; no live capture,
input, configuration mutation or production state transitions exist here.
"""
from dataclasses import dataclass

import cv2
import numpy as np


HOUSE_ROI = (4, 115, 54, 225)  # Client coordinates; Nov 1 development views.
HOUSE_BOX = (11, 129, 44, 163)  # resource/95/initiate.png, client origin 43,95.
TIME_ROI = (346, 0, 389, 25)
THRESHOLD = .90


@dataclass(frozen=True)
class HouseObservation:
    found: bool
    score: float
    center: tuple[float, float]


class HouseCheckpoint:
    def __init__(self, reference, masked=True):
        cv2.setNumThreads(1)
        rgb = np.asarray(reference.convert('RGB'))
        if rgb.shape != (34, 33, 3):
            raise ValueError('Expected 33x34 reference crop')
        self.template = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        yy, xx = np.ogrid[:34, :33]
        # Opaque inner disk excludes all pixels behind/outside the icon.
        self.mask = (((xx-16)**2 + (yy-16.5)**2) <= 14**2).astype('uint8')*255
        self.masked = masked

    def inspect(self, roi_image):
        rgb = np.asarray(roi_image.convert('RGB'))
        if rgb.shape != (110, 50, 3):
            raise ValueError('Expected 50x110 house ROI')
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kwargs = {'mask': self.mask} if self.masked else {}
        scores = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED, **kwargs)
        scores[~np.isfinite(scores) | (np.abs(scores) > 1.0001)] = -1
        _, maximum, _, point = cv2.minMaxLoc(scores)
        center = (point[0]+HOUSE_ROI[0]+16.5, point[1]+HOUSE_ROI[1]+17)
        # Bounded search, fixed UI scale. No full-screen/scale-bank search.
        found = maximum >= THRESHOLD and abs(center[0]-27.5) <= 6
        return HouseObservation(bool(found), float(maximum), center)


@dataclass(frozen=True)
class FrameEvidence:
    frame_id: str
    at: float
    house: bool
    hud_status: str
    context: str
    guarded: bool = True


class OfflinePairCheck:
    """Temporal *proposal* evaluation, not a replacement for live proof guards.

    Caller supplies distinct capture IDs, relative monotonic timestamps and a
    context token. guard=False models pause, focus loss, held input or timeout.
    This does not prove that the game will accept movement after a render.
    """
    def __init__(self, gap=.5):
        if not np.isfinite(gap) or gap <= 0:
            raise ValueError('Positive gap required')
        self.gap = gap
        self.first = None

    def feed(self, evidence):
        if not evidence.guarded or not evidence.house or evidence.hud_status != 'READY':
            self.first = None
            return False
        first = self.first
        if first is None or first.context != evidence.context or evidence.at < first.at:
            self.first = evidence
            return False
        if evidence.frame_id == first.frame_id or evidence.at-first.at < self.gap:
            return False
        self.first = None  # Consume once, no reusable production proof is made.
        return True
