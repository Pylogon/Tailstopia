"""Bounded Talk normalization fallback, promoted from the offline experiment.

No capture or input API: inspect one supplied RGB ROI, preserve threshold, and
return no point for missing or ambiguous evidence. Templates are cached once.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class NormalizedDetection:
    found: bool
    score: float
    point: tuple | None
    box: tuple | None
    path: str
    reason: str
    candidates: int = 0


def finite_scores(gray, template, mask):
    scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    scores[~np.isfinite(scores) | (np.abs(scores) > 1.0001)] = -1
    return scores


class TalkNormalizer:
    def __init__(self, templates, masks, spec):
        self.roi = tuple(spec['normalized_fallback']['search_roi_client'])
        self.threshold = spec['threshold']
        self.lower = np.array(spec['glyph_hsv_lower'], np.uint8)
        self.upper = np.array(spec['glyph_hsv_upper'], np.uint8)
        self.profiles = []
        scales = spec.get('template_scales', [1.0])
        offset = scales.index(1.0)
        stride = len(scales)
        for gray, mask in zip(templates[offset::stride], masks[offset::stride]):
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            parts = stats[1:n]
            h = max(parts[:, 3])
            parts = [p for p in parts if p[3] >= h * .5 and p[4] >= 4]
            if not parts:
                continue
            x = min(p[0] for p in parts); y = min(p[1] for p in parts)
            right = max(p[0]+p[2] for p in parts); bottom = max(p[1]+p[3] for p in parts)
            self.profiles.append((gray[y:bottom, x:right].copy(), mask[y:bottom, x:right].copy()))

    def _finish(self, hits, score, path, count):
        # All passing proposals must describe the same word. Separate matching
        # labels are ambiguous: never select an arbitrary highest-scoring one.
        hits.sort(reverse=True)
        if not hits:
            return NormalizedDetection(False,float(score),None,None,path,'safe_stop: no verified Talk',count)
        best_score, box = hits[0]
        center = ((box[0]+box[2])//2,(box[1]+box[3])//2)
        for _, other in hits[1:]:
            point=((other[0]+other[2])//2,(other[1]+other[3])//2)
            if max(abs(point[0]-center[0]),abs(point[1]-center[1])) > max(10,(box[3]-box[1])*.6):
                return NormalizedDetection(False,float(best_score),None,None,path,'safe_stop: ambiguous labels',count)
        x,y=self.roi[:2]
        return NormalizedDetection(True,float(best_score),(center[0]+x,center[1]+y),
                         (box[0]+x,box[1]+y,box[2]+x,box[3]+y),path,'verified normalized Talk',count)

    def proposals(self, rgb):
        mask=cv2.inRange(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV),self.lower,self.upper)
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        parts=[tuple(map(int,p)) for p in stats[1:] if 5<=p[3]<=40 and
               2<=p[2]<=p[3]*1.5 and p[4]>=5]
        if len(parts)>512:return None
        boxes=set()
        for ax,ay,aw,ah,_ in parts:
            line=[p for p in parts if .5*ah<=p[3]<=1.2*ah and
                  abs(p[1]+p[3]-ay-ah)<=max(2,ah*.18) and
                  abs(p[0]-ax)<=4*ah]
            line.sort()
            for size in (3,4,5):
                for start in range(len(line)-size+1):
                    group=line[start:start+size]
                    x=min(p[0] for p in group);y=min(p[1] for p in group)
                    right=max(p[0]+p[2] for p in group);bottom=max(p[1]+p[3] for p in group)
                    if 2.3<=(right-x)/(bottom-y)<=4.0:
                        boxes.add((x,y,right,bottom))
                        if len(boxes)>128:return None
        return sorted(boxes)

    def normalize(self, rgb):
        boxes=self.proposals(rgb)
        if boxes is None:
            return NormalizedDetection(False,0,None,None,'normalize','safe_stop: proposal budget exhausted',129)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        hits=[];best=-1
        for box in boxes:
            x,y,right,bottom=box
            candidate=gray[y:bottom,x:right]
            for template,mask in self.profiles:
                # Reject strong aspect distortion before resizing the glyphs.
                distortion=(candidate.shape[1]/candidate.shape[0])/(template.shape[1]/template.shape[0])
                if not .85<=distortion<=1.18:continue
                normalized=cv2.resize(candidate,(template.shape[1],template.shape[0]),interpolation=cv2.INTER_AREA)
                score=float(finite_scores(normalized,template,mask)[0,0])
                best=max(best,score)
                if score>=self.threshold:hits.append((score,box))
        return self._finish(hits,best,'normalize',len(boxes))
