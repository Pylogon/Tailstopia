"""Offline Talk experiments: saved RGB client images only, no input/capture API.

The live detector remains the fast path. Fallbacks inspect the *same* supplied
frame. A result is a proposal for offline evaluation, never permission to click.
"""
from dataclasses import dataclass
import cv2
import numpy as np

from tailstopia_workflow.mission_vision import MissionVision


@dataclass(frozen=True)
class Detection:
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


class TalkExperiment:
    """Precompute reference glyphs; bounded proposals and strict final threshold.

    Expanded ROI adds only 50px below the live ROI: a recorded Talk at y=484
    exceeded its old bottom=470. No live configuration is changed.
    """
    def __init__(self, config, base, roi=(280, 170, 1000, 520)):
        self.vision = MissionVision(config, base)
        self.roi = tuple(roi)
        self.spec = config['targets']['talk']
        self.threshold = self.spec['threshold']
        self.lower = np.array(self.spec['glyph_hsv_lower'], np.uint8)
        self.upper = np.array(self.spec['glyph_hsv_upper'], np.uint8)
        self.profiles = []
        # Scale=1 references only. No test screenshot is made into a template.
        stride = len(self.spec['template_scales'])
        for gray, mask in zip(self.vision.targets['talk'][0][::stride],
                              self.vision.target_masks['talk'][::stride]):
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            parts = stats[1:n]
            h = max(parts[:, 3])
            parts = [p for p in parts if p[3] >= h * .5 and p[4] >= 4]
            if not parts:
                continue
            x = min(p[0] for p in parts); y = min(p[1] for p in parts)
            right = max(p[0]+p[2] for p in parts); bottom = max(p[1]+p[3] for p in parts)
            self.profiles.append((gray[y:bottom, x:right].copy(), mask[y:bottom, x:right].copy()))
        self.coarse_bank = []
        for index, (gray, mask) in enumerate(self.profiles):
            for scale in np.arange(.7, 2.01, .2):
                w = max(3, round(gray.shape[1]*scale*.5))
                h = max(3, round(gray.shape[0]*scale*.5))
                self.coarse_bank.append((index, float(scale),
                    cv2.resize(gray, (w,h), interpolation=cv2.INTER_AREA),
                    cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)))

    @property
    def cache_bytes(self):
        return sum(a.nbytes+b.nbytes for a,b in self.profiles) + sum(
            a.nbytes+b.nbytes for _,_,a,b in self.coarse_bank)

    def _frame(self, client):
        if client.mode != 'RGB' or client.size != (1280,600):
            raise ValueError('Expected a saved 1280x600 RGB client image')
        return np.array(client.crop(self.roi))

    def baseline(self, client, expanded=False):
        if client.mode != 'RGB' or client.size != (1280,600):
            raise ValueError('Expected a saved 1280x600 RGB client image')
        if not expanded:
            match = self.vision.match('talk', lambda box:client.crop(box))
        else:
            # Separate control arm: changed ROI, same template bank and score.
            templates, _, threshold = self.vision.targets['talk']
            gray = np.array(client.crop(self.roi).convert('L'))
            best = (-1, None)
            for template,mask in zip(templates,self.vision.target_masks['talk']):
                _,score,_,point=cv2.minMaxLoc(finite_scores(gray,template,mask))
                if score > best[0]:
                    best=(score,(self.roi[0]+point[0]+template.shape[1]//2,
                                 self.roi[1]+point[1]+template.shape[0]//2))
            from tailstopia_workflow.mission_vision import Match
            match=Match(best[0]>=threshold,best[0],best[1] if best[0]>=threshold else None)
        return Detection(match.found,match.score,match.point,None,
                         'legacy_expanded' if expanded else 'fast',
                         'matched' if match.found else 'safe_stop: no verified Talk')

    def detect(self, client, method='normalize'):
        if method not in ('normalize','coarse_to_fine'):
            raise ValueError('Unknown offline fallback')
        fast = self.baseline(client)
        if fast.found:
            return fast
        rgb = self._frame(client)
        return self.normalize(rgb) if method == 'normalize' else self.coarse_to_fine(rgb)

    def _finish(self, hits, score, path, count):
        # All passing proposals must describe the same word. Separate matching
        # labels are ambiguous: never select an arbitrary highest-scoring one.
        hits.sort(reverse=True)
        if not hits:
            return Detection(False,float(score),None,None,path,'safe_stop: no verified Talk',count)
        best_score, box = hits[0]
        center = ((box[0]+box[2])//2,(box[1]+box[3])//2)
        for _, other in hits[1:]:
            point=((other[0]+other[2])//2,(other[1]+other[3])//2)
            if max(abs(point[0]-center[0]),abs(point[1]-center[1])) > max(10,(box[3]-box[1])*.6):
                return Detection(False,float(best_score),None,None,path,'safe_stop: ambiguous labels',count)
        x,y=self.roi[:2]
        return Detection(True,float(best_score),(center[0]+x,center[1]+y),
                         (box[0]+x,box[1]+y,box[2]+x,box[3]+y),path,'verified offline proposal',count)

    def proposals(self, rgb):
        mask=cv2.inRange(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV),self.lower,self.upper)
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        parts=[tuple(map(int,p)) for p in stats[1:] if 5<=p[3]<=40 and
               2<=p[2]<=p[3]*1.5 and p[4]>=5]
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
            return Detection(False,0,None,None,'normalize','safe_stop: proposal budget exhausted',129)
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

    def coarse_to_fine(self, rgb):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        small=cv2.resize(gray,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
        proposals=[]
        for index,scale,template,mask in self.coarse_bank:
            scores=finite_scores(small,template,mask)
            for _ in range(2):
                _,score,_,point=cv2.minMaxLoc(scores)
                if score<.45:break  # proposal only; final acceptance is still .82
                proposals.append((score,index,scale,point[0]*2,point[1]*2))
                x,y=point
                scores[max(0,y-5):y+6,max(0,x-10):x+11]=-1
        proposals.sort(reverse=True)
        selected=[]
        for p in proposals:
            if any(p[1]==q[1] and abs(p[3]-q[3])<8 and abs(p[4]-q[4])<5 for q in selected):continue
            selected.append(p)
            if len(selected)==8:break
        hits=[];best=-1
        for _,index,scale,x,y in selected:
            source,source_mask=self.profiles[index]
            for fine in np.arange(max(.5,scale-.1),scale+.101,.025):
                w=round(source.shape[1]*fine);h=round(source.shape[0]*fine)
                template=cv2.resize(source,(w,h),interpolation=cv2.INTER_AREA)
                mask=cv2.resize(source_mask,(w,h),interpolation=cv2.INTER_NEAREST)
                left=max(0,x-4);top=max(0,y-4)
                area=gray[top:min(gray.shape[0],y+h+5),left:min(gray.shape[1],x+w+5)]
                if area.shape[0]<h or area.shape[1]<w:continue
                _,score,_,point=cv2.minMaxLoc(finite_scores(area,template,mask))
                best=max(best,score)
                if score>=self.threshold:
                    xx=left+point[0];yy=top+point[1]
                    hits.append((score,(xx,yy,xx+w,yy+h)))
        return self._finish(hits,best,'coarse_to_fine',len(selected))
