"""Separate offline profile for the larger TalesofTails stat font."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from offline.stat_templates import Policy, Prediction, feature, rank, split_digits


@dataclass(frozen=True)
class Sample:
    name: str
    value: int
    source: str
    split: str
    pixel_hash: str
    image: Image.Image


def load_samples(folder):
    result = []
    for row in json.loads((Path(folder)/"manifest.json").read_text())["cases"]:
        with Image.open(Path(folder)/row["file"]) as image:
            image = image.convert("RGB").copy()
        digest = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
        result.append(Sample(row["file"], row["value"], row["source"], row["split"], digest, image))
    return result


def native_ink(image):
    gray = np.asarray(image.convert("L"))
    if gray.shape != (18, 24):
        return None
    mask = (gray <= 140).astype(np.uint8)
    if not mask.any():
        return None
    x, y, w, h = cv2.boundingRect(mask)
    # Development glyphs are 11px high, versus 7-8px in the original game.
    if x == 0 or y == 0 or x+w == 24 or y+h == 18 or not (5 <= h <= 14 and 1 <= w <= 20):
        return None
    if not 5 <= int(mask.sum()) <= 180:
        return None
    return mask[y:y+h, x:x+w].copy()


class NativeTemplates:
    def __init__(self, samples, kind, policy):
        if kind not in ("whole", "digit") or any(s.split != "development" for s in samples):
            raise ValueError("Only development samples may build templates")
        self.kind, self.policy = kind, policy
        self.bank, self.training_failures = [], []
        self.training_hashes = frozenset(s.pixel_hash for s in samples)
        seen = set()
        for sample in samples:
            mask = native_ink(sample.image)
            pieces = split_digits(mask) if kind == "digit" and mask is not None else [mask]
            labels = list(map(int, str(sample.value))) if kind == "digit" else [sample.value]
            if not pieces or len(pieces) != len(labels) or any(p is None for p in pieces):
                self.training_failures.append(sample.name)
                continue
            for value, piece in zip(labels, pieces):
                vector, ratio = feature(piece), piece.shape[1]/piece.shape[0]
                key = (value, ratio, vector.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                vector.setflags(write=False)
                self.bank.append((value, vector, ratio))
        self.supported = sorted({value for value, _, _ in self.bank})

    @property
    def template_bytes(self):
        return sum(vector.nbytes for _, vector, _ in self.bank)

    def read(self, image):
        mask = native_ink(image)
        if mask is None:
            return Prediction(None, -1, 0, "UNKNOWN: geometry/ink")
        pieces = split_digits(mask) if self.kind == "digit" else [mask]
        if not pieces or len(pieces) > 2:
            return Prediction(None, -1, 0, "UNKNOWN: segmentation")
        labels, scores, margins = [], [], []
        for piece in pieces:
            label, score, margin = rank(piece, self.bank)
            if label is None or score < self.policy.threshold or margin < self.policy.margin:
                return Prediction(None, score, margin, "UNKNOWN: score/ambiguity")
            labels.append(label)
            scores.append(score)
            margins.append(margin)
        if self.kind == "digit" and len(labels) > 1 and labels[0] == 0:
            return Prediction(None, min(scores), min(margins), "UNKNOWN: leading zero")
        value = int("".join(map(str, labels))) if self.kind == "digit" else labels[0]
        if not 0 <= value <= 30:
            return Prediction(None, min(scores), min(margins), "UNKNOWN: range")
        return Prediction(value, min(scores), min(margins), "accepted")


def calibrate_sources(samples, kind):
    if any(s.split != "development" for s in samples):
        raise ValueError("Evaluation samples cannot select thresholds")
    grid = []
    for threshold in (.80, .85, .90, .93, .95, .97, .99):
        for margin in (.03, .05, .08, .12):
            policy = Policy(threshold, margin)
            correct = wrong = unknown = 0
            for source in sorted({s.source for s in samples}):
                queries = [s for s in samples if s.source == source]
                hashes = {s.pixel_hash for s in queries}
                train = [s for s in samples if s.source != source and s.pixel_hash not in hashes]
                model = NativeTemplates(train, kind, policy)
                for sample in queries:
                    value = model.read(sample.image).value
                    correct += value == sample.value
                    wrong += value is not None and value != sample.value
                    unknown += value is None
            grid.append(dict(threshold=threshold, margin=margin, correct=correct, wrong=wrong, unknown=unknown))
    safe = [r for r in grid if not r["wrong"]]
    if not safe:
        raise ValueError("No development policy without false reads")
    selected = max(safe, key=lambda r: (r["correct"], r["threshold"], r["margin"]))
    return selected, grid
