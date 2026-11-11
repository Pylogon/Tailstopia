"""Offline-only stat recognition. No capture, input, or production imports."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ALGORITHM_VERSION = 1
DEV_SOURCE = "historical_local"
HOLDOUT_SOURCE = "failure_20261104"


@dataclass(frozen=True)
class Case:
    name: str
    value: int
    source: str
    pixel_hash: str
    image: Image.Image


@dataclass(frozen=True)
class Prediction:
    value: int | None
    score: float
    margin: float
    reason: str


@dataclass(frozen=True)
class Policy:
    threshold: float
    margin: float


def load_corpus(root):
    folder = Path(root) / "tests/fixtures/reroll_ocr"
    cases = []
    for row in json.loads((folder / "manifest.json").read_text()):
        with Image.open(folder / row["file"]) as image:
            image = image.convert("RGB").copy()
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        cases.append(Case(row["file"], row["value"], row["source"], digest, image))
    development = [c for c in cases if c.source == DEV_SOURCE]
    if not development:
        raise ValueError("Development source missing")
    trained = {c.pixel_hash for c in development}
    excluded = [c for c in cases if c.source == HOLDOUT_SOURCE and c.pixel_hash in trained]
    evaluation = [c for c in cases if c.source == HOLDOUT_SOURCE and c.pixel_hash not in trained]
    if len({c.pixel_hash for c in evaluation}) != len(evaluation):
        raise ValueError("Duplicate evaluation pixels")
    return development, evaluation, excluded


def corpus_signature(cases):
    rows = sorted((c.name, c.value, c.source, c.pixel_hash) for c in cases)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def ink(image):
    """Fixed 24x18 saved ROI; detect dark strokes using the existing 140 cutoff."""
    gray = np.asarray(image.convert("L"))
    if gray.shape != (18, 24):
        return None
    mask = (gray <= 140).astype(np.uint8)
    if not mask.any():
        return None
    x, y, w, h = cv2.boundingRect(mask)
    # Reject clipped UI, empty boxes and shapes outside development geometry.
    if x == 0 or y == 0 or x+w == 24 or y+h == 18 or not (5 <= h <= 10 and 1 <= w <= 16):
        return None
    if not 5 <= int(mask.sum()) <= 100:
        return None
    return mask[y:y+h, x:x+w].copy()


def tight(mask):
    if not mask.any():
        return None
    x, y, w, h = cv2.boundingRect(mask)
    return mask[y:y+h, x:x+w].copy()


def split_digits(mask):
    """Weak column valleys also separate the joined, antialiased reference 12."""
    columns = mask.sum(axis=0)
    strong = columns >= 2
    indices = np.flatnonzero(strong)
    if not len(indices):
        return None
    breaks = np.flatnonzero(np.diff(indices) > 1)
    if len(breaks) > 1:
        return None
    if not len(breaks):
        pieces = [mask]
    else:
        before = int(indices[breaks[0]])
        after = int(indices[breaks[0]+1])
        cut = (before + after + 1) // 2
        pieces = [tight(mask[:, :cut]), tight(mask[:, cut:])]
    if any(p is None or p.shape[0] < 5 or int(p.sum()) < 5 for p in pieces):
        return None
    return pieces


def feature(mask):
    return cv2.resize(mask.astype(np.float32), (16, 24), interpolation=cv2.INTER_LINEAR)


def rank(mask, bank):
    query = feature(mask)
    aspect = mask.shape[1] / mask.shape[0]
    scores = {}
    for value, reference, ratio in bank:
        if abs(aspect - ratio) > .22:
            continue
        score = float(cv2.matchTemplate(query, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
        if np.isfinite(score) and -1 <= score <= 1.0001:
            scores[value] = max(score, scores.get(value, -1.0))
    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    if not ordered:
        return None, -1.0, 0.0
    value, score = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else -1.0
    return value, min(score, 1.0), score-runner_up


class Templates:
    def __init__(self, cases, kind, policy):
        if kind not in ("whole", "digit"):
            raise ValueError("Unknown template kind")
        if not 0 < policy.threshold <= 1 or not policy.margin > 0:
            raise ValueError("Invalid acceptance policy")
        if any(c.source != DEV_SOURCE for c in cases):
            raise ValueError("Only development images may create templates")
        self.kind, self.policy = kind, policy
        self.bank = []
        self.training_hashes = frozenset(c.pixel_hash for c in cases)
        self.training_failures = []
        for case in cases:
            mask = ink(case.image)
            pieces = split_digits(mask) if mask is not None and kind == "digit" else [mask]
            labels = list(map(int, str(case.value))) if kind == "digit" else [case.value]
            if pieces is None or len(pieces) != len(labels) or any(p is None for p in pieces):
                self.training_failures.append(case.name)
                continue
            for label, piece in zip(labels, pieces):
                f = feature(piece)
                f.setflags(write=False)
                self.bank.append((label, f, piece.shape[1] / piece.shape[0]))
        self.supported = sorted({entry[0] for entry in self.bank})

    @property
    def template_bytes(self):
        return sum(entry[1].nbytes for entry in self.bank)

    def read(self, image):
        mask = ink(image)
        if mask is None:
            return Prediction(None, -1.0, 0, "UNKNOWN: geometry/ink")
        pieces = split_digits(mask) if self.kind == "digit" else [mask]
        if not pieces or len(pieces) > 2:
            return Prediction(None, -1.0, 0, "UNKNOWN: ambiguous segmentation")
        labels, scores, margins = [], [], []
        for piece in pieces:
            label, score, margin = rank(piece, self.bank)
            if label is None or score < self.policy.threshold or margin < self.policy.margin:
                return Prediction(None, score, margin, "UNKNOWN: score or competing label")
            labels.append(label)
            scores.append(score)
            margins.append(margin)
        if self.kind == "digit" and len(labels) > 1 and labels[0] == 0:
            return Prediction(None, min(scores), min(margins), "UNKNOWN: leading zero")
        value = int("".join(map(str, labels))) if self.kind == "digit" else labels[0]
        if not 0 <= value <= 30:
            return Prediction(None, min(scores), min(margins), "UNKNOWN: outside current OCR range")
        return Prediction(value, min(scores), min(margins), "accepted")


def calibrate(development, kind):
    """Leave-one-image-out on development only; never inspect evaluation cases."""
    if any(c.source != DEV_SOURCE for c in development):
        raise ValueError("Calibration cannot use evaluation sources")
    candidates = []
    for threshold in (.80, .85, .90, .93, .95, .97, .99):
        for margin in (.03, .05, .08, .12):
            policy = Policy(threshold, margin)
            correct = wrong = unknown = 0
            for held in development:
                # Also remove pixel duplicates; label-level impostors remain.
                train = [c for c in development if c.pixel_hash != held.pixel_hash]
                value = Templates(train, kind, policy).read(held.image).value
                correct += value == held.value
                wrong += value is not None and value != held.value
                unknown += value is None
            candidates.append(dict(threshold=threshold, margin=margin,
                                   correct=correct, wrong=wrong, unknown=unknown))
    safe = [row for row in candidates if row["wrong"] == 0]
    if not safe:
        raise ValueError("No development policy avoids false reads")
    selected = max(safe, key=lambda r: (r["correct"], r["threshold"], r["margin"]))
    return Policy(selected["threshold"], selected["margin"]), selected, candidates
