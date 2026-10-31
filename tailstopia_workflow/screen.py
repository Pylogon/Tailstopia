import struct
import zlib
from pathlib import Path


class MockScreen:
    def __init__(self, script=None, rewards=None):
        self.script = {key: list(values) for key, values in (script or {}).items()}
        self.rewards = rewards or []

    def check(self, name):
        values = self.script.get(name, [True])
        value = values.pop(0) if len(values) > 1 else values[0]
        if isinstance(value, str):
            raise ValueError(value)
        return bool(value)

    def scan(self):
        return self.rewards

    def save_debug(self, path):
        # Synthetic 1x1 PNG, explicitly not a desktop screenshot.
        def chunk(kind, payload):
            return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                               + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b""))


class RecordedScreen:
    """Offline fixed ROI comparison; Pillow only, no OCR or live capture."""
    def __init__(self, frames, checks, rewards=None):
        self.frames = {name: [Path(p) for p in paths] for name, paths in frames.items()}
        self.checks = checks
        self.rewards = rewards or []
        self.last_frame = None

    def _frame(self, name):
        paths = self.frames[name]
        self.last_frame = paths.pop(0) if len(paths) > 1 else paths[0]
        return self.last_frame

    @staticmethod
    def matches(frame, spec):
        from PIL import Image, ImageChops, ImageStat
        with Image.open(frame) as source, Image.open(spec["template"]) as reference:
            roi = spec["roi"]
            if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= source.width and 0 <= roi[1] < roi[3] <= source.height):
                raise ValueError("ROI outside recorded frame")
            crop, template = source.crop(roi).convert("RGB"), reference.convert("RGB")
            if crop.size != template.size:
                raise ValueError("ROI/template size mismatch")
            threshold = float(spec.get("max_error", 0.02))
            if not 0 <= threshold <= 1:
                raise ValueError("max_error must be in [0,1]")
            return sum(ImageStat.Stat(ImageChops.difference(crop, template)).mean) / (3 * 255) <= threshold

    def check(self, name):
        return self.matches(self._frame(name), self.checks[name])

    def scan(self):
        frame = self._frame("rewards")
        return [{"name": item["name"], "x": item["x"], "y": item["y"]}
                for item in self.rewards if self.matches(frame, item)]

    def save_debug(self, path):
        from PIL import Image
        if self.last_frame is None:
            raise ValueError("No recorded frame available")
        with Image.open(self.last_frame) as image:
            image.save(path, format="PNG")
