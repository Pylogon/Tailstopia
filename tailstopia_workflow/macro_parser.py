import csv
import json
import math
from pathlib import Path


def load_macro(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        data = list(csv.DictReader(stream)) if path.suffix.lower() == ".csv" else json.load(stream)
    rows = data["events"] if isinstance(data, dict) else data
    result, held = [], set()
    previous = 0.0
    for source in rows:
        row = {key: value for key, value in source.items() if value != "" and value is not None}
        t = float(row["t"])
        if not math.isfinite(t) or t < previous:
            raise ValueError("Timestamps must be finite, nonnegative and ordered")
        previous = row["t"] = t
        kind = row["type"]
        if kind not in {"key_down", "key_up", "key_hold", "mouse_down", "mouse_up", "mouse_hold", "scroll"}:
            raise ValueError("Unsupported event: " + kind)
        family = "key" if kind.startswith("key_") else "mouse"
        token = (family, row.get("key") if family == "key" else row.get("button"))
        if kind.endswith("_hold"):
            if row.get("ended_by") not in {"pause", "stop", "release"}:
                raise ValueError("Hold-only recordings unsupported; ended_by required")
            if token in held and row["ended_by"] in {"pause", "stop"}:
                held.remove(token)
                result.append({**row, "type": family + "_up"})
            continue
        if kind != "scroll" and not token[1]:
            raise ValueError("Missing key/button")
        if family == "mouse":
            for field in (("x", "y", "dx", "dy") if kind == "scroll" else ("x", "y")):
                value = float(row[field])
                if not math.isfinite(value) or not value.is_integer():
                    raise ValueError("Invalid coordinate/scroll")
                row[field] = int(value)
        if kind.endswith("_down"):
            if token in held:
                raise ValueError("Duplicate down")
            held.add(token)
        elif kind.endswith("_up"):
            if token not in held:
                raise ValueError("Unmatched up")
            held.remove(token)
        result.append(row)
    if held:
        raise ValueError("Recording ends with unreleased inputs")
    return result
