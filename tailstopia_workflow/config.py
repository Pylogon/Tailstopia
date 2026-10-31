import json
import math
from pathlib import Path

STATES = ("VERIFY_START", "PLAY_MACRO_A", "ENTER_STAGE", "VERIFY_STAGE", "PLAY_MACRO_B",
          "VERIFY_COMPLETE", "SCAN_REWARD", "SELECT_REWARD", "RETURN_START")


def number(value, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Expected finite " + ("positive" if positive else "nonnegative") + " number")


def coordinate(point):
    for key in ("x", "y"):
        value = point[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Coordinates must be nonnegative integers")


def validate(config):
    loops = config["max_loops"]
    if isinstance(loops, bool) or not isinstance(loops, int) or loops < 1:
        raise ValueError("max_loops must be a positive integer")
    for state in STATES:
        policy = config["states"][state]
        number(policy["timeout"], positive=True)
        retries = policy["retries"]
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("retries must be a nonnegative integer")
        number(policy["retry_delay"])
    for name in ("enter_stage", "return_start"):
        for action in config[name]:
            if action["type"] == "click":
                coordinate(action)
            elif action["type"] != "wait":
                raise ValueError("Config actions support click/wait only")
            number(action.get("wait", 0))
    if len([a for a in config["enter_stage"] if a["type"] == "click"]) != 4:
        raise ValueError("enter_stage requires four configured clicks")
    if config["reward"]["fallback"] is not None:
        coordinate(config["reward"]["fallback"])
    if config.get("screen", {}).get("backend", "mock") not in {"mock", "recorded"}:
        raise ValueError("Only mock/recorded screens supported")
    return config


def load_config(path):
    path = Path(path).resolve()
    config = validate(json.loads(path.read_text(encoding="utf-8-sig")))
    base = path.parent
    for key in ("macro_a", "macro_b", "output"):
        config[key] = str((base / config[key]).resolve())
    screen = config.get("screen", {})
    for key, paths in screen.get("frames", {}).items():
        screen["frames"][key] = [str((base / p).resolve()) for p in paths]
    for spec in list(screen.get("checks", {}).values()) + screen.get("rewards", []):
        if "template" in spec:
            spec["template"] = str((base / spec["template"]).resolve())
    return config
