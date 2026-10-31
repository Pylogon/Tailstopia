"""Keyboard/mouse timing recorder for Windows. F8 start/pause/resume; F9 stop.
Records only while armed. Never automatically replays inputs.
Install: py -m pip install pynput
Run: py tailstopia_input_recorder.py
"""
from __future__ import annotations
import csv
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from pynput import keyboard, mouse

START_PAUSE_KEY = keyboard.Key.f8
STOP_KEY = keyboard.Key.f9
SAVE_DIR = Path(__file__).resolve().parent / "recordings"
RECORD_SCROLL = True
# For privacy, keyboard recordings may contain passwords/chat typed while recording.
# Start recording only inside the target application; pause before typing sensitive data.

lock = threading.RLock()
events: list[dict] = []
held_keys: dict[str, float] = {}
held_mouse: dict[str, tuple[float, int, int]] = {}
recording = False
stopped = False
f8_pressed = False
session_start = time.perf_counter()
segment_start = 0.0
active_elapsed = 0.0


def clock() -> float:
    return time.perf_counter()


def elapsed(now: float) -> float:
    return active_elapsed + (now - segment_start if recording else 0.0)


def key_name(key) -> str:
    try:
        if isinstance(key, keyboard.KeyCode):
            # Use virtual-key value for non-printables; char for printable keys.
            return key.char.lower() if key.char else f"vk_{key.vk}"
        return key.name
    except Exception:
        return str(key)


def log(kind: str, now: float, **fields):
    events.append({"t": round(elapsed(now), 4), "type": kind, **fields})


def release_all(now: float, reason: str):
    """Close any open holds when paused/stopped to avoid negative/spurious durations."""
    for name, began in list(held_keys.items()):
        log("key_hold", now, key=name, duration=round(now - began, 4), ended_by=reason)
    held_keys.clear()
    for name, (began, x, y) in list(held_mouse.items()):
        log("mouse_hold", now, button=name, x=x, y=y,
            duration=round(now - began, 4), ended_by=reason)
    held_mouse.clear()


def toggle():
    global recording, segment_start, active_elapsed
    with lock:
        now = clock()
        if not recording:
            segment_start = now
            recording = True
            print("[REC] Recording. F8=pause, F9=save & stop", flush=True)
        else:
            release_all(now, "pause")
            active_elapsed += now - segment_start
            recording = False
            print(f"[PAUSED] Active recorded time: {active_elapsed:.2f}s. F8=resume", flush=True)


def finish():
    global stopped, recording, active_elapsed
    with lock:
        if stopped:
            return
        now = clock()
        if recording:
            release_all(now, "stop")
            active_elapsed += now - segment_start
            recording = False
        stopped = True


def on_press(key):
    global f8_pressed
    if key == STOP_KEY:
        finish()
        return False
    if key == START_PAUSE_KEY:
        with lock:
            should_toggle = not f8_pressed
            f8_pressed = True
        if should_toggle:
            toggle()
        return
    with lock:
        if not recording or stopped:
            return
        name = key_name(key)
        if name not in held_keys:
            now = clock()
            held_keys[name] = now
            log("key_down", now, key=name)


def on_release(key):
    global f8_pressed
    if key == START_PAUSE_KEY:
        with lock:
            f8_pressed = False
        return
    if key == STOP_KEY:
        return
    with lock:
        name = key_name(key)
        if not recording or stopped:
            held_keys.pop(name, None)
            return
        now = clock()
        began = held_keys.pop(name, None)
        if began is not None:
            log("key_up", now, key=name)
            log("key_hold", now, key=name, duration=round(now - began, 4), ended_by="release")


def on_click(x, y, button, pressed):
    with lock:
        if not recording or stopped:
            return
        now = clock()
        name = button.name
        if pressed:
            if name not in held_mouse:
                held_mouse[name] = (now, int(x), int(y))
                log("mouse_down", now, button=name, x=int(x), y=int(y))
        else:
            start = held_mouse.pop(name, None)
            log("mouse_up", now, button=name, x=int(x), y=int(y))
            if start:
                began, sx, sy = start
                log("mouse_hold", now, button=name, x=sx, y=sy,
                    end_x=int(x), end_y=int(y),
                    duration=round(now - began, 4), ended_by="release")


def on_scroll(x, y, dx, dy):
    with lock:
        if RECORD_SCROLL and recording and not stopped:
            log("scroll", clock(), x=int(x), y=int(y), dx=dx, dy=dy)


def save():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = SAVE_DIR / f"input_{stamp}"
    with lock:
        data = list(events)
        total = round(active_elapsed, 4)
    metadata = {"created_at": datetime.now().isoformat(timespec="seconds"),
                "active_duration_seconds": total, "event_count": len(data),
                "events": data}
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = base.with_suffix(".csv")
    cols = ["t", "type", "key", "button", "x", "y", "end_x", "end_y", "duration", "dx", "dy", "ended_by"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(data)
    print(f"\nSaved {len(data)} events, active time {total:.2f}s")
    print("JSON:", json_path)
    print(" CSV:", csv_path)


def main():
    print("Input Recorder | F8=START/PAUSE/RESUME | F9=STOP & SAVE")
    print("WARNING: records keyboard input system-wide while active; do not type passwords.")
    print("Run as a normal user first. Some games with anti-cheat may not expose inputs.")
    k = keyboard.Listener(on_press=on_press, on_release=on_release)
    m = mouse.Listener(on_click=on_click, on_scroll=on_scroll)
    k.start()
    m.start()
    try:
        while not stopped:
            time.sleep(0.1)
            if not k.is_alive() or not m.is_alive():
                print("Input listener stopped unexpectedly; saving captured events.")
                finish()
                break
    except KeyboardInterrupt:
        finish()
    finally:
        k.stop()
        m.stop()
        save()


if __name__ == "__main__":
    main()
