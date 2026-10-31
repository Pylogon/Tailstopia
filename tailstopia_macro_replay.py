"""Simple recording replay based on the working test.py.
F8 = focus game and replay once. F9 = release all held inputs and exit.
py tailstopia_macro_replay.py [recording.json]
py tailstopia_macro_replay.py --dry-run
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import math
from pathlib import Path
import time

from tailstopia_workflow.macro_parser import load_macro

GAME_TITLE_PART = "TailsTopiaOnline"
DEFAULT_RECORDING = Path(__file__).resolve().parent / "recordings/start.json"


class StopRequested(RuntimeError):
    pass


def recording(path):
    path = Path(path)
    actions = load_macro(path)
    duration = actions[-1]["t"] if actions else 0.0
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            duration = float(data.get("active_duration_seconds", duration))
    if not math.isfinite(duration) or duration < (actions[-1]["t"] if actions else 0):
        raise ValueError("Invalid recording duration")
    return actions, duration


def validate_inputs(actions, api):
    aliases = {"ctrl_l":"ctrlleft", "ctrl_r":"ctrlright", "shift_l":"shiftleft",
               "shift_r":"shiftright", "alt_l":"altleft", "alt_r":"altright",
               "cmd":"win", "cmd_l":"winleft", "cmd_r":"winright",
               "caps_lock":"capslock", "page_up":"pageup", "page_down":"pagedown",
               "num_lock":"numlock", "scroll_lock":"scrolllock", "print_screen":"printscreen"}
    result = []
    for original in actions:
        action = dict(original)
        if action["type"].startswith("key_"):
            key = aliases.get(action["key"], action["key"])
            if key in {"f8", "f9"} or key not in api.KEYBOARD_KEYS:
                raise ValueError(f"Unsupported/reserved key: {key}")
            action["key"] = key
        elif action["type"].startswith("mouse_") and action["button"] not in {"left","right","middle"}:
            raise ValueError("Unsupported mouse button: " + action["button"])
        result.append(action)
    return result


def replay(actions, duration, api, check, clock=time.perf_counter, sleep=time.sleep, dry_run=False, on_action=None):
    """Down stays held until its matching up. Summary rows are removed by parser."""
    held_keys, held_buttons = set(), set()
    started = clock()

    def wait_until(offset):
        while True:
            check()
            if not dry_run:
                api.failSafeCheck()
            remaining = started + offset - clock()
            if remaining <= 0:
                return
            sleep(min(remaining, 0.01))

    try:
        for action in actions:
            wait_until(action["t"])
            kind = action["type"]
            dispatched = clock()-started
            print(f"{dispatched:7.4f}s {action}", flush=True)
            if dry_run:
                if on_action is not None:
                    on_action(action,dispatched)
                continue
            if kind == "key_down":
                held_keys.add(action["key"])
                api.keyDown(action["key"], _pause=False)
            elif kind == "key_up":
                api.keyUp(action["key"], _pause=False)
                held_keys.discard(action["key"])
            elif kind == "mouse_down":
                held_buttons.add(action["button"])
                api.mouseDown(x=action["x"], y=action["y"], button=action["button"], _pause=False)
            elif kind == "mouse_up":
                api.mouseUp(x=action["x"], y=action["y"], button=action["button"], _pause=False)
                held_buttons.discard(action["button"])
            elif kind == "scroll":
                api.moveTo(action["x"], action["y"], _pause=False)
                if action["dy"]:
                    api.scroll(action["dy"], _pause=False)
                if action["dx"]:
                    api.hscroll(action["dx"], _pause=False)
            if on_action is not None:
                on_action(action,dispatched)
        wait_until(duration)
    finally:
        if not dry_run:
            enabled = api.FAILSAFE
            api.FAILSAFE = False
            errors = []
            try:
                for key in held_keys:
                    try:
                        api.keyUp(key, _pause=False)
                    except Exception as exc:
                        errors.append(str(exc))
                for button in held_buttons:
                    try:
                        api.mouseUp(button=button, _pause=False)
                    except Exception as exc:
                        errors.append(str(exc))
            finally:
                api.FAILSAFE = enabled
            if errors:
                raise RuntimeError("Release failed: " + "; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", nargs="?", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    actions, duration = recording(args.recording)
    print(f"Loaded {len(actions)} actions; {duration:.4f}s. Hold summaries are not replayed twice.")
    if args.dry_run:
        replay(actions, duration, None, lambda: None, dry_run=True)
        print("Dry-run complete. No inputs sent.")
        return
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    signatures = {
        "EnumWindows": ([callback_type, wintypes.LPARAM], wintypes.BOOL),
        "GetForegroundWindow": ([], wintypes.HWND),
        "GetWindowTextLengthW": ([wintypes.HWND], ctypes.c_int),
        "GetWindowTextW": ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
        "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
        "IsIconic": ([wintypes.HWND], wintypes.BOOL),
        "ShowWindow": ([wintypes.HWND, ctypes.c_int], wintypes.BOOL),
        "SetForegroundWindow": ([wintypes.HWND], wintypes.BOOL),
        "GetAsyncKeyState": ([ctypes.c_int], ctypes.c_short),
    }
    for name, (args, result) in signatures.items():
        function = getattr(user32, name)
        function.argtypes, function.restype = args, result

    def find_game():
        matches = []
        @callback_type
        def visit(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)
                if GAME_TITLE_PART.casefold() in title.value.casefold():
                    matches.append(hwnd)
            return True
        user32.EnumWindows(visit, 0)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {GAME_TITLE_PART} window; found {len(matches)}")
        return matches[0]

    def pressed(vk):
        return bool(user32.GetAsyncKeyState(vk) & 0x8000)


    actions = validate_inputs(actions, pyautogui)
    was_f8 = pressed(0x77)
    print("F8 = focus game and replay ONCE | F9 = release inputs and EXIT", flush=True)
    print("Uses ORIGINAL desktop coordinates. Keep the game in its recorded position.", flush=True)
    try:
        while not pressed(0x78):
            f8 = pressed(0x77)
            if f8 and not was_f8:
                try:
                    hwnd = find_game()
                    if user32.IsIconic(hwnd):
                        user32.ShowWindow(hwnd, 9)
                    user32.SetForegroundWindow(hwnd)
                    for _ in range(100):
                        if pressed(0x78) or user32.GetForegroundWindow() == hwnd:
                            break
                        time.sleep(0.01)
                    if pressed(0x78):
                        break
                    if user32.GetForegroundWindow() != hwnd:
                        raise RuntimeError("Cannot focus game. Click the game and press F8 again.")

                    def check():
                        if pressed(0x78):
                            raise StopRequested("F9 pressed")
                        if user32.GetForegroundWindow() != hwnd:
                            raise RuntimeError("Game lost focus")

                    from tailstopia_workflow.game_window import require_english_keyboard
                    language = require_english_keyboard(user32,hwnd)
                    print(f"Keyboard EN verified (0x{language:04X}).",flush=True)
                    print("Game focused. Playing recording...", flush=True)
                    replay(actions, duration, pyautogui, check)
                    print("Complete. F8 = replay again | F9 = exit", flush=True)
                except StopRequested:
                    break
                except Exception as exc:
                    print(f"STOP: {type(exc).__name__}: {exc}", flush=True)
                f8 = pressed(0x77)
            was_f8 = f8
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Ctrl+C: stopped")
    print("Stopped. Replay inputs released.", flush=True)


if __name__ == "__main__":
    main()
