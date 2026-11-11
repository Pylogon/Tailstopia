"""Experimental TalesofTails reroll. F8 starts; F9 exits. Never clicks Okay.

Run from the repository: python talesoftails_auto_reroll.py
Requires the 1280x600 client/font used by resource/stat_new; moving the window is OK.
Settings below are independent of the original TailsTopia reroll and mission flow.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from offline.new_stat_templates import NativeTemplates, load_samples
from offline.stat_templates import Policy
from tailstopia_macro_replay import StopRequested


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    window_title: str = "TalesofTails"
    client_size: tuple = (1280, 600)
    random_xy: tuple = (695, 367)  # Client coordinates, not desktop coordinates.
    result_wait: float = 2.75
    confirmation_wait: float = 0.20
    read_attempts: int = 4  # Re-read only; UNKNOWN never triggers another click.
    match_threshold: float = 0.96  # Live profile; frozen offline benchmark stays 0.97.
    match_margin: float = 0.12
    agi_cha_min: int = 18
    cha_min: int = 9
    max_rolls: int = 0  # 0 = until success or F9.


SETTINGS = Settings()
PANEL_BOX = (590, 200, 745, 380)
STAT_BOXES = {"AGI": (5, 47, 29, 65), "CHA": (5, 107, 29, 125)}
BUTTON_BOX = (67, 162, 143, 172)


def target_met(values, settings=SETTINGS):
    if any(type(values.get(k)) is not int for k in STAT_BOXES):
        return False
    pair = values["AGI"] + values["CHA"]
    return pair >= settings.agi_cha_min and values["CHA"] >= settings.cha_min


def select_game_window(candidates, foreground, settings=SETTINGS):
    """Prefer the focused game, never choose arbitrarily between game clients."""
    eligible = [c for c in candidates if c["size"] == settings.client_size
                and settings.window_title.casefold() in c["title"].casefold()]
    focused = [c for c in eligible if c["hwnd"] == foreground]
    if len(focused) == 1:
        return focused[0]["hwnd"]
    exact = [c for c in eligible if c["title"].casefold() == settings.window_title.casefold()]
    pool = exact or eligible
    if len(pool) != 1:
        raise RuntimeError(f"Cannot select one game window: {candidates}. "
                           "Focus the intended 1280x600 game and press F8 again.")
    return pool[0]["hwnd"]


def activate_game(window, settings=SETTINGS):
    import ctypes
    from ctypes import wintypes

    api = window.api
    foreground = api.GetForegroundWindow()
    candidates = []
    @window.callback_type
    def visit(hwnd, _):
        if api.IsWindowVisible(hwnd):
            text = ctypes.create_unicode_buffer(api.GetWindowTextLengthW(hwnd) + 1)
            api.GetWindowTextW(hwnd, text, len(text))
            if settings.window_title.casefold() in text.value.casefold():
                rect = wintypes.RECT()
                if api.GetClientRect(hwnd, ctypes.byref(rect)):
                    candidates.append(dict(hwnd=hwnd, title=text.value,
                                           size=(rect.right, rect.bottom)))
        return True
    api.EnumWindows(visit, 0)
    window.hwnd = select_game_window(candidates, foreground, settings)
    if api.IsIconic(window.hwnd):
        api.ShowWindow(window.hwnd, 9)
    if foreground != window.hwnd:
        api.SetForegroundWindow(window.hwnd)
        for _ in range(100):
            if window.pressed(0x78) or api.GetForegroundWindow() == window.hwnd:
                break
            time.sleep(.01)
    window.check()


class Reader:
    def __init__(self, root=ROOT, settings=SETTINGS):
        development = [s for s in load_samples(root / "tests/fixtures/stat_new")
                       if s.split == "development"]
        # Supplemental, manually labelled live failure. Never train on evaluation crops.
        development += load_samples(root / "tests/fixtures/stat_new_live")
        self.model = NativeTemplates(development, "whole",
                                     Policy(settings.match_threshold, settings.match_margin))
        if self.model.training_failures or not self.model.bank:
            raise RuntimeError("Stat template bank is incomplete")
        self.buttons = []
        for name in ("random_stats_normal.png", "random_stats_hover.png"):
            with Image.open(root / "tests/fixtures/stat_new" / name) as im:
                self.buttons.append(np.asarray(im.convert("L")).copy())

    def check_panel(self, image):
        if image.size != (155, 180):
            raise RuntimeError("Unexpected character panel size")
        button = np.asarray(image.crop(BUTTON_BOX).convert("L"))
        score = max(float(cv2.matchTemplate(button, ref, cv2.TM_CCOEFF_NORMED)[0, 0])
                    for ref in self.buttons)
        if not np.isfinite(score) or score < .98:
            raise RuntimeError(f"RANDOM STATS screen not verified: score={score:.6f}")
        return score

    def read(self, image):
        self.check_panel(image)
        predictions = {name: self.model.read(image.crop(box)) for name, box in STAT_BOXES.items()}
        return ({name: p.value for name, p in predictions.items()},
                {name: asdict(p) for name, p in predictions.items()})


def wait_checked(seconds, check, clock=time.perf_counter, sleep=time.sleep):
    deadline = clock() + seconds
    while True:
        check()
        remaining = deadline - clock()
        if remaining <= 0:
            return
        sleep(min(.025, remaining))


def run(io, reader, log, settings=SETTINGS):
    """Injected IO keeps all tests offline. Only stable, known reads allow reroll."""
    if settings.result_wait < 2.75 or settings.confirmation_wait <= 0 or settings.read_attempts < 2:
        raise ValueError("Require >=2.75s result wait and at least two fresh reads")
    roll = 0
    while not settings.max_rolls or roll < settings.max_rolls:
        io.check()
        reader.check_panel(io.capture())
        io.click_random()
        roll += 1
        log("roll", roll=roll)
        io.wait(settings.result_wait)
        previous = None
        for attempt in range(1, settings.read_attempts + 1):
            io.check()
            values, predictions = reader.read(io.capture())
            io.check()
            log("read", roll=roll, attempt=attempt, values=values, predictions=predictions)
            known = all(type(values.get(k)) is int for k in STAT_BOXES)
            if known and values == previous:
                pair = values["AGI"] + values["CHA"]
                log("verified", roll=roll, values=values, agi_cha=pair,
                    matched=target_met(values, settings))
                if target_met(values, settings):
                    return values
                break
            previous = values.copy() if known else None
            if attempt == settings.read_attempts:
                raise RuntimeError("UNKNOWN or unstable stats; stopped without another reroll")
            io.wait(settings.confirmation_wait)
    raise RuntimeError("Roll limit reached")


class LiveIO:
    def __init__(self, window, mouse, settings=SETTINGS):
        self.window, self.mouse, self.settings = window, mouse, settings
        self.last_image = None

    def check(self):
        self.window.check()
        geometry = self.window.geometry()
        if geometry[2:] != self.settings.client_size:
            raise RuntimeError(f"Expected client {self.settings.client_size}; got {geometry[2:]}")
        return geometry

    def capture(self):
        x, y, _, _ = self.check()
        left, top, right, bottom = PANEL_BOX
        image = self.mouse.screenshot(region=(x+left, y+top, right-left, bottom-top))
        if self.check() != (x, y, *self.settings.client_size):
            raise RuntimeError("Window moved during capture")
        self.last_image = image
        return image

    def click_random(self):
        x, y, _, _ = self.check()
        px, py = self.settings.random_xy
        self.mouse.moveTo(x+px, y+py, _pause=False)
        # Check F9 and foreground again immediately before the actual input.
        self.check()
        try:
            self.mouse.mouseDown(button="left", _pause=False)
            self.wait(.05)
        finally:
            self.mouse.mouseUp(button="left", _pause=False)

    def wait(self, seconds):
        wait_checked(seconds, self.check)


def notify(title, text):
    """Large, closeable result notification; no game confirmation input."""
    import tkinter as tk
    popup = tk.Tk()
    popup.title(title)
    popup.attributes("-topmost", True)
    popup.geometry("680x300")
    tk.Label(popup, text=title, font=("Segoe UI", 24, "bold")).pack(pady=20)
    tk.Label(popup, text=text, font=("Segoe UI", 14), wraplength=640).pack(padx=20)
    tk.Button(popup, text="Close", font=("Segoe UI", 16), command=popup.destroy).pack(pady=18)
    popup.mainloop()


def main():
    import pyautogui
    from tailstopia_workflow.game_window import GameWindow

    cv2.setNumThreads(1)
    reader = Reader()
    window = GameWindow(SETTINGS.window_title)
    pyautogui.FAILSAFE, pyautogui.PAUSE = False, 0
    folder = ROOT / "workflow_runs/new_game_reroll" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    started = time.perf_counter()
    with (folder / "execution.jsonl").open("w", encoding="utf-8", buffering=1) as handle:
        def log(event, **data):
            row = dict(event=event, elapsed=round(time.perf_counter()-started, 6), **data)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(row, flush=True)

        print(f"Log: {folder.resolve()}")
        print("F8 = focus TalesofTails and start | F9 = EXIT")
        print(f"AGI + CHA >= {SETTINGS.agi_cha_min} AND CHA >= {SETTINGS.cha_min} "
              f"wait {SETTINGS.result_wait}s after click")
        log("settings", settings=asdict(SETTINGS))
        while not window.pressed(0x77):
            if window.pressed(0x78):
                log("stopped", reason="F9 before start")
                return
            time.sleep(.025)
        io = LiveIO(window, pyautogui)
        try:
            log("start_requested", foreground=window.api.GetForegroundWindow())
            activate_game(window)
            log("game_selected", hwnd=window.hwnd, geometry=window.geometry())
            values = run(io, reader, log)
            log("success", values=values)
            if io.last_image is not None:
                io.last_image.save(folder / "success_panel.png")
            notify("TARGET FOUND", str(values) + "\nStopped. Please confirm the character yourself.")
        except (StopRequested, KeyboardInterrupt) as exc:
            log("stopped", reason=str(exc) or "Ctrl+C")
        except Exception as exc:
            path = None
            if io.last_image is not None:
                path = str((folder / "last_panel_failure.png").resolve())
                io.last_image.save(path)
            log("failure", reason=f"{type(exc).__name__}: {exc}", last_capture=path)
            notify("SAFE STOP", f"{exc}\nLog: {folder}")
        finally:
            pyautogui.mouseUp(button="left", _pause=False)


if __name__ == "__main__":
    main()
