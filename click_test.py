"""Manual diagnostic: F8 sends ONE PyAutoGUI left click; F9 exits.

Focus the game and place the pointer on RANDOM STATS before pressing F8.
No window activation, coordinate movement, OCR, or automatic repeating.
"""
import ctypes
from ctypes import wintypes
import time


HOLD_SECONDS = .15


def click_once(mouse, stopped, focus_ok, clock=time.perf_counter, sleep=time.sleep):
    if stopped() or not focus_ok():
        return False
    try:
        mouse.mouseDown(button="left", _pause=False)
        deadline = clock() + HOLD_SECONDS
        while clock() < deadline:
            if stopped() or not focus_ok():
                return False
            sleep(min(.01, max(0, deadline - clock())))
        return True
    finally:
        mouse.mouseUp(button="left", _pause=False)


def main():
    api = ctypes.windll.user32
    api.SetProcessDPIAware()
    api.GetAsyncKeyState.argtypes = [ctypes.c_int]
    api.GetAsyncKeyState.restype = ctypes.c_short
    api.GetForegroundWindow.restype = wintypes.HWND
    api.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    import pyautogui

    pyautogui.FAILSAFE, pyautogui.PAUSE = False, 0
    stopped = lambda: bool(api.GetAsyncKeyState(0x78) & 0x8000)
    print("Focus game, place mouse on RANDOM STATS.", flush=True)
    print(f"F8 = ONE left click ({HOLD_SECONDS}s hold) | F9 = EXIT", flush=True)
    print(f"Python running as admin: {bool(ctypes.windll.shell32.IsUserAnAdmin())}", flush=True)
    was_pressed = False
    while not stopped():
        pressed = bool(api.GetAsyncKeyState(0x77) & 0x8000)
        if pressed and not was_pressed:
            hwnd = api.GetForegroundWindow()
            title = ctypes.create_unicode_buffer(1024)
            api.GetWindowTextW(hwnd, title, len(title))
            print(f"F8 received | foreground={title.value!r} | mouse={pyautogui.position()}", flush=True)
            sent = click_once(pyautogui, stopped, lambda: api.GetForegroundWindow() == hwnd)
            print("mouseDown + mouseUp sent; check game response." if sent
                  else "Stopped or focus changed; mouse released.", flush=True)
        was_pressed = pressed
        time.sleep(.02)
    print("Stopped.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C.")
