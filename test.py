"""F8: focus TailsTopia and hold A. F9: release A and exit.
Install: py -m pip install pyautogui
Run: py test.py
"""
import ctypes
from ctypes import wintypes
import time

GAME_TITLE_PART = "TailsTopiaOnline"


def main():
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

    def release_a():
        # Always allow release even after the pointer triggers PyAutoGUI failsafe.
        enabled = pyautogui.FAILSAFE
        try:
            pyautogui.FAILSAFE = False
            pyautogui.keyUp("a", _pause=False)
        finally:
            pyautogui.FAILSAFE = enabled

    held = False
    hwnd = None
    was_f8 = pressed(0x77)
    print("F8 = focus game and hold A | F9 = release A and EXIT", flush=True)
    try:
        while not pressed(0x78):
            f8 = pressed(0x77)
            if f8 and not was_f8 and not held:
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
                    from tailstopia_workflow.game_window import require_english_keyboard
                    require_english_keyboard(user32,hwnd)
                    held = True
                    pyautogui.keyDown("a", _pause=False)
                    print("Holding A on game. F9 = stop", flush=True)
                except Exception as exc:
                    if held:
                        release_a()
                    held = False
                    print(f"STOP: {exc}", flush=True)
            if held:
                if user32.GetForegroundWindow() != hwnd:
                    release_a()
                    held = False
                    print("Game lost focus: A released. F8 = start again", flush=True)
                pyautogui.failSafeCheck()
            was_f8 = f8
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Ctrl+C: stop")
    except pyautogui.FailSafeException:
        print("Mouse failsafe: stop")
    finally:
        if held:
            release_a()
        print("Stopped. A released.", flush=True)


if __name__ == "__main__":
    main()
