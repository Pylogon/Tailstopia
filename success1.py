"""TailsTopia stat reroller using ONLY visible screen reading and normal mouse clicks.

Targets: INT >= 11, TAL >= 12, VIT + AGI >= 10.
Reference: 1280x600 game client, as in the user's 1366x768 screenshot.

Safety: F9 stops, move mouse to top-left stops (PyAutoGUI failsafe),
OCR failure/unstable screen stops rather than clicking, no auto-CREATE.
No game memory injection, anti-cheat interference, or bypass.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from collections import Counter
from pathlib import Path
import re
import threading
import time

import mss
from PIL import Image, ImageOps
import pytesseract

try:
    import pyautogui
except ImportError:
    raise SystemExit("Missing pyautogui. Install: py -m pip install pyautogui mss pillow pytesseract")

# ---- SETTINGS ----
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
GAME_TITLE_PART = "TailsTopiaOnline"
REFERENCE_CLIENT_SIZE = (1280, 600)    # Screenshot: game content from (43,95) to (1323,695)
# Centers RELATIVE TO GAME CLIENT, not the desktop.
STAT_CENTERS = {
    "INT": (1176, 410),   # image approx (1219,505)
    "TAL": (1176, 450),   # image approx (1219,545)
    "VIT": (1088, 470),   # image approx (1131,565)
    "AGI": (1088, 450),   # image approx (1131,545)
}
REROLL_BUTTON = (1097, 497)              # image approx (1140,592)
STAT_CROP_SIZE = (24, 19)
ROLL_SETTLE_SECONDS = 1.0                 # allow the UI animation to finish
READ_CONFIRM_SECONDS = 0.25
OCR_RETRY_ATTEMPTS = 2                    # extra retries after initial failed pair
OCR_RETRY_DELAY_SECONDS = 0.4              # allow the UI to settle before retry
BETWEEN_ROLLS_SECONDS = 0.25
MAX_ROLLS = 1000                          # prevent infinite clicking
DEBUG_DIR = Path("tailstopia_debug")
# ------------------

if Path(TESSERACT_EXE).is_file():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.15
user32 = ctypes.windll.user32
# On 64-bit Windows, HWND is pointer-sized. Avoid truncation from ctypes defaults.
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
try:
    user32.SetProcessDPIAware()
except Exception:
    pass

stop_requested = threading.Event()


def stop_key_monitor():
    """F9 checked via built-in Windows API; no keyboard-hook library."""
    while not stop_requested.is_set():
        if user32.GetAsyncKeyState(0x78) & 0x8000:  # F9
            stop_requested.set()
            break
        time.sleep(0.05)


def find_game_window():
    """Return (hwnd, x, y, width, height) where x/y is CLIENT top-left."""
    matches = []
    cb_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @cb_type
    def enum_cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        count = user32.GetWindowTextLengthW(hwnd)
        if not count:
            return True
        buf = ctypes.create_unicode_buffer(count + 1)
        user32.GetWindowTextW(hwnd, buf, count + 1)
        if GAME_TITLE_PART.lower() in buf.value.lower():
            matches.append(hwnd)
        return True

    user32.EnumWindows(enum_cb, 0)
    if not matches:
        raise RuntimeError("Cannot find game window 'TailsTopiaOnline'. Open the game first.")
    hwnd = matches[0]
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("Cannot read game client size")
    point = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        raise RuntimeError("Cannot read game window position")
    return hwnd, point.x, point.y, rect.right - rect.left, rect.bottom - rect.top


def window_geometry():
    hwnd, left, top, width, height = find_game_window()
    if abs(width - REFERENCE_CLIENT_SIZE[0]) > 2 or abs(height - REFERENCE_CLIENT_SIZE[1]) > 2:
        raise RuntimeError(
            f"Game client is {width}x{height}, but reference screenshot is "
            f"{REFERENCE_CLIENT_SIZE[0]}x{REFERENCE_CLIENT_SIZE[1]}. "
            "Use 1280x600 windowed mode, or send a new screenshot for recalibration."
        )
    return hwnd, left, top


def capture_stats(sct, left, top):
    """Take 1 screenshot of panel, crop each stat. All positions are client-relative."""
    left0, top0, width, height = 1063, 391, 137, 96
    shot = sct.grab({"left": left + left0, "top": top + top0,
                     "width": width, "height": height})
    panel = Image.frombytes("RGB", shot.size, shot.rgb)
    crops = {}
    w, h = STAT_CROP_SIZE
    for name, (x, y) in STAT_CENTERS.items():
        cx, cy = x - left0, y - top0
        crops[name] = panel.crop((cx-w//2, cy-h//2, cx+w//2, cy+h//2))
    return crops, panel


def ocr_number(crop):
    """Read a stat without losing leading '1' in values like 11 and 12.

    A real INT=11 image was previously read as 1 because OCR accepted the
    first two (incorrect) low-threshold results. Instead, compare six OCR
    readings from thresholds validated against the supplied INT=11 and 12
    screenshots. If the readings are inconsistent, return None and stop
    safely rather than risking another reroll.
    """
    gray = ImageOps.grayscale(crop)
    large = gray.resize((gray.width * 8, gray.height * 8), Image.Resampling.LANCZOS)

    results = []
    for threshold in (135, 140, 145):
        bw = large.point(lambda p: 255 if p > threshold else 0)
        padded = ImageOps.expand(bw, border=35, fill=255)
        for psm in (6, 7):
            raw = pytesseract.image_to_string(
                padded, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
            ).strip()
            if re.fullmatch(r"\d{1,2}", raw):
                value = int(raw)
                if 0 <= value <= 30:
                    results.append(value)

    counts = Counter(results)
    if not counts:
        return None

    value, votes = counts.most_common(1)[0]
    # At least four agreeing results, from the different threshold/PSM
    # combinations, and no serious second interpretation.
    if votes >= 4 and all(count <= 1 for other, count in counts.items() if other != value):
        return value
    return None


def read_stats(sct, left, top, *, save_debug=False):
    crops, panel = capture_stats(sct, left, top)
    values = {name: ocr_number(crops[name]) for name in STAT_CENTERS}
    if save_debug or any(value is None for value in values.values()):
        DEBUG_DIR.mkdir(exist_ok=True)
        panel.save(DEBUG_DIR / "last_panel.png")
        for name, crop in crops.items():
            crop.resize((crop.width * 8, crop.height * 8)).save(DEBUG_DIR / f"{name}.png")
    return values


def format_stats(stats):
    return (f"INT={stats['INT']} TAL={stats['TAL']} "
            f"VIT={stats['VIT']} AGI={stats['AGI']}")


def target_met(stats):
    return (stats["INT"] >= 11 and stats["TAL"] >= 12
            and stats["VIT"] + stats["AGI"] >= 10)


def stable_stats(sct, left, top):
    """Read twice; retry an unstable pair up to OCR_RETRY_ATTEMPTS times.

    IMPORTANT: During all OCR retries, do NOT click REROLL; keep checking
    the same result so a potentially good roll cannot be overwritten.
    """
    last_first = last_second = None
    for attempt in range(OCR_RETRY_ATTEMPTS + 1):
        if stop_requested.is_set():
            raise RuntimeError("F9 pressed during OCR. Stopped safely.")
        if attempt:
            print(f"OCR unclear: retry {attempt}/{OCR_RETRY_ATTEMPTS} (no click)...", flush=True)
            time.sleep(OCR_RETRY_DELAY_SECONDS)
        first = read_stats(sct, left, top)
        if stop_requested.is_set():
            raise RuntimeError("F9 pressed during OCR. Stopped safely.")
        time.sleep(READ_CONFIRM_SECONDS)
        second = read_stats(sct, left, top)
        last_first, last_second = first, second
        if first == second and all(v is not None for v in first.values()):
            if attempt:
                print(f"OCR recovered on retry {attempt}: {format_stats(second)}", flush=True)
            return second

    read_stats(sct, left, top, save_debug=True)
    raise RuntimeError(
        f"OCR still unstable after {OCR_RETRY_ATTEMPTS} retries: "
        f"{last_first} versus {last_second}. Stopped WITHOUT another reroll. "
        f"Check {DEBUG_DIR}/ images, then recalibrate."
    )


def beep_success():
    try:
        import winsound
        for _ in range(4):
            winsound.Beep(1100, 280)
            time.sleep(0.08)
    except Exception:
        print("\a", flush=True)


def main():
    print("TailsTopia auto reroll | INT>=11 TAL>=12 VIT+AGI>=10")
    print("F9 = STOP. Mouse at TOP-LEFT corner = PyAutoGUI emergency stop.")
    if not Path(TESSERACT_EXE).is_file():
        print(f"Note: default Tesseract path not found: {TESSERACT_EXE}")
        print("Trying the Tesseract executable in your PATH instead.")
    print("Make sure game is visible, windowed 1280x600, and stats panel is open.")
    _, left, top = window_geometry()
    print(f"Game client starts at screen ({left},{top}).")

    with mss.mss() as sct:
        initial = stable_stats(sct, left, top)
        print("Initial OCR result:", format_stats(initial))
        print("COMPARE with real values in game. If incorrect, do not start.")
        ans = input("Are ALL FOUR values correct? Type YES to start, anything else to exit: ")
        if ans.strip().upper() != "YES":
            print("Stopped without clicking.")
            return
        if target_met(initial):
            print("ALREADY MATCHED! No click necessary.")
            beep_success()
            return

        print("Starting in 5 seconds. Click GAME WINDOW now to make it active.")
        threading.Thread(target=stop_key_monitor, daemon=True).start()
        for n in range(5, 0, -1):
            if stop_requested.is_set():
                print("Stop requested before starting.")
                return
            print(n, end=" ", flush=True)
            time.sleep(1)
        print("\nRunning...")
        prev = initial
        repeated = 0
        for iteration in range(1, MAX_ROLLS + 1):
            if stop_requested.is_set():
                print("F9 pressed. Stopped.")
                return
            hwnd, left, top = window_geometry()
            if user32.GetForegroundWindow() != hwnd:
                print("Game is not the active window. PAUSED to prevent clicking another app.")
                return
            if pyautogui.position().x <= 3 and pyautogui.position().y <= 3:
                print("Mouse failsafe position. Stopped.")
                return
            x, y = REROLL_BUTTON
            pyautogui.click(left + x, top + y)
            time.sleep(ROLL_SETTLE_SECONDS)
            if stop_requested.is_set():
                print("F9 pressed. Stopped.")
                return
            current = stable_stats(sct, left, top)
            ok = target_met(current)
            print(f"Reroll {iteration:4d}: {format_stats(current)} "
                  f"VIT+AGI={current['VIT'] + current['AGI']} => "
                  f"{'FOUND!' if ok else 'NO'}", flush=True)
            if ok:
                read_stats(sct, left, top, save_debug=True)
                print("\nSUCCESS: stopped rerolling. VERIFY values in game manually.")
                print("The script will NOT click the CREATE button.")
                beep_success()
                return
            repeated = repeated + 1 if current == prev else 0
            if repeated >= 2:
                print("Stats unchanged twice after clicking. PAUSED: button/window may not work.")
                return
            prev = current
            time.sleep(BETWEEN_ROLLS_SECONDS)
        print(f"Stopped after {MAX_ROLLS} attempts (safety limit). Change MAX_ROLLS if needed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C")
    except pyautogui.FailSafeException:
        print("\nEmergency stop: mouse reached upper-left corner.")
    except Exception as exc:
        print(f"\nSAFE STOP: {type(exc).__name__}: {exc}")
    finally:
        stop_requested.set()
