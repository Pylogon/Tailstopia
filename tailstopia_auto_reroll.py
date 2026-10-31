"""TailsTopia fast stat reroller using ONLY screen reading and normal mouse clicks.

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
# Faster defaults. Raise ROLL_SETTLE_SECONDS if the game animation is still playing.
ROLL_SETTLE_SECONDS = 0.75
SCREEN_STABILITY_SECONDS = 0.12
READ_CONFIRM_SECONDS = 0.18             # only when confirming a potential MATCH
OCR_RETRY_ATTEMPTS = 2
OCR_RETRY_DELAY_SECONDS = 0.25
BETWEEN_ROLLS_SECONDS = 0.10
BENCHMARK_EVERY_ROLL = True
MAX_ROLLS = 5000                          # prevent infinite clicking
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


def prepared_ocr_image(crop, threshold):
    """Preprocess digit crop. Tested on the provided 11 and 12 screenshots."""
    gray = ImageOps.grayscale(crop)
    large = gray.resize((gray.width * 8, gray.height * 8), Image.Resampling.LANCZOS)
    bw = large.point(lambda p: 255 if p > threshold else 0)
    return ImageOps.expand(bw, border=35, fill=255)


def read_digit(crop, threshold, psm):
    raw = pytesseract.image_to_string(
        prepared_ocr_image(crop, threshold),
        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
    ).strip()
    if not re.fullmatch(r"\d{1,2}", raw):
        return None
    value = int(raw)
    return value if 0 <= value <= 30 else None


def ocr_number_fast(crop):
    """Same shared OCR logic for ALL stats; fail closed on uncertain readings.

    Use two nearby binarization thresholds with the SAME page segmentation mode.
    Older 135/psm7 + 145/psm6 failed systematically on a visible '8' and '5':
    one of those two modes would return blank, forcing unnecessary SAFE STOPs.
    Tested 140/psm7 and 145/psm7 against the supplied debug panels (11,12,8,5).
    """
    a = read_digit(crop, threshold=140, psm=7)
    b = read_digit(crop, threshold=145, psm=7)
    if a is not None and a == b:
        return a
    # One fallback on different contrast before declaring uncertainty.
    c = read_digit(crop, threshold=135, psm=7)
    if a is not None and c == a:
        return a
    if b is not None and c == b:
        return b
    return None


def ocr_number_robust(crop):
    """Thorough check ONLY when a roll may pass the target condition."""
    results = []
    for threshold in (135, 140, 145):
        for psm in (6, 7):
            value = read_digit(crop, threshold, psm)
            if value is not None:
                results.append(value)
    counts = Counter(results)
    if not counts:
        return None
    value, votes = counts.most_common(1)[0]
    if votes >= 4 and all(count <= 1 for other, count in counts.items() if other != value):
        return value
    return None


def screenshot_stable(first_crops, second_crops):
    """Require same thresholded glyph within a few pixels across screenshots.

    This is a quick pixel check, NOT an additional OCR pass.
    If the UI is still animating, retry capture without clicking the reroll button.
    """
    for name in STAT_CENTERS:
        a = ImageOps.grayscale(first_crops[name]).point(lambda p: 255 if p > 140 else 0)
        b = ImageOps.grayscale(second_crops[name]).point(lambda p: 255 if p > 140 else 0)
        different = sum(x != y for x, y in zip(a.tobytes(), b.tobytes()))
        if different > 4:
            return False
    return True


def save_debug(crops, panel):
    DEBUG_DIR.mkdir(exist_ok=True)
    panel.save(DEBUG_DIR / "last_panel.png")
    for name, crop in crops.items():
        crop.resize((crop.width * 8, crop.height * 8)).save(DEBUG_DIR / f"{name}.png")


def fast_stats_from_crops(crops):
    return {name: ocr_number_fast(crops[name]) for name in STAT_CENTERS}


def robust_stats_from_crops(crops):
    return {name: ocr_number_robust(crops[name]) for name in STAT_CENTERS}


def capture_and_read_fast(sct, left, top):
    """Fast loop: two SCREEN captures, only 8 OCR calls total."""
    first, _ = capture_stats(sct, left, top)
    time.sleep(SCREEN_STABILITY_SECONDS)
    crops, panel = capture_stats(sct, left, top)
    if not screenshot_stable(first, crops):
        return None, crops, panel, "stat pixels still changing"
    values = fast_stats_from_crops(crops)
    if any(v is None for v in values.values()):
        return None, crops, panel, f"OCR ambiguous: {values}"
    return values, crops, panel, "ok"


def verify_potential_match(sct, left, top, expected):
    """Candidate must survive TWO fresh robust OCR reads before halting."""
    print("POTENTIAL MATCH: doing full 6-method verification x2...", flush=True)
    for i in range(2):
        if stop_requested.is_set():
            raise RuntimeError("F9 pressed during verification")
        time.sleep(READ_CONFIRM_SECONDS)
        crops, panel = capture_stats(sct, left, top)
        values = robust_stats_from_crops(crops)
        if values != expected or not target_met(values):
            save_debug(crops, panel)
            raise RuntimeError(
                f"Candidate verification failed: fast={expected}, robust={values}. "
                "Stopped WITHOUT another reroll; check debug images."
            )
    return True


def stable_stats(sct, left, top, *, verify_candidate=True):
    """Read with a fast path; retry twice on uncertainty WITHOUT clicking.

    A confirmed candidate gets two independent robust reads. Never advance
    another reroll when OCR is ambiguous; there is no risk-free OCR guarantee.
    """
    last_reason = "unreadable"
    for attempt in range(OCR_RETRY_ATTEMPTS + 1):
        if stop_requested.is_set():
            raise RuntimeError("F9 pressed during OCR")
        if attempt:
            print(f"OCR unclear: retry {attempt}/{OCR_RETRY_ATTEMPTS} (NO CLICK)...", flush=True)
            time.sleep(OCR_RETRY_DELAY_SECONDS)
        values, crops, panel, reason = capture_and_read_fast(sct, left, top)
        last_reason = reason
        if values is None:
            continue
        if verify_candidate and target_met(values):
            verify_potential_match(sct, left, top, values)
        return values

    save_debug(crops, panel)
    raise RuntimeError(
        f"OCR still unclear after {OCR_RETRY_ATTEMPTS} retries: {last_reason}. "
        "Stopped WITHOUT another reroll; check tailstopia_debug/"
    )


def format_stats(stats):
    return (f"INT={stats['INT']} TAL={stats['TAL']} "
            f"VIT={stats['VIT']} AGI={stats['AGI']}")


def target_met(stats):
    return (stats["INT"] >= 11 and stats["TAL"] >= 12
            and stats["VIT"] + stats["AGI"] >= 10)


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
            roll_start = time.perf_counter()
            pyautogui.click(left + x, top + y)
            time.sleep(ROLL_SETTLE_SECONDS)
            if stop_requested.is_set():
                print("F9 pressed. Stopped.")
                return
            read_start = time.perf_counter()
            current = stable_stats(sct, left, top)
            read_elapsed = time.perf_counter() - read_start
            ok = target_met(current)
            print(f"Reroll {iteration:4d}: {format_stats(current)} "
                  f"VIT+AGI={current['VIT'] + current['AGI']} => "
                  f"{'FOUND!' if ok else 'NO'}", flush=True)
            if ok:
                crops, panel = capture_stats(sct, left, top)
                save_debug(crops, panel)
                print("\nSUCCESS: stopped rerolling. VERIFY values in game manually.")
                print("The script will NOT click the CREATE button.")
                beep_success()
                return
            repeated = repeated + 1 if current == prev else 0
            if repeated >= 2:
                print("Stats unchanged twice after clicking. PAUSED: button/window may not work.")
                return
            prev = current
            if BENCHMARK_EVERY_ROLL:
                print(f"  Timing: OCR+capture={read_elapsed:.2f}s | "
                      f"round={time.perf_counter()-roll_start:.2f}s", flush=True)
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
