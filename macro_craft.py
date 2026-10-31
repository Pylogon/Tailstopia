"""Craft recording loop: F8 starts all configured loops, F9 releases and exits.
Run: python macro_craft.py
Offline check: python macro_craft.py --dry-run --loops 2
"""
import argparse
import math
from pathlib import Path
import time

from tailstopia_macro_replay import recording, replay, validate_inputs, StopRequested

# Configuration: edit these values before launching the script.
ROOT = Path(__file__).resolve().parent
RECORDING_PATH = ROOT / "recordings/craft.json"
MAX_LOOPS = 100
LOOP_DELAY_SECONDS = 0.0  # Extra delay; recorded waits are already preserved.
WINDOW_TITLE = "TailsTopiaOnline"


def run_loops(actions, duration, api, check, loops, loop_delay,
              clock=time.perf_counter, sleep=time.sleep, dry_run=False):
    if isinstance(loops,bool) or not isinstance(loops,int) or loops < 1:
        raise ValueError("Loop count must be a positive integer")
    if isinstance(loop_delay,bool) or not isinstance(loop_delay,(int,float)) or not math.isfinite(loop_delay) or loop_delay < 0:
        raise ValueError("Loop delay must be finite and nonnegative")
    if not actions:
        raise ValueError("Recording has no actions; nothing to craft")
    for index in range(1,loops+1):
        check()
        print(f"Craft loop {index}/{loops}",flush=True)
        # The proven replayer releases held inputs on F9, lost focus, errors,
        # and Ctrl+C. Do not replace recorded holds with instantaneous clicks.
        replay(actions,duration,api,check,clock=clock,sleep=sleep,dry_run=dry_run)
        check()
        if index < loops and loop_delay:
            until = clock()+loop_delay
            while True:
                check()
                if not dry_run:
                    api.failSafeCheck()
                remaining = until-clock()
                if remaining <= 0:
                    break
                sleep(min(.01,remaining))
    return loops


def hotkeys(window,api,actions,duration,loops,loop_delay):
    was_f8 = window.pressed(0x77)
    has_keyboard = any(a["type"].startswith("key_") for a in actions)
    print("F8 = focus game and start loops | F9 = release inputs and EXIT",flush=True)
    print("Uses ORIGINAL desktop coordinates. Keep the recorded game position/size.",flush=True)
    while not window.pressed(0x78):
        f8 = window.pressed(0x77)
        if f8 and not was_f8:
            try:
                window.activate()
                if has_keyboard:
                    from tailstopia_workflow.game_window import require_english_keyboard
                    require_english_keyboard(window.api,window.hwnd)
                completed = run_loops(actions,duration,api,window.check,loops,loop_delay)
                print(f"Complete: {completed} loops. F8 = start again | F9 = exit",flush=True)
            except StopRequested:
                break
            except Exception as exc:
                print(f"STOP: {type(exc).__name__}: {exc}",flush=True)
                print("F8 = restart from loop 1 | F9 = exit",flush=True)
            # Holding F8 cannot restart another batch after the last loop.
            f8 = window.pressed(0x77)
        was_f8 = f8
        time.sleep(.02)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loops",type=int,default=MAX_LOOPS,help="Override MAX_LOOPS for this run")
    parser.add_argument("--dry-run",action="store_true",help="Fast virtual time; no game focus, input, or hotkeys")
    args = parser.parse_args()
    if args.loops < 1:
        parser.error("--loops / MAX_LOOPS must be at least 1")
    if isinstance(LOOP_DELAY_SECONDS,bool) or not isinstance(LOOP_DELAY_SECONDS,(int,float)) or not math.isfinite(LOOP_DELAY_SECONDS) or LOOP_DELAY_SECONDS < 0:
        parser.error("LOOP_DELAY_SECONDS must be finite and nonnegative")
    actions,duration = recording(RECORDING_PATH)
    if not actions:
        parser.error("Recording has no actions; nothing to craft")
    print(f"Recording: {RECORDING_PATH}",flush=True)
    print(f"Loaded {len(actions)} actions; {duration:.4f}s per loop; loops={args.loops}; extra delay={LOOP_DELAY_SECONDS}s.",flush=True)
    print("Hold summaries are not replayed twice.",flush=True)
    if args.dry_run:
        virtual = [0.0]
        def advance(seconds):
            virtual[0] += seconds
        print("DRY RUN: virtual time; no OS input.",flush=True)
        completed = run_loops(actions,duration,None,lambda:None,args.loops,LOOP_DELAY_SECONDS,
                              clock=lambda:virtual[0],sleep=advance,dry_run=True)
        print(f"Dry-run complete: {completed} loops; virtual duration {virtual[0]:.4f}s.",flush=True)
        return 0
    # Set DPI awareness before importing PyAutoGUI, as in the working replayer.
    from tailstopia_workflow.game_window import GameWindow
    window = GameWindow(WINDOW_TITLE)
    import pyautogui
    pyautogui.FAILSAFE,pyautogui.PAUSE = True,0
    actions = validate_inputs(actions,pyautogui)
    try:
        hotkeys(window,pyautogui,actions,duration,args.loops,LOOP_DELAY_SECONDS)
    except KeyboardInterrupt:
        print("Ctrl+C: stopped",flush=True)
    print("Stopped. Replay inputs released.",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
