"""Step 1 only: F8 checks Time until ready; F9 exits. Sends no game input."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

from PIL import Image, ImageGrab
from tailstopia_workflow.start_detection import StartDetector, wait_for_start

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"examples/step1.json")
    parser.add_argument("--image",type=Path,nargs="+",help="Inspect recorded full screenshots instead of live capture")
    args = parser.parse_args()
    detector = StartDetector(args.config)
    if args.image:
        for path in args.image:
            with Image.open(path) as image:
                observation = detector.inspect(detector.crop_screenshot(image))
            print(path.name,json.dumps(asdict(observation)))
        return
    from tailstopia_workflow.game_window import GameWindow
    from tailstopia_macro_replay import StopRequested
    window = GameWindow()
    output = ROOT/"workflow_runs/step1"
    output.mkdir(parents=True,exist_ok=True)
    previous_f8 = window.pressed(0x77)
    print("STEP 1 ONLY: F8 = focus game and verify Time | F9 = exit. No macro/clicks.",flush=True)
    try:
        while not window.pressed(0x78):
            f8 = window.pressed(0x77)
            if f8 and not previous_f8:
                last_roi = None
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                try:
                    window.activate()
                    def capture():
                        nonlocal last_roi
                        left,top,width,height = window.geometry()
                        x1,y1,x2,y2 = detector.roi
                        if x2 > width or y2 > height:
                            raise RuntimeError("Game client smaller than configured ROI")
                        last_roi = ImageGrab.grab(bbox=(left+x1,top+y1,left+x2,top+y2),all_screens=True)
                        return last_roi
                    def report(observation,count,elapsed):
                        row = {**asdict(observation),"confirmations":count,"elapsed":elapsed}
                        print(json.dumps(row),flush=True)
                        with (output/f"{stamp}.jsonl").open("a",encoding="utf-8") as stream:
                            stream.write(json.dumps(row)+"\n")
                    wait_for_start(detector,capture,window.check,report)
                    last_roi.save(output/f"{stamp}_ready.png")
                    print("SUCCESS: Time confirmed. Step 1 finished; no start recording played.",flush=True)
                except StopRequested:
                    if last_roi is not None:
                        last_roi.save(output/f"{stamp}_stopped.png")
                    break
                except Exception as exc:
                    if last_roi is not None:
                        last_roi.save(output/f"{stamp}_failure.png")
                    print(f"FAIL: {type(exc).__name__}: {exc}",flush=True)
                previous_f8 = window.pressed(0x77)
            else:
                previous_f8 = f8
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C")
    print("Stopped.")


if __name__ == "__main__":
    main()
