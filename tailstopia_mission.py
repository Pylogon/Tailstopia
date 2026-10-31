"""Complete note.txt flow. F8=start, F9=stop/exit, F7=request pause/resume."""
import argparse
from datetime import datetime
from pathlib import Path
import time
import sys

from tailstopia_workflow.mission_config import load_mission_config, validate_mission_config, STATES
from tailstopia_workflow.mission_flow import MissionFlow

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"examples/mission_flow.json")
    parser.add_argument("--loops",type=int,help="Number of full loops, default from config")
    parser.add_argument("--dry-run",action="store_true",help="Fast simulated input/time; real reference-image recognition")
    parser.add_argument("--through",choices=STATES,help="Run initial steps through this state, then stop")
    args = parser.parse_args()
    config,base = load_mission_config(args.config)
    if args.loops is not None:
        config["max_loops"] = args.loops
        validate_mission_config(config)
    try:
        from tailstopia_workflow.mission_runtime import LiveSession, FixtureSession
        from tailstopia_workflow.mission_vision import MissionVision
    except ModuleNotFoundError as exc:
        if exc.name not in {"cv2", "numpy", "PIL"}:
            raise
        environment_python = ROOT / ".venv" / "Scripts" / "python.exe"
        raise SystemExit(
            f"Missing dependency: {exc.name}\n"
            f"Current Python: {sys.executable}\n"
            f'Run with the project environment:\n  & "{environment_python}" "{Path(__file__).resolve()}"\n'
            f'Or install dependencies for your current Python:\n'
            f'  & "{sys.executable}" -m pip install -r "{ROOT / "requirements-mission.txt"}"'
        ) from None
    # Load and validate templates before any game interaction.
    vision = MissionVision(config,base)
    def directory(mode):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return (base/config["output"]).resolve()/f"{stamp}_{mode}"
    if args.dry_run:
        session = FixtureSession(config,directory("dry"),ROOT/"resource")
        print("DRY RUN: no OS input; virtual waits. Reward pickup is a synthetic image change.",flush=True)
        result = MissionFlow(config,base,session,vision).run(args.through)
        print(result)
        print("Log:",session.log.directory)
        return 0 if result.status == "SUCCESS" else 1
    from tailstopia_workflow.game_window import GameWindow
    from tailstopia_macro_replay import StopRequested
    window = GameWindow(config["window_title"])
    previous_f8 = window.pressed(0x77)
    print("F8 = run flow | F9 = stop/exit | F7 = pause/resume (active holds finish first)",flush=True)
    print("Recordings use absolute desktop positions: keep the original game window position/size.",flush=True)
    try:
        while not window.pressed(0x78):
            f8 = window.pressed(0x77)
            if f8 and not previous_f8:
                session = None
                try:
                    window.activate()
                    session = LiveSession(config,directory("live"),window)
                    log_path = (session.log.directory/"execution.jsonl").resolve()
                    print("Log:",log_path,flush=True)
                    session.event("run_start",log_path=str(log_path))
                    result = MissionFlow(config,base,session,vision).run(args.through)
                    print(result)
                    print("Log:",session.log.directory)
                    if result.stopped:
                        break
                except StopRequested:
                    break
                except Exception as exc:
                    if session is not None:
                        session.event("run_result",status="FAIL",reason=f"{type(exc).__name__}: {exc}")
                    print(f"FAIL: {type(exc).__name__}: {exc}",flush=True)
                finally:
                    if session is not None:
                        session.api.release_all()
                print("Ready: F8 = start from first step | F9 = exit",flush=True)
                previous_f8 = window.pressed(0x77)
            else:
                previous_f8 = f8
            time.sleep(.02)
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
