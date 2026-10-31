import argparse
import threading
from .config import load_config
from .controller import Controller, Status
from .control import Control
from .input_adapter import MockInputAdapter
from .logger import ExecutionLog
from .screen import MockScreen, RecordedScreen


def main():
    parser = argparse.ArgumentParser(description="Offline TailsTopia workflow; mock inputs only")
    parser.add_argument("config")
    parser.add_argument("--realtime", action="store_true", help="Honor waits; inputs remain mocked")
    parser.add_argument("--interactive", action="store_true", help="Console p/r/s + Enter: pause/resume/stop")
    args = parser.parse_args()
    config = load_config(args.config)
    spec = config.get("screen", {})
    screen = (RecordedScreen(spec["frames"], spec["checks"], spec.get("rewards"))
              if spec.get("backend") == "recorded" else MockScreen(spec.get("script"), spec.get("rewards")))
    log, control = ExecutionLog(config["output"]), Control()
    if args.interactive:
        def commands():
            while True:
                try:
                    command = input().strip().lower()
                except EOFError:
                    return
                if command in {"p", "r", "s"}:
                    {"p": control.pause, "r": control.resume, "s": control.stop}[command]()
        threading.Thread(target=commands, daemon=True).start()
    result = Controller(config, MockInputAdapter(log), screen, log, control, args.realtime).run()
    print(f"{result.status.value}: {result.reason}; loops={result.loops_completed}; state={result.state}")
    return 0 if result.status == Status.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
