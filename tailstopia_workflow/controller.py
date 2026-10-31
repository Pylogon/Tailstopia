from dataclasses import dataclass
from enum import Enum
import time
import uuid

from .config import STATES, validate, coordinate
from .control import Control
from .macro_parser import load_macro
from .reward_selector import select_reward


class Status(str, Enum):
    SUCCESS = "SUCCESS"
    RETRY = "RETRY"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Result:
    status: Status
    reason: str
    state: str
    loops_completed: int


class Controller:
    def __init__(self, config, adapter, screen, log, control=None, realtime=False):
        self.config = validate(config)
        self.adapter, self.screen, self.log = adapter, screen, log
        self.control = control or Control()
        self.realtime = realtime
        self.run_id = uuid.uuid4().hex
        self.completed = 0
        self.state = "VERIFY_START"
        self.items = []
        self.macros = {}
        self.simulated = 0.0

    def now(self):
        return self.control.now() + self.simulated

    def guard(self, deadline):
        self.control.checkpoint()
        if self.now() >= deadline:
            raise TimeoutError("State deadline exceeded")

    def wait(self, duration, deadline):
        if self.realtime:
            self.control.wait(duration, deadline)
        else:
            self.simulated += duration
            self.guard(deadline)

    def actions(self, actions, deadline, loop, recorded=False):
        previous = 0.0
        for index, action in enumerate(actions):
            self.guard(deadline)
            if recorded:
                self.wait(action["t"] - previous, deadline)
                previous = action["t"]
            if action["type"] != "wait":
                self.adapter.execute(action, run_id=self.run_id, loop=loop, state=self.state,
                                     index=index, offset=action.get("t"))
            self.wait(action.get("wait", 0), deadline)

    def verify(self, name, deadline, loop):
        policy = self.config["states"][self.state]
        for attempt in range(policy["retries"] + 1):
            self.guard(deadline)
            try:
                matched = self.screen.check(name)
                reason = "image not matched"
            except Exception as exc:
                matched, reason = False, f"recognition error: {type(exc).__name__}: {exc}"
            self.guard(deadline)
            if matched:
                return
            if attempt == policy["retries"]:
                raise RuntimeError(f"{name}: {reason}; retry budget exhausted")
            self.log.emit("state_retry", run_id=self.run_id, loop=loop, state=self.state,
                          status=Status.RETRY.value, reason=reason, attempt=attempt + 1)
            self.wait(policy["retry_delay"], deadline)

    def execute_state(self, deadline, loop):
        state = self.state
        if state.startswith("VERIFY_"):
            self.verify({"VERIFY_START": "start", "VERIFY_STAGE": "stage", "VERIFY_COMPLETE": "complete"}[state], deadline, loop)
        elif state in ("PLAY_MACRO_A", "PLAY_MACRO_B"):
            self.actions(self.macros[state], deadline, loop, recorded=True)
        elif state == "ENTER_STAGE":
            self.actions(self.config["enter_stage"], deadline, loop)
        elif state == "SCAN_REWARD":
            self.items = self.screen.scan()
            for item in self.items:
                coordinate(item)
                if not isinstance(item["name"], str):
                    raise ValueError("Invalid reward name")
            self.log.emit("reward_scan", run_id=self.run_id, loop=loop, items=self.items)
        elif state == "SELECT_REWARD":
            action, reason = select_reward(self.items, self.config["reward"]["priority"], self.config["reward"]["fallback"])
            self.log.emit("reward_selection", run_id=self.run_id, loop=loop, reason=reason, action=action)
            self.actions([action], deadline, loop)
        elif state == "RETURN_START":
            self.actions(self.config["return_start"], deadline, loop)
            # Retry only the observation, never the return command.
            self.verify("start", deadline, loop)
        else:
            raise ValueError("Unknown state")
        self.guard(deadline)

    def run(self):
        result = None
        try:
            # Preflight both files before the first side effect.
            self.macros = {"PLAY_MACRO_A": load_macro(self.config["macro_a"]),
                           "PLAY_MACRO_B": load_macro(self.config["macro_b"])}
            for loop in range(1, self.config["max_loops"] + 1):
                for self.state in STATES:
                    self.simulated = 0.0
                    active_start, wall_start = self.now(), time.monotonic()
                    deadline = active_start + self.config["states"][self.state]["timeout"]
                    self.log.emit("state_entry", run_id=self.run_id, loop=loop, state=self.state)
                    try:
                        self.execute_state(deadline, loop)
                    except Exception as exc:
                        self.log.emit("state_result", run_id=self.run_id, loop=loop, state=self.state,
                                      status="FAIL", reason=str(exc), elapsed=self.now() - active_start,
                                      wall_elapsed=time.monotonic() - wall_start)
                        raise
                    self.log.emit("state_result", run_id=self.run_id, loop=loop, state=self.state,
                                  status="SUCCESS", reason="expected result satisfied",
                                  elapsed=self.now() - active_start, wall_elapsed=time.monotonic() - wall_start)
                self.completed += 1
            result = Result(Status.SUCCESS, "Loop limit reached; return verified", self.state, self.completed)
        except (Exception, KeyboardInterrupt) as exc:
            self.control.stop()
            self.adapter.release_all()
            result = Result(Status.FAIL, f"{type(exc).__name__}: {exc}", self.state, self.completed)
            try:
                path = self.log.directory / f"{self.run_id}_{self.state}_failure.png"
                self.screen.save_debug(path)
                self.log.emit("debug_image", path=str(path), synthetic=type(self.screen).__name__ == "MockScreen")
            except Exception as debug_exc:
                self.log.emit("debug_error", reason=str(debug_exc))
        finally:
            self.adapter.release_all()
        self.log.emit("run_result", run_id=self.run_id, status=result.status.value, reason=result.reason,
                      state=result.state, loops_completed=result.loops_completed)
        return result
