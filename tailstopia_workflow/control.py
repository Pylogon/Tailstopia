import threading
import time


class Stopped(RuntimeError):
    pass


class Control:
    """Pause freezes active deadlines. Stop interrupts paused and waiting states."""
    def __init__(self):
        self.condition = threading.Condition()
        self.paused = False
        self.stopped = False
        self.pause_start = None
        self.pause_total = 0.0

    def pause(self):
        with self.condition:
            if not self.paused:
                self.paused = True
                self.pause_start = time.monotonic()

    def resume(self):
        with self.condition:
            if self.paused:
                self.pause_total += time.monotonic() - self.pause_start
                self.paused = False
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def now(self):
        with self.condition:
            now = time.monotonic()
            return now - self.pause_total - (now - self.pause_start if self.paused else 0)

    def checkpoint(self):
        with self.condition:
            while self.paused and not self.stopped:
                self.condition.wait(0.05)
            if self.stopped:
                raise Stopped("Stop requested")

    def wait(self, seconds, deadline):
        target = self.now() + seconds
        while True:
            self.checkpoint()
            now = self.now()
            if now >= deadline:
                raise TimeoutError("State deadline exceeded")
            if now >= target:
                return
            with self.condition:
                self.condition.wait(min(0.05, target - now, deadline - now))
