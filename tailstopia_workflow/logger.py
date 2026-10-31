import json
import time
from datetime import datetime, timezone
from pathlib import Path


class ExecutionLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stream = None
        self.reset_timing()

    def __enter__(self):
        # Mission runs keep one handle; other callers retain one-shot writes.
        if self._stream is None:
            self._stream = (self.directory / "execution.jsonl").open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()

    def reset_timing(self):
        self._calls, self._seconds = 0, 0.0

    def timing_summary(self):
        # A state's summary excludes the write of that summary itself.
        return {"calls": self._calls, "seconds": round(self._seconds, 6)}

    def emit(self, kind, **fields):
        started = time.perf_counter()
        try:
            line = json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                               "kind": kind, **fields}) + "\n"
            if self._stream is None:
                with (self.directory / "execution.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(line)
            else:
                self._stream.write(line)
                # Keep each event visible immediately; do not batch or drop it.
                self._stream.flush()
        finally:
            self._calls += 1
            self._seconds += time.perf_counter() - started
