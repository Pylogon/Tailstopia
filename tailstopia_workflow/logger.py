import json
from datetime import datetime, timezone
from pathlib import Path


class ExecutionLog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def emit(self, kind, **fields):
        with (self.directory / "execution.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                     "kind": kind, **fields}) + "\n")
