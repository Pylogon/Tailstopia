class MockInputAdapter:
    """Only supplied adapter: no desktop input calls."""
    def __init__(self, log):
        self.log = log
        self.held = set()

    def execute(self, action, **context):
        kind = action["type"]
        token = ("key", action.get("key")) if kind.startswith("key_") else ("mouse", action.get("button"))
        if kind.endswith("_down"):
            self.held.add(token)
        elif kind.endswith("_up"):
            self.held.discard(token)
        self.log.emit("action", action=action, **context)

    def release_all(self):
        self.log.emit("release_all", inputs=sorted(self.held))
        self.held.clear()
