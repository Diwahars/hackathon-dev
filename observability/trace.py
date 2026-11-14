from contextlib import contextmanager
from datetime import datetime, timezone
import time
import uuid

def now():
    return datetime.now(timezone.utc).isoformat()

class Span:
    def __init__(self, trace, name, input_data=None, parent_id=None):
        self.trace, self.name = trace, name
        self.input_data, self.parent_id = input_data, parent_id
    def __enter__(self):
        self.started = time.perf_counter()
        self.record = {
            "span_id": str(uuid.uuid4()), "parent_id": self.parent_id,
            "agent_name": self.name, "status": "running",
            "started_at": now(), "duration_ms": None,
            "input": self.input_data, "output": None, "error": None,
        }
        self.trace.spans.append(self.record)
        return self
    def set_output(self, output):
        self.record["output"] = output
    def __exit__(self, typ, exc, tb):
        self.record["duration_ms"] = round((time.perf_counter()-self.started)*1000, 2)
        self.record["ended_at"] = now()
        self.record["status"] = "failed" if exc else "completed"
        if exc:
            self.record["error"] = str(exc)
        return False

class Trace:
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.started_at = now()
        self.spans = []
    @contextmanager
    def span(self, name, input_data=None, parent_id=None):
        span = Span(self, name, input_data, parent_id)
        with span:
            yield span
    def to_dict(self):
        return {"trace_id": self.trace_id, "started_at": self.started_at,
                "spans": self.spans}
