
# observability.py

import json
import os
import sqlite3
import time
import uuid

from datetime import datetime, timezone
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv(
    "DATABASE_PATH",
    "agent_traces.db",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_id():
    return str(uuid.uuid4())


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY,
                query TEXT,
                answer TEXT,
                status TEXT,
                started_at TEXT,
                duration_ms REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT,
                kind TEXT,
                status TEXT,
                started_at TEXT,
                ended_at TEXT,
                duration_ms REAL,
                attributes TEXT,
                events TEXT
            )
        """)


class Span:

    def __init__(
        self,
        trace,
        name,
        kind,
        parent_span_id=None,
        attributes=None,
    ):
        self.trace = trace
        self.span_id = new_id()
        self.trace_id = trace.trace_id
        self.parent_span_id = parent_span_id

        self.name = name
        self.kind = kind
        self.status = "running"

        self.started_at = utc_now()
        self.started = time.perf_counter()

        self.attributes = attributes or {}
        self.events = []

    def event(self, name, **attributes):
        self.events.append({
            "name": name,
            "timestamp": utc_now(),
            "attributes": attributes,
        })

    def set_attributes(self, **attributes):
        self.attributes.update(attributes)

    def end(self, status="success", **attributes):
        if self.status != "running":
            return

        self.set_attributes(**attributes)

        self.status = status
        self.ended_at = utc_now()

        self.duration_ms = round(
            (time.perf_counter() - self.started) * 1000,
            2,
        )

        self.trace.save_span(self)

    def to_dict(self):
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": getattr(
                self,
                "ended_at",
                None,
            ),
            "duration_ms": getattr(
                self,
                "duration_ms",
                None,
            ),
            "attributes": self.attributes,
            "events": self.events,
        }


class Trace:

    def __init__(self, query):
        init_db()

        self.trace_id = new_id()
        self.query = query
        self.answer = ""
        self.status = "running"

        self.started_at = utc_now()
        self.started = time.perf_counter()

        self.spans = []
        self.root_span = None

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT INTO traces (
                    trace_id,
                    query,
                    answer,
                    status,
                    started_at,
                    duration_ms
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.trace_id,
                query,
                "",
                "running",
                self.started_at,
                0,
            ))

    def start_span(
        self,
        name,
        kind,
        parent_span=None,
        attributes=None,
    ):
        parent_id = (
            parent_span.span_id
            if parent_span
            else None
        )

        span = Span(
            trace=self,
            name=name,
            kind=kind,
            parent_span_id=parent_id,
            attributes=attributes,
        )

        self.spans.append(span)

        return span

    @contextmanager
    def span(
        self,
        name,
        kind,
        parent_span=None,
        attributes=None,
    ):
        current = self.start_span(
            name=name,
            kind=kind,
            parent_span=parent_span,
            attributes=attributes,
        )

        try:
            yield current

        except Exception as exc:
            current.end(
                status="error",
                error=str(exc),
            )
            raise

        else:
            current.end()

    def save_span(self, span):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO spans (
                    span_id,
                    trace_id,
                    parent_span_id,
                    name,
                    kind,
                    status,
                    started_at,
                    ended_at,
                    duration_ms,
                    attributes,
                    events
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                span.span_id,
                span.trace_id,
                span.parent_span_id,
                span.name,
                span.kind,
                span.status,
                span.started_at,
                getattr(span, "ended_at", None),
                getattr(span, "duration_ms", None),
                json.dumps(span.attributes),
                json.dumps(span.events),
            ))

    def finish(self, answer, status="completed"):
        self.answer = answer
        self.status = status

        duration_ms = round(
            (time.perf_counter() - self.started) * 1000,
            2,
        )

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                UPDATE traces
                SET answer = ?,
                    status = ?,
                    duration_ms = ?
                WHERE trace_id = ?
            """, (
                answer,
                status,
                duration_ms,
                self.trace_id,
            ))

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "answer": self.answer,
            "status": self.status,
            "spans": [
                span.to_dict()
                for span in self.spans
            ],
        }
