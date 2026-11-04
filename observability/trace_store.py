import json
import os
import sqlite3
from pathlib import Path

DB_PATH = os.getenv("TRACE_DB_PATH", str(Path(__file__).resolve().parents[1] / "traces.db"))

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS runs (
            trace_id TEXT PRIMARY KEY, query TEXT, original_answer TEXT, final_answer TEXT,
            was_refreshed INTEGER, time_sensitive INTEGER, verification_status TEXT,
            started_at TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS spans (
            span_id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, agent_name TEXT,
            status TEXT, started_at TEXT, duration_ms REAL, input_json TEXT,
            output_json TEXT, error TEXT, FOREIGN KEY(trace_id) REFERENCES runs(trace_id))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, title TEXT, url TEXT,
            published_date TEXT, snippet TEXT, score REAL,
            FOREIGN KEY(trace_id) REFERENCES runs(trace_id))""")
    migrate_db()

def migrate_db():
    """Add missing columns to older local DBs without dropping saved trace history."""
    required = {
        "parent_id": "TEXT", "agent_name": "TEXT", "status": "TEXT",
        "started_at": "TEXT", "duration_ms": "REAL", "input_json": "TEXT",
        "output_json": "TEXT", "error": "TEXT",
    }
    with connect() as conn:
        exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='spans'").fetchone()
        if not exists:
            return
        current = {row["name"] for row in conn.execute("PRAGMA table_info(spans)")}
        for name, column_type in required.items():
            if name not in current:
                conn.execute(f"ALTER TABLE spans ADD COLUMN {name} {column_type}")

def save_run(result, query):
    init_db()
    trace = result.get("trace", {})
    freshness = result.get("freshness", {})
    verification = result.get("verification") or {}
    with connect() as conn:
        conn.execute("""INSERT OR REPLACE INTO runs
            (trace_id, query, original_answer, final_answer, was_refreshed, time_sensitive,
             verification_status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (result.get("trace_id"), query, result.get("original_answer"), result.get("answer"),
             int(bool(result.get("was_refreshed"))), int(bool(freshness.get("time_sensitive"))),
             verification.get("overall_status", freshness.get("verification_status")), trace.get("started_at")))
        for span in trace.get("spans", []):
            conn.execute("""INSERT OR REPLACE INTO spans
                (span_id, trace_id, parent_id, agent_name, status, started_at, duration_ms,
                 input_json, output_json, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (span.get("span_id"), result.get("trace_id"), span.get("parent_id"),
                 span.get("agent_name"), span.get("status"), span.get("started_at"),
                 span.get("duration_ms"), json.dumps(span.get("input"), ensure_ascii=False),
                 json.dumps(span.get("output"), ensure_ascii=False), span.get("error")))
        for item in result.get("evidence", {}).get("results", []):
            conn.execute("""INSERT INTO evidence
                (trace_id, title, url, published_date, snippet, score) VALUES (?, ?, ?, ?, ?, ?)""",
                (result.get("trace_id"), item.get("title"), item.get("url"),
                 item.get("published_date"), item.get("content"), item.get("score")))

def list_runs(limit=50):
    init_db()
    with connect() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()]

def get_run(trace_id):
    init_db()
    with connect() as conn:
        run = conn.execute("SELECT * FROM runs WHERE trace_id=?", (trace_id,)).fetchone()
        if not run:
            return None
        result = dict(run)
        result["spans"] = [dict(row) for row in conn.execute(
            "SELECT * FROM spans WHERE trace_id=? ORDER BY started_at", (trace_id,)).fetchall()]
        result["evidence"] = [dict(row) for row in conn.execute(
            "SELECT * FROM evidence WHERE trace_id=?", (trace_id,)).fetchall()]
        return result
