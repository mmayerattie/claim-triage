"""SQLite persistence for triage reports and agent traces.

A connection is opened per operation: FastAPI runs sync handlers on a thread
pool, and sqlite3 connections must not be shared across threads.
"""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    claim_id   TEXT PRIMARY KEY,
    report     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE TABLE IF NOT EXISTS traces (
    claim_id   TEXT PRIMARY KEY,
    trace      TEXT NOT NULL
);
"""


class ClaimStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def save(self, claim_id: str, report: dict, trace: dict) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO reports (claim_id, report) VALUES (?, ?)",
                         (claim_id, json.dumps(report)))
            conn.execute("INSERT OR REPLACE INTO traces (claim_id, trace) VALUES (?, ?)",
                         (claim_id, json.dumps(trace)))

    def get_report(self, claim_id: str) -> dict | None:
        return self._get("SELECT report FROM reports WHERE claim_id = ?", claim_id)

    def get_trace(self, claim_id: str) -> dict | None:
        return self._get("SELECT trace FROM traces WHERE claim_id = ?", claim_id)

    def _get(self, sql: str, claim_id: str) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(sql, (claim_id,)).fetchone()
        return json.loads(row[0]) if row else None
