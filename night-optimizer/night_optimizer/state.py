from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import (
    AgentTaskRecord,
    AttemptRecord,
    ExecutionRecord,
    InsightRecord,
    to_plain_dict,
)


class StateStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    insight_id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_name, created_at)
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_insights_session ON insights(session_name, created_at)
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_session ON agent_tasks(session_name, created_at)
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_session ON executions(session_name, created_at)
                """)

    def upsert_attempt(self, attempt: AttemptRecord) -> None:
        payload = json.dumps(to_plain_dict(attempt), sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO attempts(attempt_id, session_name, status, created_at, updated_at, payload)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(attempt_id) DO UPDATE SET
                    session_name=excluded.session_name,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    payload=excluded.payload
                """,
                (
                    attempt.attempt_id,
                    attempt.session_name,
                    attempt.status.value,
                    attempt.created_at,
                    attempt.updated_at,
                    payload,
                ),
            )

    def upsert_insight(self, insight: InsightRecord) -> None:
        payload = json.dumps(to_plain_dict(insight), sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO insights(insight_id, session_name, status, created_at, payload)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(insight_id) DO UPDATE SET
                    session_name=excluded.session_name,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    payload=excluded.payload
                """,
                (
                    insight.insight_id,
                    insight.session_name,
                    insight.status.value,
                    insight.created_at,
                    payload,
                ),
            )

    def upsert_agent_task(self, task: AgentTaskRecord) -> None:
        payload = json.dumps(to_plain_dict(task), sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_tasks(task_id, session_name, created_at, payload)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    session_name=excluded.session_name,
                    created_at=excluded.created_at,
                    payload=excluded.payload
                """,
                (task.task_id, task.session_name, task.created_at, payload),
            )

    def upsert_execution(self, execution: ExecutionRecord) -> None:
        payload = json.dumps(to_plain_dict(execution), sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO executions(execution_id, session_name, attempt_id, created_at, payload)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    session_name=excluded.session_name,
                    attempt_id=excluded.attempt_id,
                    created_at=excluded.created_at,
                    payload=excluded.payload
                """,
                (
                    execution.execution_id,
                    execution.session_name,
                    execution.attempt_id,
                    execution.created_at,
                    payload,
                ),
            )

    def list_attempt_payloads(self, session_name: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM attempts WHERE session_name = ? ORDER BY created_at ASC",
                (session_name,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def list_validated_insight_payloads(self, session_name: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM insights
                WHERE session_name = ? AND status = 'validated'
                ORDER BY created_at ASC
                """,
                (session_name,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def list_agent_task_payloads(self, session_name: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM agent_tasks WHERE session_name = ? ORDER BY created_at ASC",
                (session_name,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def get_attempt_payload(self, attempt_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM attempts WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload"])
