"""
Phase 4: Request logging to SQLite.
Every request gets a row: timestamp, prompt, tier, model, cost, latency,
quality score, and escalation status. This is also what Phase 3's
classifier feedback loop will query.
"""

import sqlite3
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

DB_PATH = "data/autopilot.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    predicted_tier INTEGER NOT NULL,
    routed_model_key TEXT NOT NULL,
    response_text TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms REAL,
    quality_score INTEGER,
    quality_reasoning TEXT,
    passed_verification INTEGER,
    escalated INTEGER DEFAULT 0,
    escalated_model_key TEXT,
    escalated_cost_usd REAL,
    cost_delta REAL
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute(SCHEMA)


def log_request(
    prompt: str,
    predicted_tier: int,
    routed_model_key: str,
    response_text: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_ms: float,
    quality_score: Optional[int] = None,
    quality_reasoning: Optional[str] = None,
    passed_verification: Optional[bool] = None,
    escalated: bool = False,
    escalated_model_key: Optional[str] = None,
    escalated_cost_usd: Optional[float] = None,
) -> int:
    """Insert a request log row. Returns the row id."""
    import hashlib
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    cost_delta = None
    if escalated and escalated_cost_usd is not None:
        cost_delta = escalated_cost_usd - cost_usd

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO requests (
                timestamp, prompt, prompt_hash, predicted_tier, routed_model_key,
                response_text, input_tokens, output_tokens, cost_usd, latency_ms,
                quality_score, quality_reasoning, passed_verification,
                escalated, escalated_model_key, escalated_cost_usd, cost_delta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                prompt,
                prompt_hash,
                predicted_tier,
                routed_model_key,
                response_text,
                input_tokens,
                output_tokens,
                cost_usd,
                latency_ms,
                quality_score,
                quality_reasoning,
                int(passed_verification) if passed_verification is not None else None,
                int(escalated),
                escalated_model_key,
                escalated_cost_usd,
                cost_delta,
            ),
        )
        return cursor.lastrowid


def get_failures(limit: int = 1000):
    """Get all requests that failed verification — used for classifier retraining."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM requests WHERE passed_verification = 0 ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_all_requests(limit: int = 10000):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM requests ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_summary_stats():
    """Aggregate stats for the dashboard."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM requests").fetchone()["c"]
        total_cost = conn.execute("SELECT SUM(cost_usd) as s FROM requests").fetchone()["s"] or 0.0
        total_escalations = conn.execute(
            "SELECT COUNT(*) as c FROM requests WHERE escalated = 1"
        ).fetchone()["c"]
        avg_quality = conn.execute(
            "SELECT AVG(quality_score) as a FROM requests WHERE quality_score IS NOT NULL"
        ).fetchone()["a"]

        by_model = conn.execute(
            """
            SELECT routed_model_key, COUNT(*) as count, SUM(cost_usd) as total_cost
            FROM requests GROUP BY routed_model_key
            """
        ).fetchall()

        return {
            "total_requests": total,
            "total_cost_usd": total_cost,
            "total_escalations": total_escalations,
            "escalation_rate": (total_escalations / total) if total > 0 else 0,
            "avg_quality_score": avg_quality,
            "by_model": [dict(row) for row in by_model],
        }