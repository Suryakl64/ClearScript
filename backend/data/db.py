"""
SQLite report storage — persists uploaded reports and their findings.

Stores report sessions so users can upload multiple reports over time
and the chat/trend features can reference report history.

Database: backend/data/reports.db (auto-created on first use)
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.config import DB_PATH

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS reports (
    id              TEXT PRIMARY KEY,
    report_id       TEXT UNIQUE NOT NULL,
    filename        TEXT,
    upload_time     TEXT NOT NULL,
    report_type     TEXT,
    finding_count   INTEGER DEFAULT 0,
    findings_json   TEXT,
    summary         TEXT,
    status          TEXT DEFAULT 'processed'
);
"""


def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection (auto-creates DB + table)."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def generate_report_id(filename: str = "") -> str:
    """Generate a unique report ID."""
    short = uuid.uuid4().hex[:10]
    return f"rpt_{short}"


def save_report(
    report_id: str,
    filename: str,
    report_type: str,
    findings: list[dict],
    summary: str = "",
) -> dict:
    """
    Save a processed report to SQLite.

    Parameters
    ----------
    report_id : str
        Unique report identifier.
    filename : str
        Original uploaded filename.
    report_type : str
        Detected report type (structured/narrative/mixed).
    findings : list[dict]
        Structured findings from Phase 2.
    summary : str
        Report summary from Phase 3.

    Returns
    -------
    dict
        Confirmation with report_id and timestamp.
    """
    conn = _get_conn()
    now = datetime.now(timezone.utc).isoformat()

    try:
        conn.execute(
            """INSERT OR REPLACE INTO reports
               (id, report_id, filename, upload_time, report_type,
                finding_count, findings_json, summary, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report_id,
                report_id,
                filename,
                now,
                report_type,
                len(findings),
                json.dumps(findings, default=str),
                summary,
                "processed",
            ),
        )
        conn.commit()
        logger.info("Saved report %s (%d findings)", report_id, len(findings))
        return {
            "report_id": report_id,
            "filename": filename,
            "finding_count": len(findings),
            "upload_time": now,
            "status": "saved",
        }
    except Exception as exc:
        logger.error("Failed to save report: %s", exc)
        return {"report_id": report_id, "error": str(exc)}
    finally:
        conn.close()


def get_report(report_id: str) -> Optional[dict]:
    """Retrieve a report by its ID."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        if not row:
            return None

        data = dict(row)
        # Parse findings JSON back to list
        if data.get("findings_json"):
            data["findings"] = json.loads(data["findings_json"])
        else:
            data["findings"] = []
        return data
    finally:
        conn.close()


def list_reports(limit: int = 50) -> list[dict]:
    """List all stored reports, newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT report_id, filename, upload_time, report_type, "
            "finding_count, status FROM reports "
            "ORDER BY upload_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_report(report_id: str) -> bool:
    """Delete a report by ID."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "DELETE FROM reports WHERE report_id = ?", (report_id,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
