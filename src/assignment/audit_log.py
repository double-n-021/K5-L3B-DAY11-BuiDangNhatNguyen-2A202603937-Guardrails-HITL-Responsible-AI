"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def _sanitize_for_log(text: str) -> str:
    """Redact secrets and PII before persisting text to audit logs."""
    if not text:
        return ""
    return content_filter(text)["redacted"]


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending_inputs: dict[str, str] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store sanitized input + start timestamp keyed by request_id/user_id."""
        key = request_id or user_id
        self._open[key] = time.perf_counter()
        self._pending_inputs[key] = _sanitize_for_log(text)[:300]

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store sanitized output, layer decision, latency; append to self.logs."""
        key = request_id or user_id
        start = self._open.pop(key, None)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2) if start is not None else None
        input_preview = self._pending_inputs.pop(key, "")

        self.logs.append(
            {
                "timestamp": utc_now_iso(),
                "request_id": key,
                "user_id": user_id,
                "input_preview": input_preview,
                "output_preview": _sanitize_for_log(text)[:300],
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": latency_ms,
            }
        )

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
