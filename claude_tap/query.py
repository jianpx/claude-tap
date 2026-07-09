"""Read-only query subcommand mirroring the dashboard web API.

``claude-tap query`` lets agents consume the same trace data the browser
dashboard exposes, but as machine-readable JSON on stdout. Every verb maps 1:1
to a dashboard HTTP route and assembles the same payload shape, so the CLI and
the web return identical data for the same database.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from claude_tap.dashboard import (
    build_session_query,
    list_trace_agents,
    list_trace_sessions,
    load_trace_session,
)
from claude_tap.trace_store import get_trace_store, reset_trace_store

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEFAULT_SESSION_PAGE_LIMIT = 100
MAX_SESSION_PAGE_LIMIT = 500

_EXPORT_FORMATS = ("jsonl", "compact", "log")


def _print_json(obj: Any) -> None:
    """Write a JSON object to stdout with stable, web-compatible formatting."""
    sys.stdout.write(json.dumps(obj, indent=2, ensure_ascii=False))
    sys.stdout.write("\n")


def _fail(message: str) -> int:
    """Print an error to stderr and return a non-zero exit code."""
    print(f"Error: {message}", file=sys.stderr)
    return 1


def _parse_session_limit(value: int | None) -> int:
    """Clamp the sessions page limit to the dashboard range (default 100, max 500)."""
    if value is None:
        return DEFAULT_SESSION_PAGE_LIMIT
    return max(1, min(MAX_SESSION_PAGE_LIMIT, value))


def _parse_offset(value: int | None) -> int:
    """Normalize an offset argument (default 0, never negative)."""
    if value is None:
        return 0
    return max(0, value)


def _parse_record_limit(value: int | None) -> int | None:
    """Normalize a records limit (None means all, otherwise non-negative)."""
    if value is None:
        return None
    return max(0, value)


def _bootstrap_store(args: argparse.Namespace) -> None:
    """Apply an explicit --db override before any store access."""
    db_path: Path | None = getattr(args, "db", None)
    if db_path is not None:
        os.environ["CLOUDTAP_DB"] = str(db_path)
        reset_trace_store()


def _finalize_stale_active_sessions() -> None:
    """Release abandoned active sessions, matching the dashboard list endpoints."""
    # In dashboard mode the live session id is None, so nothing is protected.
    get_trace_store().finalize_stale_active_sessions(protected_session_ids=set())


def _run_dates(_args: argparse.Namespace) -> int:
    dates, has_legacy = get_trace_store().list_dates()
    _print_json({"dates": dates, "has_legacy": has_legacy})
    return 0


def _run_traces(args: argparse.Namespace) -> int:
    date_key: str = args.date
    if date_key != "legacy" and not _DATE_RE.match(date_key):
        return _fail(f"invalid date format: {date_key} (expected YYYY-MM-DD or 'legacy')")
    records = get_trace_store().load_records_for_date(date_key)
    _print_json(records)
    return 0


def _run_agents(_args: argparse.Namespace) -> int:
    _finalize_stale_active_sessions()
    _print_json({"agents": list_trace_agents(None, live_record_count=0)})
    return 0


def _run_sessions(args: argparse.Namespace) -> int:
    _finalize_stale_active_sessions()
    store = get_trace_store()
    query = build_session_query(
        date=args.date,
        status=args.status,
        search=args.search,
        agent=args.agent,
    )
    aggregates = store.get_session_aggregates(query)
    total = aggregates["total_sessions"]
    total_records = aggregates["total_records"]
    total_tokens = aggregates["total_tokens"]
    total_errors = aggregates["total_errors"]
    limit = _parse_session_limit(args.limit)
    offset = _parse_offset(args.offset)
    sessions = list_trace_sessions(None, live_record_count=0, limit=limit, offset=offset, query=query)
    dates, has_legacy = store.list_dates()
    _print_json(
        {
            "sessions": sessions,
            "total": total,
            "total_records": total_records,
            "total_tokens": total_tokens,
            "total_errors": total_errors,
            "offset": offset,
            "limit": limit,
            "has_more": offset + len(sessions) < total,
            "dates": dates,
            "has_legacy": has_legacy,
        }
    )
    return 0


def _run_records(args: argparse.Namespace) -> int:
    record_limit = _parse_record_limit(args.limit)
    record_offset = _parse_offset(args.offset)
    session = load_trace_session(
        args.session_id,
        current_session_id=None,
        record_limit=record_limit,
        record_offset=record_offset,
        live_record_count=0,
    )
    if session is None:
        return _fail(f"session not found: {args.session_id}")
    _print_json(session)
    return 0


def _run_export(args: argparse.Namespace) -> int:
    store = get_trace_store()
    if store.load_session_row(args.session_id) is None:
        return _fail(f"session not found: {args.session_id}")
    if args.format == "jsonl":
        body = store.export_jsonl(args.session_id)
    elif args.format == "compact":
        body = store.export_compact(args.session_id)
    else:
        body = store.export_log(args.session_id)
    # Write the raw export body verbatim so it matches the dashboard route byte-for-byte.
    sys.stdout.write(body)
    return 0


def parse_query_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the argument parser for the query subcommand."""
    parser = argparse.ArgumentParser(
        prog="claude-tap query",
        description=(
            "Query trace data from the local database. Mirrors the dashboard web API "
            "and prints JSON to stdout for agent consumption."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to the trace SQLite database (default: $CLOUDTAP_DB or the standard data dir).",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    sub.add_parser("dates", help="List available trace dates (GET /api/dates).")

    p_traces = sub.add_parser("traces", help="List all records for a date (GET /api/traces/{date}).")
    p_traces.add_argument("date", help="Date key: YYYY-MM-DD or 'legacy'.")

    sub.add_parser("agents", help="List agent buckets (GET /api/agents).")

    p_sessions = sub.add_parser("sessions", help="List trace sessions (GET /api/sessions).")
    p_sessions.add_argument(
        "--search", default="", help="Free-text search across session metadata and record payloads."
    )
    p_sessions.add_argument("--limit", type=int, default=None, help="Max sessions to return (default: 100, max: 500).")
    p_sessions.add_argument("--offset", type=int, default=None, help="Number of sessions to skip (default: 0).")
    p_sessions.add_argument("--date", default="", help="Filter by date key: YYYY-MM-DD or 'legacy'.")
    p_sessions.add_argument("--status", default="", help="Filter by status: active, complete, error, or empty.")
    p_sessions.add_argument("--agent", default="", help="Filter by agent key (e.g. claude-code, codex, gemini).")

    p_records = sub.add_parser("records", help="Show one session summary and records (GET /api/sessions/{id}/records).")
    p_records.add_argument("session_id", help="Session id.")
    p_records.add_argument("--limit", type=int, default=None, help="Max records to return (default: all).")
    p_records.add_argument("--offset", type=int, default=None, help="Number of records to skip (default: 0).")

    p_export = sub.add_parser("export", help="Export a session's raw trace (GET /api/sessions/{id}/export/{format}).")
    p_export.add_argument("session_id", help="Session id.")
    p_export.add_argument("--format", choices=_EXPORT_FORMATS, required=True, help="Export format.")

    return parser.parse_args(argv)


def query_main(argv: list[str] | None = None) -> int:
    """Entry point for the query subcommand."""
    args = parse_query_args(argv)
    _bootstrap_store(args)
    verb = args.verb
    if verb == "dates":
        return _run_dates(args)
    if verb == "traces":
        return _run_traces(args)
    if verb == "agents":
        return _run_agents(args)
    if verb == "sessions":
        return _run_sessions(args)
    if verb == "records":
        return _run_records(args)
    if verb == "export":
        return _run_export(args)
    return _fail(f"unknown verb: {verb}")
