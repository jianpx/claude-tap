"""Tests for the ``claude-tap query`` subcommand.

Unit tests call ``query_main`` in-process and assert on the JSON it prints to
stdout. The e2e test seeds a real database, starts the dashboard web server,
fetches every ``/api/*`` route, runs the matching ``claude-tap query`` verb as a
subprocess against the same database, and asserts the CLI and web return
identical data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from claude_tap.live import LiveViewerServer
from claude_tap.query import query_main
from claude_tap.trace_store import get_trace_store

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_record(
    *,
    client: str,
    turn: int,
    user_text: str,
    response_text: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_create: int = 0,
    timestamp: str = "2026-07-08T08:00:00+00:00",
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "request_id": request_id or f"req_{client}_{turn}",
        "turn": turn,
        "duration_ms": 1200 + turn,
        "capture": {"client": client, "proxy_mode": "reverse"},
        "request": {
            "method": "POST",
            "path": "/v1/messages",
            "headers": {"Host": "api.anthropic.com"},
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": user_text}],
            },
        },
        "response": {
            "status": 200,
            "headers": {"content-type": "application/json"},
            "body": {
                "model": model,
                "content": [{"type": "text", "text": response_text}],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": cache_create,
                },
            },
        },
    }


def _seed_store(store) -> dict[str, str]:
    """Create two complete sessions (claude + codex) and return their ids."""
    # Pin the session start to 2026-07-08 so date_key matches the record
    # timestamps below; create_session derives date_key from started_at, not
    # from the records, so without this the sessions would bucket onto today.
    session_started_at = datetime(2026, 7, 8, 8, 0, tzinfo=timezone.utc)
    claude_id = store.create_session(
        client="claude",
        proxy_mode="reverse",
        started_at=session_started_at,
    )
    store.append_record(
        claude_id,
        _make_record(
            client="claude",
            turn=1,
            user_text="Explain this repository",
            response_text="This is a trace viewer for agents.",
            model="claude-sonnet-4-6",
            input_tokens=42,
            output_tokens=9,
            cache_read=100,
        ),
    )
    store.append_record(
        claude_id,
        _make_record(
            client="claude",
            turn=2,
            user_text="Show me the tools",
            response_text="Here are the registered tools.",
            model="claude-sonnet-4-6",
            input_tokens=60,
            output_tokens=12,
            timestamp="2026-07-08T08:01:00+00:00",
            cache_create=50,
        ),
    )
    store.finalize_session(claude_id)
    store.append_log(claude_id, "10:00:00 proxy started", level="INFO")

    codex_id = store.create_session(
        client="codex",
        proxy_mode="reverse",
        started_at=session_started_at,
    )
    store.append_record(
        codex_id,
        _make_record(
            client="codex",
            turn=1,
            user_text="Refactor the parser",
            response_text="I will refactor the parser module.",
            model="gpt-5.3-codex",
            input_tokens=200,
            output_tokens=30,
        ),
    )
    store.finalize_session(codex_id)

    return {"claude": claude_id, "codex": codex_id}


def _run_query(argv: list[str], capsys) -> tuple[int, Any]:
    """Run query_main in-process and return (exit_code, parsed_json_or_None)."""
    code = query_main(argv)
    out = capsys.readouterr().out
    try:
        return code, json.loads(out)
    except json.JSONDecodeError:
        return code, out


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #


def test_query_dates(trace_db, capsys) -> None:
    store = get_trace_store()
    _seed_store(store)

    code, payload = _run_query(["dates"], capsys)
    assert code == 0
    assert "2026-07-08" in payload["dates"]
    assert payload["has_legacy"] is False


def test_query_agents(trace_db, capsys) -> None:
    store = get_trace_store()
    _seed_store(store)

    code, payload = _run_query(["agents"], capsys)
    assert code == 0
    labels = {bucket["label"] for bucket in payload["agents"]}
    assert "Claude Code" in labels
    assert "Codex" in labels
    claude_bucket = next(b for b in payload["agents"] if b["label"] == "Claude Code")
    assert claude_bucket["sessions"] == 1
    assert claude_bucket["records"] == 2


def test_query_sessions_default_payload_shape(trace_db, capsys) -> None:
    store = get_trace_store()
    ids = _seed_store(store)

    code, payload = _run_query(["sessions"], capsys)
    assert code == 0
    assert payload["total"] == 2
    assert payload["total_records"] == 3
    assert payload["total_tokens"] == 42 + 9 + 100 + 60 + 12 + 50 + 200 + 30
    assert payload["offset"] == 0
    assert payload["limit"] == 100
    assert payload["has_more"] is False
    assert {s["id"] for s in payload["sessions"]} == {ids["claude"], ids["codex"]}
    # Newest activity first; the codex session was created after the claude one.
    assert payload["sessions"][0]["id"] == ids["codex"]


def test_query_sessions_search_filter(trace_db, capsys) -> None:
    store = get_trace_store()
    ids = _seed_store(store)

    code, payload = _run_query(["sessions", "--search", "Refactor"], capsys)
    assert code == 0
    assert [s["id"] for s in payload["sessions"]] == [ids["codex"]]
    assert payload["total"] == 1


def test_query_sessions_agent_filter(trace_db, capsys) -> None:
    store = get_trace_store()
    ids = _seed_store(store)

    code, payload = _run_query(["sessions", "--agent", "claude-code"], capsys)
    assert code == 0
    assert [s["id"] for s in payload["sessions"]] == [ids["claude"]]
    assert payload["total"] == 1


def test_query_sessions_status_filter(trace_db, capsys) -> None:
    store = get_trace_store()
    _seed_store(store)

    code, payload = _run_query(["sessions", "--status", "error"], capsys)
    assert code == 0
    assert payload["sessions"] == []
    assert payload["total"] == 0


def test_query_sessions_limit_and_offset(trace_db, capsys) -> None:
    store = get_trace_store()
    _seed_store(store)

    code, payload = _run_query(["sessions", "--limit", "1", "--offset", "1"], capsys)
    assert code == 0
    assert payload["limit"] == 1
    assert payload["offset"] == 1
    assert len(payload["sessions"]) == 1
    assert payload["has_more"] is False

    code, payload = _run_query(["sessions", "--limit", "1"], capsys)
    assert payload["has_more"] is True
    assert payload["total"] == 2


def test_query_sessions_limit_clamps_to_max(trace_db, capsys) -> None:
    store = get_trace_store()
    _seed_store(store)

    code, payload = _run_query(["sessions", "--limit", "99999"], capsys)
    assert code == 0
    assert payload["limit"] == 500


def test_query_records_returns_session_and_records(trace_db, capsys) -> None:
    store = get_trace_store()
    ids = _seed_store(store)

    code, payload = _run_query(["records", ids["claude"]], capsys)
    assert code == 0
    assert payload["session"]["id"] == ids["claude"]
    assert payload["session"]["record_count"] == 2
    assert [r["turn"] for r in payload["records"]] == [1, 2]
    assert payload["records"][0]["request_id"] == "req_claude_1"


def test_query_records_pagination(trace_db, capsys) -> None:
    store = get_trace_store()
    ids = _seed_store(store)

    code, payload = _run_query(["records", ids["claude"], "--limit", "1", "--offset", "1"], capsys)
    assert code == 0
    assert payload["session"]["record_count"] == 2
    assert [r["turn"] for r in payload["records"]] == [2]


def test_query_records_not_found_exits_nonzero(trace_db, capsys) -> None:
    store = get_trace_store()
    _seed_store(store)

    code = query_main(["records", "no-such-session"])
    assert code != 0
    assert "session not found" in capsys.readouterr().err


def test_query_traces_by_date(trace_db, capsys) -> None:
    store = get_trace_store()
    _seed_store(store)

    code, payload = _run_query(["traces", "2026-07-08"], capsys)
    assert code == 0
    assert isinstance(payload, list)
    assert len(payload) == 3
    assert {r["request_id"] for r in payload} == {"req_claude_1", "req_claude_2", "req_codex_1"}


def test_query_traces_invalid_date(trace_db, capsys) -> None:
    store = get_trace_store()
    _seed_store(store)

    code = query_main(["traces", "not-a-date"])
    assert code != 0
    assert "invalid date format" in capsys.readouterr().err


def test_query_export_jsonl(trace_db, capsys) -> None:
    store = get_trace_store()
    ids = _seed_store(store)

    code = query_main(["export", ids["claude"], "--format", "jsonl"])
    assert code == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["request_id"] == "req_claude_1"
    assert json.loads(lines[1])["request_id"] == "req_claude_2"


def test_query_export_compact(trace_db, capsys) -> None:
    store = get_trace_store()
    ids = _seed_store(store)

    code = query_main(["export", ids["claude"], "--format", "compact"])
    assert code == 0
    out = capsys.readouterr().out
    bundle = json.loads(out)
    assert "__claude_tap_compact_trace__" in bundle
    assert "records" in bundle


def test_query_export_log(trace_db, capsys) -> None:
    store = get_trace_store()
    ids = _seed_store(store)

    code = query_main(["export", ids["claude"], "--format", "log"])
    assert code == 0
    assert "proxy started" in capsys.readouterr().out


def test_query_export_not_found(trace_db, capsys) -> None:
    store = get_trace_store()
    _seed_store(store)

    code = query_main(["export", "no-such-session", "--format", "jsonl"])
    assert code != 0
    assert "session not found" in capsys.readouterr().err


def test_query_db_flag_overrides_database(tmp_path, monkeypatch, capsys) -> None:
    # Seed into an explicit database path.
    override_db = tmp_path / "override.sqlite3"
    monkeypatch.setenv("CLOUDTAP_DB", str(override_db))
    from claude_tap.trace_store import reset_trace_store

    reset_trace_store()
    store = get_trace_store()
    ids = _seed_store(store)

    # Repoint CLOUDTAP_DB elsewhere so the default store no longer sees the data.
    monkeypatch.setenv("CLOUDTAP_DB", str(tmp_path / "ignored.sqlite3"))
    reset_trace_store()

    # --db must win over CLOUDTAP_DB.
    code, payload = _run_query(["--db", str(override_db), "sessions"], capsys)
    assert code == 0
    assert payload["total"] == 2
    assert {s["id"] for s in payload["sessions"]} == {ids["claude"], ids["codex"]}


def test_query_requires_verb(capsys) -> None:
    with pytest.raises(SystemExit):
        query_main([])


# --------------------------------------------------------------------------- #
# E2E: web server vs CLI subprocess consistency
# --------------------------------------------------------------------------- #


def _run_query_subprocess(db_path: Path, *args: str) -> tuple[int, str, str]:
    """Run `claude-tap query ...` as a subprocess against db_path."""
    env = os.environ.copy()
    env["CLOUDTAP_DB"] = str(db_path)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Avoid network/update side effects in the subprocess.
    env["CLAUDE_TAP_PYPI_URL"] = "http://127.0.0.1:1/invalid"
    proc = subprocess.run(
        [sys.executable, "-m", "claude_tap", "query", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout, proc.stderr


@pytest.mark.asyncio
async def test_query_cli_matches_dashboard_web_api(trace_db, tmp_path: Path) -> None:
    store = get_trace_store()
    ids = _seed_store(store)
    db_path = trace_db

    server = LiveViewerServer(port=0, dashboard_mode=True)
    port = await server.start()
    base = f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            # /api/dates
            async with session.get(f"{base}/api/dates") as resp:
                web_dates = await resp.json()
            rc, out, _ = _run_query_subprocess(db_path, "dates")
            assert rc == 0, out
            assert json.loads(out) == web_dates

            # /api/agents
            async with session.get(f"{base}/api/agents") as resp:
                web_agents = await resp.json()
            rc, out, _ = _run_query_subprocess(db_path, "agents")
            assert rc == 0, out
            assert json.loads(out) == web_agents

            # /api/sessions (default)
            async with session.get(f"{base}/api/sessions") as resp:
                web_sessions = await resp.json()
            rc, out, _ = _run_query_subprocess(db_path, "sessions")
            assert rc == 0, out
            assert json.loads(out) == web_sessions

            # /api/sessions?search=Refactor
            async with session.get(f"{base}/api/sessions?search=Refactor") as resp:
                web_search = await resp.json()
            rc, out, _ = _run_query_subprocess(db_path, "sessions", "--search", "Refactor")
            assert rc == 0, out
            assert json.loads(out) == web_search

            # /api/sessions?agent=claude-code
            async with session.get(f"{base}/api/sessions?agent=claude-code") as resp:
                web_agent = await resp.json()
            rc, out, _ = _run_query_subprocess(db_path, "sessions", "--agent", "claude-code")
            assert rc == 0, out
            assert json.loads(out) == web_agent

            # /api/sessions?limit=1&offset=1
            async with session.get(f"{base}/api/sessions?limit=1&offset=1") as resp:
                web_paged = await resp.json()
            rc, out, _ = _run_query_subprocess(db_path, "sessions", "--limit", "1", "--offset", "1")
            assert rc == 0, out
            assert json.loads(out) == web_paged

            # /api/sessions/{id}/records
            async with session.get(f"{base}/api/sessions/{ids['claude']}/records") as resp:
                web_records = await resp.json()
            rc, out, _ = _run_query_subprocess(db_path, "records", ids["claude"])
            assert rc == 0, out
            assert json.loads(out) == web_records

            # /api/sessions/{id}/records?limit=1&offset=1
            async with session.get(f"{base}/api/sessions/{ids['claude']}/records?limit=1&offset=1") as resp:
                web_records_paged = await resp.json()
            rc, out, _ = _run_query_subprocess(db_path, "records", ids["claude"], "--limit", "1", "--offset", "1")
            assert rc == 0, out
            assert json.loads(out) == web_records_paged

            # /api/traces/{date}
            async with session.get(f"{base}/api/traces/2026-07-08") as resp:
                web_traces = await resp.json()
            rc, out, _ = _run_query_subprocess(db_path, "traces", "2026-07-08")
            assert rc == 0, out
            assert json.loads(out) == web_traces

            # /api/sessions/{id}/export/jsonl (raw body compared as text)
            async with session.get(f"{base}/api/sessions/{ids['claude']}/export/jsonl") as resp:
                web_jsonl = await resp.text()
            rc, out, _ = _run_query_subprocess(db_path, "export", ids["claude"], "--format", "jsonl")
            assert rc == 0, out
            assert out == web_jsonl

            # /api/sessions/{id}/export/compact (raw body compared as text)
            async with session.get(f"{base}/api/sessions/{ids['claude']}/export/compact") as resp:
                web_compact = await resp.text()
            rc, out, _ = _run_query_subprocess(db_path, "export", ids["claude"], "--format", "compact")
            assert rc == 0, out
            assert out == web_compact

            # /api/sessions/{id}/export/log (raw body compared as text)
            async with session.get(f"{base}/api/sessions/{ids['claude']}/export/log") as resp:
                web_log = await resp.text()
            rc, out, _ = _run_query_subprocess(db_path, "export", ids["claude"], "--format", "log")
            assert rc == 0, out
            assert out == web_log
    finally:
        await server.stop()
