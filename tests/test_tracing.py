"""Week 11: per-request traces and the support tools over them -- no API key needed."""
import contextvars
import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agents.agent as agent_module
from agents.kitchen import ask
from observability.logs import filter_traces, format_trace, parse_when, stats
from observability.tracing import MAX_FIELD_CHARS, clip, read_traces, request, span
from rag_chat.client import record_usage
from tests.test_agent_policy import FakeMessage, tool_call
from tests.test_squad import QUESTION, both_servers, fake_team

TOKENS = SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110)


def _reply(message: FakeMessage):
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=TOKENS)


# -- the tracer ----------------------------------------------------------

def test_span_outside_a_request_records_nothing(tmp_path) -> None:
    with span("llm", kind="llm") as item:
        assert item is None
    assert not (tmp_path / "t.jsonl").exists()


def test_one_line_per_request_with_nested_spans_and_their_own_cost(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    with request("q", route="single", user_id="u1", path=path) as trace:
        with span("agent", kind="agent"):
            with span("llm", kind="llm"):
                record_usage(TOKENS, "openai/gpt-oss-120b")
            with span("tool:search_menus", kind="tool", args={"item": "Idli"}) as tool:
                tool.set(result={"matches": []})
            with span("llm", kind="llm"):
                record_usage(TOKENS, "openai/gpt-oss-120b")
        trace.answer, trace.outcome = "a", "done"
    [logged] = read_traces(path)
    assert logged["request_id"].startswith("req_") and logged["user_id"] == "u1"
    assert logged["llm_calls"] == 2 and logged["prompt_tokens"] == 200 and logged["cost_usd"] > 0
    by_name = {}
    for item in logged["spans"]:
        by_name.setdefault(item["name"], []).append(item)
    [root], [agent], [tool] = by_name["request"], by_name["agent"], by_name["tool:search_menus"]
    assert agent["parent_id"] == root["id"] and tool["parent_id"] == agent["id"]
    assert all(s["parent_id"] == agent["id"] and s["llm_calls"] == 1 for s in by_name["llm"])
    assert agent["cost_usd"] == pytest.approx(sum(s["cost_usd"] for s in by_name["llm"]))
    assert tool["attrs"]["args"] == {"item": "Idli"} and tool["llm_calls"] == 0


def test_a_request_that_crashes_is_still_logged(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    with pytest.raises(RuntimeError):
        with request("q", route="single", path=path):
            with span("tool:x", kind="tool"):
                raise RuntimeError("server died")
    [logged] = read_traces(path)
    assert logged["outcome"] == "error" and "server died" in logged["attrs"]["error"]
    assert any(s["name"] == "tool:x" and "server died" in s["error"] for s in logged["spans"])


def test_worker_thread_spans_land_under_the_caller(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    with request("q", route="squad", path=path):
        with span("dispatch") as parent:
            def worker(n: int) -> None:
                with span(f"specialist:{n}", kind="agent"):
                    record_usage(TOKENS, "openai/gpt-oss-120b")
            threads = [threading.Thread(target=contextvars.copy_context().run, args=(worker, n)) for n in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    [logged] = read_traces(path)
    specialists = [s for s in logged["spans"] if s["name"].startswith("specialist:")]
    assert len(specialists) == 4 and all(s["parent_id"] == parent.id for s in specialists)
    assert logged["llm_calls"] == 4


def test_long_fields_are_clipped_and_a_torn_last_line_is_skipped(tmp_path) -> None:
    assert len(clip("x" * (MAX_FIELD_CHARS + 500))) < MAX_FIELD_CHARS + 50
    path = tmp_path / "t.jsonl"
    with request("q", route="single", path=path):
        pass
    with path.open("a", encoding="utf-8") as file:
        file.write('{"request_id": "req_torn", "times')  # crash mid-write
    assert len(read_traces(path)) == 1


# -- the app writes a trace for every answer -------------------------------

def test_kitchen_single_route_logs_every_step(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    replies = iter([
        _reply(FakeMessage(tool_calls=[tool_call("lookup_ingredients", {"dish": "Masala Dosa"})])),
        _reply(FakeMessage("Masala Dosa contains ghee, so it is not dairy-free.")),
    ])
    with both_servers() as host, patch.object(agent_module, "create_chat_completion", side_effect=lambda **_: next(replies)):
        reply = ask("Is Masala Dosa dairy-free?", host, user_id="u_9", trace_path=path)
    [logged] = read_traces(path)
    assert logged["request_id"] == reply.request_id and logged["answer"] == reply.answer
    assert logged["outcome"] == "done" and logged["attrs"]["tool_calls"] == ["lookup_ingredients"]
    assert set(logged["attrs"]["tools"]) == {"search_menus", "price_dishes", "convert_currency", "lookup_ingredients", "find_dishes"}
    names = [s["name"] for s in logged["spans"]]
    assert names == ["request", "agent", "llm", "tool:lookup_ingredients", "llm", "validate"]
    tool = next(s for s in logged["spans"] if s["kind"] == "tool")
    assert "ghee" in json.dumps(tool["attrs"]["result"])  # the evidence is in the log, not just the answer
    assert logged["cost_usd"] == pytest.approx(sum(s["cost_usd"] for s in logged["spans"] if s["kind"] == "llm"))


def test_kitchen_squad_route_logs_manager_and_specialists(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    good = "Masala Dosa is not dairy-free: it contains ghee. It is cheapest at Sri Kaveri Bhavan, Rs. 70."
    plan = '{"menu": "Cheapest Masala Dosa?", "ingredients": "Does Masala Dosa contain dairy?"}'
    patchers, _ = fake_team(plan, good)
    with both_servers() as host, patchers[0], patchers[1]:
        ask(QUESTION, host, route="squad", trace_path=path)
    [logged] = read_traces(path)
    names = {s["name"] for s in logged["spans"]}
    assert {"manager:plan", "specialist:menu", "specialist:ingredients", "manager:synthesize"} <= names
    assert logged["llm_calls"] == 6 and logged["attrs"]["plan"]["menu"] == "Cheapest Masala Dosa?"


# -- the support tools ------------------------------------------------------

def _trace(rid: str, question: str, answer: str, *, user: str = "u", hours_ago: float = 0, cost: float = 0.001,
           outcome: str = "done", tools: list[str] | None = None) -> dict:
    when = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {"request_id": rid, "timestamp": when.isoformat(), "route": "single", "question": question, "answer": answer,
            "user_id": user, "session_id": None, "outcome": outcome, "duration_ms": 2000, "cost_usd": cost,
            "llm_calls": 2, "prompt_tokens": 1000, "completion_tokens": 100, "rate_limit_wait_seconds": 0,
            "attrs": {"tool_calls": tools or []}, "spans": [
                {"id": "a", "parent_id": None, "name": "llm", "kind": "llm", "start_ms": 0, "duration_ms": 1500,
                 "llm_calls": 1, "prompt_tokens": 1000, "completion_tokens": 100, "cost_usd": cost,
                 "rate_limit_wait_seconds": 0, "attrs": {}, "error": None},
                {"id": "b", "parent_id": None, "name": "tool:find_dishes", "kind": "tool", "start_ms": 1500,
                 "duration_ms": 50, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0,
                 "rate_limit_wait_seconds": 0, "attrs": {"args": {"free_from": ["dairy"]}, "result": {}}, "error": None},
            ]}


def test_find_by_words_user_time_and_tool() -> None:
    traces = [
        _trace("req_1", "dairy-free swap for dosa?", "Try Plain Idli.", user="u1", hours_ago=50, tools=["find_dishes"]),
        _trace("req_2", "cheapest coffee", "Thanjai, Rs. 20.", user="u2", hours_ago=2),
        _trace("req_3", "Swap for Ven Pongal? no dairy", "Rava Kesari is dairy-free.", user="u3", hours_ago=1),
    ]
    assert [t["request_id"] for t in filter_traces(traces, words=["dairy"])] == ["req_3", "req_1"]  # newest first
    assert [t["request_id"] for t in filter_traces(traces, words=["swap", "dairy"], since=parse_when("1d"))] == ["req_3"]
    assert [t["request_id"] for t in filter_traces(traces, user="u2")] == ["req_2"]
    assert [t["request_id"] for t in filter_traces(traces, tool="find_dishes")] == ["req_1"]


def test_parse_when() -> None:
    now = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)
    assert parse_when("3d", now) == datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    assert parse_when("2026-10-01") == datetime(2026, 10, 1, tzinfo=timezone.utc)


def test_show_has_time_and_cost_per_step() -> None:
    text = format_trace(_trace("req_1", "q", "a", cost=0.0005))
    assert "llm" in text and "1.50s" in text and "$0.000500" in text
    assert "tool:find_dishes" in text and '"free_from": ["dairy"]' in text


def test_stats_per_request_and_per_step() -> None:
    result = stats([_trace("r1", "q", "a", cost=0.001), _trace("r2", "q", "a", cost=0.003)])
    assert result["requests"] == 2 and result["cost_per_request_mean"] == pytest.approx(0.002)
    assert result["by_step"]["llm"]["count"] == 2 and result["by_step"]["tool:find_dishes"]["cost"] == 0


def test_cached_prompt_tokens_are_billed_at_the_cached_rate() -> None:
    """Groq served 768 of most calls' prompt tokens from its cache; billing them at full price over-stated cost."""
    from rag_chat.client import usage_scope
    cached = SimpleNamespace(prompt_tokens=1000, completion_tokens=0, total_tokens=1000,
                             prompt_tokens_details=SimpleNamespace(cached_tokens=800))
    with usage_scope() as usage:
        record_usage(cached, "openai/gpt-oss-120b")
    assert usage.cached_prompt_tokens == 800
    assert usage.cost_usd == pytest.approx(200 * 0.15e-6 + 800 * 0.075e-6)


def test_reprice_and_compare_pair_runs_by_question() -> None:
    from observability.logs import compare, reprice
    old = _trace("r1", "swap for dosa?", "a", outcome="max_steps")
    old["spans"][0].update(prompt_tokens=10_000, completion_tokens=0, cost_usd=999.0)  # logged with an old price
    old["spans"][0]["attrs"] = {"model": "openai/gpt-oss-120b", "cached_tokens": 0}
    new = _trace("r2", "swap for dosa?", "b")
    new["spans"][0].update(prompt_tokens=2_000, completion_tokens=0)
    new["spans"][0]["attrs"] = {"model": "openai/gpt-oss-120b", "cached_tokens": 768}
    assert reprice(old)["cost_usd"] == pytest.approx(10_000 * 0.15e-6)
    result = compare([old, _trace("r3", "only before", "c")], [new])
    assert len(result["pairs"]) == 1
    assert result["before"]["answered"] == 0 and result["after"]["answered"] == 1
    assert result["after"]["cost_mean"] == pytest.approx(1232 * 0.15e-6 + 768 * 0.075e-6)


def test_promote_adds_a_regression_case_once(tmp_path) -> None:
    from observability.regressions import add_case, judge, load_cases
    path = tmp_path / "regressions.jsonl"
    failed = _trace("req_9", "cheapest vegan breakfast item?", "Stopped: exceeded step limit.", outcome="max_steps")
    add_case(failed, "max_steps", path)
    with pytest.raises(ValueError):
        add_case(failed, "again", path)
    [case] = load_cases(path)
    assert case["source_request_id"] == "req_9" and case["failed_outcome"] == "max_steps"
    assert judge(case, failed) == ["outcome max_steps"]
    fixed = _trace("req_10", case["question"], "Plain Idli at Vaigai Tiffin Center, Rs. 30. It is vegan.")
    assert judge(case, fixed) == []
