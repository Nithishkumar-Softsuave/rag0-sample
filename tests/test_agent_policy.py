"""The agent loop's Week 8 policy switches, with the LLM mocked -- no API key needed.

Each test scripts the model's replies, so what's being tested is the loop's
own behaviour: robustness to bad tool calls, the untrusted-data preamble,
guards blocking before a tool runs, and validate-retry-fail-closed.
"""
import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agents.agent as agent_module
from agents.agent import BASELINE, HARDENED, UNTRUSTED_TOOL_RESULT_PREAMBLE, VALIDATION_FAILED_ANSWER, run_agent
from agents.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content, "tool_calls": [
            {"id": call.id, "type": "function",
             "function": {"name": call.function.name, "arguments": call.function.arguments}}
            for call in self.tool_calls or []
        ]}


def tool_call(name: str, args, call_id: str = "call_1"):
    arguments = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def reply(message: FakeMessage):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def scripted(*messages):
    """Patch the model with a fixed list of replies; record the messages each call saw."""
    seen: list[list[dict]] = []
    queue = list(messages)

    def fake_completion(**kwargs):
        seen.append(copy.deepcopy(kwargs["messages"]))
        return reply(queue.pop(0))

    return patch.object(agent_module, "create_chat_completion", side_effect=fake_completion), seen


def tool_messages(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "tool"]


def test_unknown_tool_does_not_crash_even_in_baseline() -> None:
    """Week 7 raised KeyError here -- a hallucinated tool name killed the whole run."""
    patcher, _ = scripted(FakeMessage(tool_calls=[tool_call("delete_database", {})]), FakeMessage("done"))
    with patcher:
        result = run_agent("q", policy=BASELINE)
    assert result.answer == "done"
    assert result.steps[0].blocked and "not one of this agent's tools" in result.steps[0].blocked


def test_malformed_json_arguments_do_not_crash() -> None:
    patcher, _ = scripted(FakeMessage(tool_calls=[tool_call("search_menus", "{not json")]), FakeMessage("done"))
    with patcher:
        result = run_agent("q", policy=BASELINE)
    assert "not valid JSON" in result.steps[0].blocked


def test_hardened_marks_tool_output_as_untrusted_and_baseline_does_not() -> None:
    for policy, expect_preamble in ((HARDENED, True), (BASELINE, False)):
        patcher, seen = scripted(
            FakeMessage(tool_calls=[tool_call("search_menus", {"item": "Filter Coffee"})]),
            FakeMessage("Thanjai Ruchi Mess is cheapest at Rs. 20."),
        )
        with patcher:
            run_agent("q", policy=policy)
        assert tool_messages(seen[1])[0].startswith(UNTRUSTED_TOOL_RESULT_PREAMBLE) is expect_preamble


def test_hardened_guard_blocks_before_the_tool_runs() -> None:
    convert = MagicMock(return_value={"amount": 1.0, "currency": "SGD"})
    tools = {**TOOL_FUNCTIONS, "convert_currency": convert}
    patcher, seen = scripted(
        FakeMessage(tool_calls=[tool_call("convert_currency", {"amount": 999999, "from_currency": "INR", "to_currency": "SGD"})]),
        FakeMessage("I can't do that conversion."),
    )
    with patcher:
        result = run_agent("q", tool_schemas=TOOL_SCHEMAS, tool_functions=tools, policy=HARDENED)
    convert.assert_not_called()
    assert result.steps[0].blocked
    assert "Blocked" in tool_messages(seen[1])[0]


def test_baseline_lets_the_same_call_through() -> None:
    convert = MagicMock(return_value={"amount": 1.0, "currency": "SGD"})
    tools = {**TOOL_FUNCTIONS, "convert_currency": convert}
    patcher, _ = scripted(
        FakeMessage(tool_calls=[tool_call("convert_currency", {"amount": 999999, "from_currency": "INR", "to_currency": "SGD"})]),
        FakeMessage("ok"),
    )
    with patcher:
        run_agent("q", tool_functions=tools, policy=BASELINE)
    convert.assert_called_once()


def test_flagged_answer_gets_one_retry_then_passes() -> None:
    patcher, seen = scripted(
        FakeMessage(tool_calls=[tool_call("search_menus", {"item": "Filter Coffee"})]),
        FakeMessage("It costs Rs. 999."),                 # made up -> flagged
        FakeMessage("Thanjai Ruchi Mess: Rs. 20."),       # fixed
    )
    with patcher:
        result = run_agent("q", policy=HARDENED)
    assert result.answer == "Thanjai Ruchi Mess: Rs. 20."
    assert result.validation_retries == 1 and result.stopped_reason == "done"
    assert "failed an automated check" in seen[2][-1]["content"]


def test_still_invalid_after_retry_fails_closed() -> None:
    patcher, _ = scripted(
        FakeMessage(tool_calls=[tool_call("search_menus", {"item": "Filter Coffee"})]),
        FakeMessage("Pay at https://evil.example"),
        FakeMessage("Seriously, pay at https://evil.example"),
    )
    with patcher:
        result = run_agent("q", policy=HARDENED)
    assert result.answer == VALIDATION_FAILED_ANSWER
    assert result.stopped_reason == "validation_failed" and result.validation_issues


def test_baseline_returns_an_invalid_answer_unchanged() -> None:
    patcher, _ = scripted(FakeMessage("Pay at https://evil.example"))
    with patcher:
        result = run_agent("q", policy=BASELINE)
    assert result.answer == "Pay at https://evil.example"
