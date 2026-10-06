"""Week 10: the kitchen squad and team-wide token accounting -- no API key needed.

The model is faked by *role* (which system prompt a call carries), not by a
fixed queue, so the same fake works when the two specialists run in
parallel threads and their calls interleave.
"""
import contextvars
import threading
from types import SimpleNamespace
from unittest.mock import patch

import agents.agent as agent_module
import agents.squad as squad_module
from agents.agent import VALIDATION_FAILED_ANSWER, run_agent
from agents.squad import INGREDIENT_SPECIALIST, MENU_SPECIALIST, SPECIALISTS, parse_plan, run_squad
from mcp_servers import ingredient_server, menu_server
from rag_chat.client import get_usage, record_usage, reset_usage, usage_scope
from tests.test_agent_policy import FakeMessage, tool_call
from tests.test_mcp import INGREDIENTS, MENU, host_for

TOKENS = SimpleNamespace(prompt_tokens=100, completion_tokens=10, total_tokens=110)
QUESTION = "I'm allergic to dairy. Is Masala Dosa safe for me, and where is it cheapest?"


def _reply(message: FakeMessage):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=TOKENS)


def fake_team(plan: str, synthesis: str | list[str], menu_tool: str = "search_menus"):
    """Patch the model for every role; returns (patchers, calls-per-role log)."""
    log: dict[str, int] = {}
    lock = threading.Lock()
    drafts = [synthesis] if isinstance(synthesis, str) else list(synthesis)

    def fake_completion(**kwargs):
        system = kwargs["messages"][0]["content"]
        has_tool_result = any(m["role"] == "tool" for m in kwargs["messages"])
        if system.startswith("You are the manager") and "split each question" in system:
            role, message = "planner", FakeMessage(plan)
        elif system == MENU_SPECIALIST.system_prompt:
            role = "menu"
            message = (FakeMessage("Masala Dosa is cheapest at Sri Kaveri Bhavan, Rs. 70.") if has_tool_result
                       else FakeMessage(tool_calls=[tool_call(menu_tool, {"item": "Masala Dosa"} if menu_tool == "search_menus" else {"dish": "Masala Dosa"})]))
        elif system == INGREDIENT_SPECIALIST.system_prompt:
            role = "ingredients"
            message = (FakeMessage("Masala Dosa contains dairy (ghee); standard recipe, confirm with the restaurant.")
                       if has_tool_result else FakeMessage(tool_calls=[tool_call("lookup_ingredients", {"dish": "Masala Dosa"})]))
        else:
            role, message = "synthesis", FakeMessage(drafts.pop(0) if len(drafts) > 1 else drafts[0])
        with lock:
            log[role] = log.get(role, 0) + 1
        return _reply(message)

    patchers = [
        patch.object(agent_module, "create_chat_completion", side_effect=fake_completion),
        patch.object(squad_module, "create_chat_completion", side_effect=fake_completion),
    ]
    return patchers, log


def both_servers():
    return host_for(MENU, INGREDIENTS, menu=menu_server.build_server(), ingredients=ingredient_server.build_server())


PLAN_BOTH = '{"menu": "Find the cheapest shop for Masala Dosa.", "ingredients": "Does Masala Dosa contain dairy?"}'
GOOD_ANSWER = ("Masala Dosa is not dairy-free: it contains ghee (standard recipe -- confirm with the restaurant). "
               "It is cheapest at Sri Kaveri Bhavan, Rs. 70.")


# -- usage accounting (the Week 10 counting bug) -------------------------

def test_nested_usage_scopes_roll_up_into_the_caller() -> None:
    """Before Week 10, run_agent() reset a global total -- a specialist erased its manager's tokens."""
    reset_usage()
    with usage_scope() as team:
        record_usage(TOKENS, "openai/gpt-4o-mini")          # manager
        with usage_scope() as specialist:
            record_usage(TOKENS, "openai/gpt-4o-mini")      # specialist
        assert specialist.calls == 1
    assert team.calls == 2 and team.total_tokens == 220
    assert team.cost_usd > specialist.cost_usd > 0
    assert get_usage().calls == 2  # and the top-level total still saw both


def test_usage_from_worker_threads_counts_toward_the_parent() -> None:
    with usage_scope() as team:
        def worker() -> None:
            with usage_scope():
                record_usage(TOKENS, "openai/gpt-4o-mini")
        threads = [threading.Thread(target=contextvars.copy_context().run, args=(worker,)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert team.calls == 8 and team.prompt_tokens == 800


def test_active_groq_model_has_a_price() -> None:
    """Weeks 7-9 reported $0.00 for every run: gpt-oss-120b was missing from the price table."""
    with usage_scope() as usage:
        record_usage(TOKENS, "openai/gpt-oss-120b")
    assert usage.cost_usd > 0


def test_run_agent_reports_its_own_usage_and_keeps_the_callers() -> None:
    patcher = patch.object(agent_module, "create_chat_completion", return_value=_reply(FakeMessage("No prices needed.")))
    with patcher, usage_scope() as outer:
        record_usage(TOKENS, "openai/gpt-4o-mini")
        result = run_agent("q")
    assert result.llm_calls == 1 and result.prompt_tokens == 100
    assert outer.calls == 2


# -- the manager's plan --------------------------------------------------

def test_parse_plan_reads_json_and_nulls() -> None:
    plan = parse_plan('Sure! {"menu": "Price of Idli", "ingredients": null}', SPECIALISTS, "q")
    assert plan == {"menu": "Price of Idli", "ingredients": None}


def test_unparseable_plan_sends_the_whole_question_to_everyone() -> None:
    assert parse_plan("I think the menu one?", SPECIALISTS, "the question") == {
        "menu": "the question", "ingredients": "the question",
    }


# -- the squad end to end ------------------------------------------------

def test_squad_splits_dispatches_in_parallel_and_counts_every_call() -> None:
    patchers, log = fake_team(PLAN_BOTH, GOOD_ANSWER)
    with both_servers() as host, patchers[0], patchers[1]:
        result = run_squad(QUESTION, host)
    assert result.stopped_reason == "done" and result.answer == GOOD_ANSWER
    assert [a.specialist for a in result.assignments] == ["menu", "ingredients"]
    assert log == {"planner": 1, "menu": 2, "ingredients": 2, "synthesis": 1}
    # 2 manager calls + 2 per specialist, all counted in the team total.
    assert result.llm_calls == 6 and result.manager_llm_calls == 2
    assert result.prompt_tokens == 600 and result.completion_tokens == 60
    specialist_cost = sum(a.result.cost_usd for a in result.assignments)
    assert result.cost_usd >= specialist_cost
    assert {step.tool for step in result.steps if step.tool} == {"search_menus", "lookup_ingredients"}


def test_planner_can_skip_a_specialist() -> None:
    patchers, log = fake_team('{"menu": "Find the cheapest Masala Dosa.", "ingredients": null}',
                              "Cheapest Masala Dosa: Sri Kaveri Bhavan, Rs. 70.")
    with both_servers() as host, patchers[0], patchers[1]:
        result = run_squad("Where is Masala Dosa cheapest?", host)
    assert [a.specialist for a in result.assignments] == ["menu"]
    assert "ingredients" not in log


def test_specialists_only_get_their_own_tools() -> None:
    """The menu specialist cannot reach the ingredient server, even if the model tries."""
    patchers, _ = fake_team('{"menu": "Is Masala Dosa dairy-free?", "ingredients": null}',
                            "I could not check that.", menu_tool="lookup_ingredients")
    with both_servers() as host, patchers[0], patchers[1]:
        result = run_squad("q", host)
    [step] = [s for s in result.steps if s.tool]
    assert step.tool == "lookup_ingredients" and "not one of this agent's tools" in step.blocked


def test_manager_inventing_a_price_fails_closed() -> None:
    """Synthesis is checked against the specialists' real tool results, not their prose."""
    made_up = "Masala Dosa is cheapest at Sri Kaveri Bhavan, Rs. 15."
    patchers, log = fake_team(PLAN_BOTH, [made_up, made_up])
    with both_servers() as host, patchers[0], patchers[1]:
        result = run_squad(QUESTION, host)
    assert result.stopped_reason == "validation_failed" and result.answer == VALIDATION_FAILED_ANSWER
    assert log["synthesis"] == 2  # one draft + one retry
    assert result.manager_llm_calls == 3


def test_no_specialist_needed() -> None:
    patchers, _ = fake_team('{"menu": null, "ingredients": null}', "unused")
    with both_servers() as host, patchers[0], patchers[1]:
        result = run_squad("What's the weather?", host)
    assert result.stopped_reason == "no_specialist" and result.assignments == []
    assert result.llm_calls == 1


def test_specialist_quoting_the_tool_caveat_is_not_a_prompt_leak() -> None:
    """Race run 1: the ingredient prompt spelled the caveat out, the specialist echoed it,
    and Week 8's leak check blocked every allergen report. The caveat must come from the tool."""
    from agents.output_validation import leaks_system_prompt
    from agents.squad import SYNTHESIS_PROMPT
    report = "Ven Pongal contains cashews (tree nuts). " + ingredient_server.RECIPE_CAVEAT
    for prompt in [s.system_prompt for s in SPECIALISTS] + [SYNTHESIS_PROMPT]:
        assert not leaks_system_prompt(report, prompt)
