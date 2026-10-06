"""Week 10 race harness and test set -- no API key needed.

The ground-truth tests call the real tools directly, so if a menu or the
ingredient database changes, the test set fails here instead of silently
grading agents against stale answers.
"""
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import RateLimitError

import agents.squad_race as race_module
import rag_chat.client as client_module
from agents.squad_race import CaseResult, Run, load, run_race, save, summarize
from agents.squad_testset import SQUAD_CASES
from agents.tools import TOOL_FUNCTIONS
from agents.trajectory_eval import check_outcome
from evals.judge import JudgeResult
from mcp_servers.ingredient_server import find_dishes, lookup_ingredients
from rag_chat.client import create_chat_completion, usage_scope

CASES = {case.id: case for case in SQUAD_CASES}


def _prices(item: str) -> dict[str, float]:
    return {m["shop"]: m["price"] for m in TOOL_FUNCTIONS["search_menus"]({"item": item})["matches"]}


# -- the test set itself ---------------------------------------------------

def test_test_set_shape() -> None:
    assert len(SQUAD_CASES) == 12 and len(CASES) == 12
    kinds = [case.kind for case in SQUAD_CASES]
    assert kinds.count("menu") == 3 and kinds.count("ingredient") == 3
    assert kinds.count("combined") == 5 and kinds.count("not_found") == 1


@pytest.mark.parametrize("case", SQUAD_CASES, ids=lambda c: f"case{c.id}")
def test_every_ground_truth_passes_its_own_checks(case) -> None:
    """A check the reference answer fails would grade every agent wrong."""
    passed, reasons = check_outcome(case.ground_truth, case)
    assert passed, reasons


def test_ground_truth_prices_match_the_menus() -> None:
    assert min(_prices("Filter Coffee").items(), key=lambda kv: kv[1]) == ("Thanjai Ruchi Mess", 20.0)
    assert _prices("Masala Dosa") == {"Sri Kaveri Bhavan": 70.0, "Sri Lakshmi Vilas": 75.0}
    assert min(_prices("Medu Vada").values()) == _prices("Medu Vada")["Vaigai Tiffin Center"] == 35.0
    assert _prices("Ven Pongal") == {"Sri Kaveri Bhavan": 55.0, "Sri Lakshmi Vilas": 58.0}
    assert _prices("Plain Idli")["Vaigai Tiffin Center"] == 30.0 and _prices("Rava Dosa") == {"Vaigai Tiffin Center": 60.0}
    assert _prices("Paal Payasam") == {"Thanjai Ruchi Mess": 45.0}
    convert = TOOL_FUNCTIONS["convert_currency"]
    assert convert({"amount": 16, "from_currency": "SGD", "to_currency": "INR"})["amount"] == 976.0
    assert convert({"amount": 34, "from_currency": "SGD", "to_currency": "INR"})["amount"] == 2074.0
    assert _prices("Uttapam") == {} and _prices("Margherita Pizza") == {}


def test_ground_truth_ingredients_match_the_database() -> None:
    vegan_gluten_free = {d.dish for d in find_dishes(diet="vegan", free_from=["gluten"]).dishes}
    assert len(vegan_gluten_free) == 7
    for dish in vegan_gluten_free:  # case 5 must name every one of them
        word = dish.split(" (")[0].split()[-1].lower() if "TWG" not in dish else "tea"
        assert word in CASES[5].ground_truth.lower(), dish
    assert "dairy" in lookup_ingredients("Masala Dosa").matches[0].allergens
    assert "gluten" in lookup_ingredients("Rava Dosa").matches[0].allergens
    assert "sesame" in lookup_ingredients("Hainanese Chicken Rice").matches[0].allergens
    assert lookup_ingredients("Medu Vada").matches[0].diet == "vegan"
    assert "tree nuts" in lookup_ingredients("Ven Pongal").matches[0].allergens
    assert "alcohol" in lookup_ingredients("Classic Tiramisu").matches[0].allergens
    assert "alcohol" not in lookup_ingredients("Paal Payasam").matches[0].allergens
    assert lookup_ingredients("Plain Idli").matches[0].allergens == []


def test_checks_reject_a_wrong_answer() -> None:
    passed, _ = check_outcome("Masala Dosa is safe; cheapest at Sri Lakshmi Vilas, Rs. 75.", CASES[7])
    assert not passed
    passed, _ = check_outcome("Vegan gluten-free: Plain Idli, Medu Vada, Idiyappam.", CASES[5])
    assert not passed  # dropped dishes is a fail -- Week 9's failure mode


# -- the harness -------------------------------------------------------------

def _fake_variant(answer: str, tokens: int):
    def variant(question, host):
        client_module.record_usage(SimpleNamespace(prompt_tokens=tokens, completion_tokens=0, total_tokens=tokens),
                                   "openai/gpt-oss-120b")
        return answer, ["search_menus"], "done", {}
    return variant


def test_same_meter_for_every_variant_and_judge_is_not_billed() -> None:
    variants = {"single": _fake_variant(CASES[1].ground_truth, 100),
                "squad": _fake_variant("I don't know.", 300)}
    order: list[str] = []

    def judge(question, ground_truth, answer):
        client_module.record_usage(SimpleNamespace(prompt_tokens=9999, completion_tokens=0, total_tokens=9999),
                                   "openai/gpt-oss-120b")
        return JudgeResult("CORRECT" if "Thanjai" in answer else "INCORRECT", "r")

    with patch.dict(race_module.VARIANTS, variants), patch.object(race_module, "judge_answer", side_effect=judge):
        results = run_race([CASES[1]], ["single", "squad"], host=None, repeat=2,
                           log=lambda line: order.append(line.split()[1]))
    single, squad = results[0].runs["single"], results[0].runs["squad"]
    assert [r.tokens for r in single] == [100, 100] and [r.tokens for r in squad] == [300, 300]
    assert all(r.correct for r in single) and not any(r.correct for r in squad)
    assert order == ["single", "squad", "squad", "single"]  # order alternates per trial
    stats = summarize(results, "squad")
    assert stats["quality_correct"] == 0 and stats["mean_tokens"] == 300


def test_rate_limit_sleeps_are_recorded_not_hidden() -> None:
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid"))
    error = RateLimitError("slow down", response=response, body=None)
    ok = SimpleNamespace(choices=[], usage=None)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: ok)))
    calls = iter([error, ok])

    def create(**_):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    fake.chat.completions.create = create
    with patch.object(client_module, "get_chat_client", return_value=fake), \
         patch.object(client_module.time, "sleep"), usage_scope() as usage:
        create_chat_completion(model="m", messages=[])
    assert usage.rate_limit_waits == 1 and usage.rate_limit_wait_seconds == 5.0


def test_majority_vote_and_save_load_round_trip(tmp_path) -> None:
    def run(ok: bool) -> Run:
        return Run(answer="a", checks_pass=ok, check_reasons=[], judge="CORRECT", judge_reason="",
                   seconds=1.0, prompt_tokens=10, completion_tokens=1, cost_usd=0.001, llm_calls=2,
                   tool_calls=[], stopped_reason="done")
    row = CaseResult(id=1, question="q", kind="menu", ground_truth="g",
                     runs={"single": [run(True), run(True), run(False)], "squad": [run(True), run(False), run(False)]})
    assert summarize([row], "single")["quality_correct"] == 1
    assert summarize([row], "squad")["quality_correct"] == 0
    path = tmp_path / "race.json"
    save([row], path)
    assert load(path)[0].runs["squad"][1].checks_pass is False


def test_resume_only_runs_what_is_missing() -> None:
    """After a daily token limit, finished runs are kept and only the gap is run."""
    variants = {"single": _fake_variant("a", 1), "squad": _fake_variant("b", 1)}
    ran: list[str] = []
    with patch.dict(race_module.VARIANTS, variants), \
         patch.object(race_module, "judge_answer", return_value=JudgeResult("CORRECT", "")):
        first = run_race([CASES[1]], ["single"], host=None, repeat=3, log=lambda _: None)
        resumed = run_race([CASES[1], CASES[2]], ["single", "squad"], host=None, repeat=3,
                           log=lambda line: ran.append(line.split()[0] + line.split()[1]), existing=first)
    assert ran.count("#1single") == 0 and ran.count("#1squad") == 3
    assert ran.count("#2single") == 3 and ran.count("#2squad") == 3
    assert all(len(runs) == 3 for row in resumed for runs in row.runs.values())
