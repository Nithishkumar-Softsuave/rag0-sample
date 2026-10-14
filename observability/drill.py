"""Week 11 support drill: a two-week production log with one bad answer planted in it.

Track B's drill is "find the dairy-free swap that wasn't". The course's
track pack (the planted log) was not in docs/, so this builds the equivalent
from our own data: ~2,400 requests over the last 14 days, in exactly the
format agents/kitchen.py logs (observability/tracing.py), every answer
built from the real menu and ingredient data so it is *correct* -- except
one.

    python -m observability.drill                    # writes data/drill_traces.jsonl
    python -m observability.logs --log data/drill_traces.jsonl find dairy swap

The log is seeded, so it is the same every time. It is full of near-misses
on purpose -- correct dairy-free swaps, dosa questions, "not dairy-free"
answers, the same user asking other things -- so a plain text search for
"dairy" or "dosa" finds hundreds of requests, not one. The support ticket
that starts the drill is in docs/week11-production.md.

Every generated trace carries attrs.synthetic = true, so it can never be
mistaken for (or mixed into) the real log in data/traces.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agents.tools import search_menus_for_agent
from mcp_servers.ingredient_server import RECIPE_CAVEAT, find_dishes, load_dishes, lookup_ingredients
from rag_chat.client import PRICING_PER_MILLION_TOKENS
from rag_chat.config import PROJECT_ROOT

DRILL_PATH = PROJECT_ROOT / "data" / "drill_traces.jsonl"
MODEL = "openai/gpt-oss-120b"
APP_VERSION = "6241f18"
HOST_PROMPT_VERSION = "fe219c3738"
REQUESTS = 2400
DAYS = 14

# The planted bad answer. The agent looked up Masala Dosa (correctly: ghee,
# so not dairy-free), then priced Ven Pongal as the swap and called it
# dairy-free -- without ever looking it up. Ven Pongal is made with ghee.
PLANTED_USER = "u_4417"
PLANTED_QUESTION = "my wife cant have milk. what can she get instead of the masala dosa at kaveri?"
PLANTED_ANSWER = (
    "Masala Dosa is made with ghee, so it isn't dairy-free. A good swap at Sri Kaveri Bhavan is the "
    "Ven Pongal (Rs. 55) - a dairy-free rice and moong dal dish, mildly peppery and filling. "
    "Recipes can vary, so do mention the allergy when you order."
)


def _price(dish: str) -> list[dict]:
    return search_menus_for_agent(dish)["matches"]


def _llm_cost(prompt: int, completion: int, cached: int) -> float:
    rates = PRICING_PER_MILLION_TOKENS[MODEL]
    return ((prompt - cached) * rates["input"] + cached * rates["cached_input"] + completion * rates["output"]) / 1e6


class _Builder:
    """Builds one trace dict, span by span, in the tracer's JSON shape."""

    def __init__(self, rng: random.Random, when: datetime, user: str, question: str) -> None:
        self.rng, self.when, self.user, self.question = rng, when, user, question
        self.spans: list[dict] = []
        self.clock = rng.uniform(1, 4)  # ms before the first step
        self.context = 830
        self.root = self._span("request", "request", None, {"route": "single"})
        self.agent = self._span("agent", "agent", self.root["id"],
                                {"task": question, "prompt_version": HOST_PROMPT_VERSION})

    def _span(self, name: str, kind: str, parent: str | None, attrs: dict) -> dict:
        item = {"id": uuid.UUID(int=self.rng.getrandbits(128)).hex[:8], "parent_id": parent, "name": name,
                "kind": kind, "start_ms": round(self.clock, 1), "duration_ms": 0.0, "llm_calls": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "cached_prompt_tokens": 0, "cost_usd": 0.0,
                "rate_limit_wait_seconds": 0, "attrs": attrs, "error": None}
        self.spans.append(item)
        return item

    def llm(self, calls: list[str], completion: int) -> None:
        cached = 768 if self.rng.random() < 0.7 else 0
        span = self._span("llm", "llm", self.agent["id"],
                          {"model": MODEL, "finish_reason": "tool_calls" if calls else "stop",
                           "tool_calls": calls, "cached_tokens": cached})
        span.update(llm_calls=1, prompt_tokens=self.context, completion_tokens=completion, cached_prompt_tokens=cached,
                    cost_usd=round(_llm_cost(self.context, completion, cached), 8),
                    duration_ms=round(self.rng.uniform(350, 1800) + completion * 2.5, 1))
        self.clock += span["duration_ms"]
        self.context += completion + 40

    def tool(self, name: str, args: dict, result: dict) -> None:
        span = self._span(f"tool:{name}", "tool", self.agent["id"], {"args": args, "result": result, "blocked": None})
        span["duration_ms"] = round(self.rng.uniform(3, 60), 1)
        self.clock += span["duration_ms"]
        self.context += min(len(json.dumps(result)) // 4, 900) + 60

    def finish(self, answer: str, outcome: str = "done") -> dict:
        check = self._span("validate", "check", self.agent["id"], {"answer": answer, "issues": [], "enabled": True})
        check["duration_ms"] = round(self.rng.uniform(0.1, 0.6), 1)
        self.clock += check["duration_ms"]
        llm = [s for s in self.spans if s["kind"] == "llm"]
        totals = {key: sum(s[key] for s in llm) for key in ("llm_calls", "prompt_tokens", "completion_tokens",
                                                            "cached_prompt_tokens", "cost_usd")}
        totals["cost_usd"] = round(totals["cost_usd"], 8)
        for span in (self.root, self.agent):
            span.update(totals, duration_ms=round(self.clock - span["start_ms"], 1))
        self.agent["attrs"].update(task=self.question, stopped_reason=outcome, answer=answer, validation_issues=[])
        self.root["attrs"]["outcome"] = outcome
        tools = [s["name"].split(":", 1)[1] for s in self.spans if s["kind"] == "tool"]
        return {
            "request_id": "req_" + uuid.UUID(int=self.rng.getrandbits(128)).hex[:12],
            "timestamp": self.when.isoformat(timespec="milliseconds"), "route": "single",
            "question": self.question, "user_id": self.user, "session_id": None, "answer": answer, "outcome": outcome,
            "duration_ms": round(self.clock + self.rng.uniform(1, 5), 1), **totals, "rate_limit_wait_seconds": 0,
            "attrs": {"model": MODEL, "provider": "groq", "app_version": APP_VERSION,
                      "host_prompt_version": HOST_PROMPT_VERSION, "synthetic": True, "tool_calls": tools,
                      "blocked_tool_calls": 0, "validation_issues": []},
            "spans": self.spans,
        }


LABEL = {"tree nuts": "nut"}  # "nut-free", not "tree nuts-free"

ALLERGEN_SAYINGS = {
    "dairy": ["dairy-free", "allergic to milk", "lactose intolerant", "can't have dairy", "vegan"],
    "gluten": ["gluten-free", "coeliac", "can't have wheat"],
    "tree nuts": ["allergic to nuts", "nut allergy"],
    "egg": ["egg-free", "allergic to eggs"],
    "sesame": ["sesame allergy"],
}


def _dish_lookup(b: _Builder, dish: str) -> dict:
    result = lookup_ingredients(dish).model_dump()
    b.llm(["lookup_ingredients"], b.rng.randint(25, 90))
    b.tool("lookup_ingredients", {"dish": dish}, result)
    return result["matches"][0]


def _price_lookup(b: _Builder, dish: str) -> list[dict]:
    result = search_menus_for_agent(dish)
    b.llm(["search_menus"], b.rng.randint(25, 80))
    b.tool("search_menus", {"item": dish}, result)
    return result["matches"]


def _cheapest(matches: list[dict]) -> dict:
    return min(matches, key=lambda m: (m["currency"] != "INR", m["price"]))


def _money(match: dict) -> str:
    return f"Rs. {match['price']:g}" if match["currency"] == "INR" else f"SGD {match['price']:g}"


def price_question(b: _Builder, dishes: list[str]) -> str:
    dish = b.rng.choice(dishes)
    b.question = b.rng.choice([f"how much is {dish}?", f"cheapest {dish.lower()}?", f"price of {dish} pls",
                               f"Where can I get {dish} for the least money?", f"{dish} price"])
    matches = _price_lookup(b, dish)
    best = _cheapest(matches)
    b.llm([], b.rng.randint(60, 220))
    others = "; ".join(f"{m['shop']} {_money(m)}" for m in matches if m is not best)
    return (f"{dish} is cheapest at {best['shop']} ({_money(best)})." + (f" Elsewhere: {others}." if others else ""))


def allergen_question(b: _Builder, dishes: list[str]) -> str:
    dish = b.rng.choice(dishes)
    allergen = b.rng.choice(list(ALLERGEN_SAYINGS))
    label = LABEL.get(allergen, allergen)
    b.question = b.rng.choice([f"is {dish} {label}-free?", f"Does {dish} have {allergen}?",
                               f"{ALLERGEN_SAYINGS[allergen][0]} - can I eat {dish}?"])
    facts = _dish_lookup(b, dish)
    b.llm([], b.rng.randint(80, 240))
    if allergen in facts["allergens"]:
        hit = next((i for i in facts["ingredients"] if allergen.split()[0] in i or i in {"ghee", "milk", "curd", "butter", "cream", "cashews"}),
                   facts["ingredients"][0])
        return (f"No - {dish} contains {allergen} (it is made with {hit}), so it is not {label}-free. "
                f"{RECIPE_CAVEAT}")
    return f"Yes, {dish} is {label}-free in the standard recipe. {RECIPE_CAVEAT}"


def swap_question(b: _Builder, dishes: list[str]) -> str:
    allergen = b.rng.choice(["dairy", "dairy", "dairy", "gluten", "tree nuts"])
    containing = [d for d in dishes if allergen in lookup_ingredients(d).matches[0].allergens]
    dish = b.rng.choice(containing)
    saying = b.rng.choice(ALLERGEN_SAYINGS[allergen])
    label = LABEL.get(allergen, allergen)
    b.question = b.rng.choice([f"I'm {saying}. what can I have instead of {dish}?",
                               f"{saying} - suggest a swap for {dish}", f"swap for {dish.lower()}? ({saying})",
                               f"What's a {label}-free alternative to {dish}?"])
    _dish_lookup(b, dish)
    free = find_dishes(free_from=[allergen]).model_dump()
    b.llm(["find_dishes"], b.rng.randint(40, 120))
    b.tool("find_dishes", {"free_from": [allergen]}, free)
    options = [d["dish"] for d in free["dishes"] if d["dish"] in dishes]
    alt = b.rng.choice(options)
    best = _cheapest(_price_lookup(b, alt))
    b.llm([], b.rng.randint(120, 300))
    return (f"{dish} isn't {label}-free. Swap it for {alt} ({best['shop']}, {_money(best)}), which is "
            f"{label}-free. {RECIPE_CAVEAT}")


def diet_question(b: _Builder, dishes: list[str]) -> str:
    diet = b.rng.choice(["vegan", "vegetarian"])
    b.question = b.rng.choice([f"what's {diet} on the menu?", f"{diet} options?", f"list {diet} dishes"])
    result = find_dishes(diet=diet).model_dump()
    b.llm(["find_dishes"], b.rng.randint(40, 100))
    b.tool("find_dishes", {"diet": diet}, result)
    b.llm([], b.rng.randint(150, 350))
    names = ", ".join(d["dish"] for d in result["dishes"])
    return f"{diet.capitalize()} dishes: {names}. {RECIPE_CAVEAT}"


KINDS = [(price_question, 0.40), (allergen_question, 0.27), (swap_question, 0.23), (diet_question, 0.10)]


def _timestamps(rng: random.Random, now: datetime, count: int) -> list[datetime]:
    """Busier at breakfast, lunch and dinner (IST), like a food app."""
    peaks = [(8.5, 1.2, 3), (13.0, 1.0, 4), (20.0, 1.3, 4)]
    times = []
    while len(times) < count:
        day = now.date() - timedelta(days=rng.randrange(DAYS))
        mu, sigma, _ = rng.choices(peaks, weights=[p[2] for p in peaks])[0]
        hour_ist = min(23.99, max(6.0, rng.gauss(mu, sigma)))
        when = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(hours=hour_ist - 5.5)
        if when < now:
            times.append(when)
    return sorted(times)


def build(now: datetime, seed: int = 11, count: int = REQUESTS) -> list[dict]:
    rng = random.Random(seed)
    dishes = [d.dish for d in load_dishes() if _price(d.dish)]
    users = [f"u_{n}" for n in rng.sample(range(1000, 9999), 700)]
    times = _timestamps(rng, now, count - 1)
    traces = []
    for when in times:
        builder = _Builder(rng, when, rng.choice(users), "")
        kind = rng.choices([k for k, _ in KINDS], weights=[w for _, w in KINDS])[0]
        answer = kind(builder, dishes)
        traces.append(builder.finish(answer))

    # The planted one: four days ago, 7:42 pm IST.
    day = (now - timedelta(days=4)).date()
    when = datetime(day.year, day.month, day.day, 14, 12, 37, tzinfo=timezone.utc)
    planted = _Builder(rng, when, PLANTED_USER, PLANTED_QUESTION)
    _dish_lookup(planted, "Masala Dosa")
    _price_lookup(planted, "Ven Pongal")  # priced, never looked up -- the missing step
    planted.llm([], 214)
    traces.append(planted.finish(PLANTED_ANSWER))
    return sorted(traces, key=lambda t: t["timestamp"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Week 11 support-drill log.")
    parser.add_argument("--out", type=Path, default=DRILL_PATH)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    traces = build(datetime.now(timezone.utc), seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace, ensure_ascii=False) + "\n")
    print(f"wrote {len(traces)} requests over {DAYS} days to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
