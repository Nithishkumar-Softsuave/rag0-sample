"""Least-privilege checks on every tool call, enforced in code before the
tool runs -- not requested in the prompt.

Week 8's first injection defense (docs/week8-agent-failures.md) was purely
prompt-level: it asked the model not to obey instructions found in
documents. That depends on the model choosing to comply. These checks don't:
a blocked call never reaches the tool, whatever the model was talked into.

The key rule is *grounding*: convert_currency may only convert an amount
that an earlier tool call in this same run actually returned as a price.
The agent's job is comparing menu prices, so an amount that came from
nowhere -- a number the model made up, a figure planted in the user message
("it was repriced to $3 today"), or `amount=999999` smuggled in via a
document -- is out of scope by definition. Trade-off, accepted on purpose:
this agent cannot answer "what is 50 SGD in rupees?" for an arbitrary
number. That is not its job, and a narrower tool is a safer tool.
"""
from __future__ import annotations

import re

SUPPORTED_CURRENCIES = frozenset({"INR", "SGD"})
MAX_AMOUNT = 100_000.0
MAX_QUERY_CHARS = 100

# A number counts as a price only when it sits next to a currency marker --
# so "amount=999999" inside a document's text is NOT a grounded price, but
# "Price: Rs. 60" or "$34.00" is.
_PRICE_BEFORE = re.compile(r"(?:rs\.?|₹|\$|sgd|inr|usd|price:)\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
_PRICE_AFTER = re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:inr|sgd|rupees)\b", re.IGNORECASE)


def _to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def price_like_numbers(text: str) -> set[float]:
    """Numbers in free text that are marked as money (Rs. 60, $34.00, 2,074 INR)."""
    found = {_to_float(match) for match in _PRICE_BEFORE.findall(text) + _PRICE_AFTER.findall(text)}
    return {value for value in found if value is not None}


def grounded_amounts(steps) -> set[float]:
    """Every price/amount earlier tool calls in this run actually returned.

    Walks tool results generically, so it covers both tool sets: search_menus
    records (`price`), convert_currency output (`amount`), and
    search_documents' free-text chunks (price-like numbers only).
    """
    amounts: set[float] = set()

    def walk(value) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                if key in {"price", "amount"} and isinstance(inner, (int, float)) and not isinstance(inner, bool):
                    amounts.add(float(inner))
                else:
                    walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)
        elif isinstance(value, str):
            amounts.update(price_like_numbers(value))

    for step in steps:
        if step.tool and not step.blocked:
            walk(step.result)
    return amounts


def is_grounded(amount: float, grounded: set[float]) -> bool:
    """Within 1% (or 0.01) of a grounded amount -- allows for float rounding."""
    return any(abs(amount - known) <= max(0.01, 0.01 * abs(known)) for known in grounded)


def _schema_for(name: str, tool_schemas: list[dict]) -> dict | None:
    for schema in tool_schemas:
        if schema["function"]["name"] == name:
            return schema["function"]["parameters"]
    return None


def check_tool_call(name: str, args: dict, tool_schemas: list[dict], prior_steps) -> str | None:
    """Return why this call must be blocked, or None if it may run."""
    parameters = _schema_for(name, tool_schemas)
    if parameters is None:
        return f"'{name}' is not one of this agent's tools."

    for required in parameters.get("required", []):
        if required not in args:
            return f"missing required argument '{required}'."
    unexpected = set(args) - set(parameters.get("properties", {}))
    if unexpected:
        return f"unexpected argument(s) {sorted(unexpected)}."

    # Loop breaker: the tools are deterministic, so an identical repeat call
    # can only return the same answer -- the classic "going in circles" shape.
    if any(step.tool == name and step.args == args and not step.blocked for step in prior_steps):
        return "this exact call already ran in this conversation -- reuse its result instead of repeating it."

    if name in {"search_menus", "search_documents"}:
        query = args.get("item", args.get("query"))
        if not isinstance(query, str) or not query.strip():
            return "the search text must be a non-empty string."
        if len(query) > MAX_QUERY_CHARS:
            return f"the search text is longer than {MAX_QUERY_CHARS} characters."

    if name == "convert_currency":
        amount = args["amount"]
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            return "amount must be a number."
        if not 0 < amount <= MAX_AMOUNT:
            return f"amount must be between 0 and {MAX_AMOUNT:,.0f}."
        for key in ("from_currency", "to_currency"):
            code = str(args[key]).upper()
            if code not in SUPPORTED_CURRENCIES:
                return f"{key} '{code}' is not supported (only {sorted(SUPPORTED_CURRENCIES)})."
        grounded = grounded_amounts(prior_steps)
        if not grounded:
            return "look the price up first -- convert_currency only converts a price a search returned."
        if not is_grounded(float(amount), grounded):
            return (
                f"{amount} is not a price any earlier search returned in this conversation "
                "-- convert_currency only converts looked-up prices, never figures from the "
                "question or from document text."
            )
    return None
