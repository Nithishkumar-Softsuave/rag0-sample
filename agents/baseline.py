"""Frozen copy of the Week 7 agent configuration, kept only so Week 8's
before/after numbers are reproducible with a flag instead of `git stash`.

Nothing in the app uses these -- agents/trajectory_eval.py and
agents/injection_probe.py pass them when run with `--policy baseline`.
Do not "improve" these strings: their whole value is that they are exactly
what shipped at the end of Week 7.
"""
from __future__ import annotations

from agents.tools import convert_currency, search_menus

WEEK7_AGENT_SYSTEM_PROMPT = (
    "You answer questions about restaurant menu prices using two tools: "
    "search_menus (find an item's price at every shop that sells it) and "
    "convert_currency (convert an amount between currencies). Call "
    "search_menus for every item the question mentions before answering. "
    "If prices are in different currencies, convert to a common one before "
    "comparing. Name every shop explicitly in your final answer -- never "
    "leave a price unattributed. If a searched item has no matches, say so "
    "instead of guessing. Once you have enough information, answer in plain "
    "text with no further tool calls."
)

WEEK7_LIVE_AGENT_SYSTEM_PROMPT = (
    "You answer questions using two tools: search_documents (search the user's "
    "own uploaded/indexed documents for passages relevant to a query) and "
    "convert_currency (convert an amount between currencies). Call "
    "search_documents for every distinct item or entity the question asks "
    "about before answering -- each returned passage is labeled with its "
    "source document as [Source: filename]. Only use facts from the passage "
    "whose label matches the entity you are asking about; never blend facts "
    "from different sources. If the question needs comparing amounts in "
    "different currencies, convert them to a common currency before "
    "comparing. Always say which source document each fact came from in your "
    "final answer. If a search returns nothing relevant, say so instead of "
    "guessing. Once you have enough information, answer in plain text with "
    "no further tool calls."
)

# Week 7's search_menus tool returned a bare list -- no ambiguity note.
WEEK7_TOOL_FUNCTIONS = {
    "search_menus": lambda args: search_menus(args["item"]),
    "convert_currency": lambda args: convert_currency(args["amount"], args["from_currency"], args["to_currency"]),
}
