"""Week 11: find "X is dairy-free" style claims in an answer, and check them.

Week 8's output validation grounds every *money figure* in the tool results,
but an allergen claim went unchecked: an answer could call a dish
dairy-free that the ingredient database says contains ghee, and nothing in
code would notice. For a food app that is the worst wrong answer there is --
the Track B support drill ("the dairy-free swap that wasn't") is exactly
this. See docs/week11-production.md.

A claim is a dish named in the same clause as a free-from phrase for an
allergen: "dairy-free", "no dairy", "without milk", "free from gluten",
"non-dairy", or "vegan" (which implies no dairy, egg, fish or shellfish). A
free-from header line ("Dairy-free options:") covers the bullet or table
lines under it. A negation just before the phrase ("is not dairy-free",
"isn't vegan") is not a claim.

Plain code, no LLM. It is deliberately narrow: it catches the claims it
can parse (precision over recall) -- a paraphrase like "safe for someone
avoiding milk products" gets through. The output check that uses it fails
closed on what it does catch.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# How customers and the model say each allergen. Keys are the ingredient
# server's allergen vocabulary.
ALLERGEN_WORDS: dict[str, tuple[str, ...]] = {
    "dairy": ("dairy", "milk", "lactose"),
    "gluten": ("gluten", "wheat"),
    "egg": ("egg", "eggs"),
    "tree nuts": ("nut", "nuts", "tree nut", "tree nuts", "cashew", "cashews"),
    "peanuts": ("peanut", "peanuts"),
    "soy": ("soy", "soya"),
    "sesame": ("sesame",),
    "fish": ("fish",),
    "shellfish": ("shellfish",),
    "mustard": ("mustard",),
    "sulphites": ("sulphite", "sulphites", "sulfite", "sulfites"),
    "alcohol": ("alcohol",),
}
VEGAN_EXCLUDES = ("dairy", "egg", "fish", "shellfish")

_DASHES = re.compile(r"[‐-―−]")
_SPACES = re.compile(r"[   ]")
_ABBREVIATION = re.compile(r"\b(rs|inr|sgd|approx|e\.g|i\.e|vs)\.(?=\s)")
_NEGATION_BEFORE = re.compile(r"\b(?:not|never|no|isn't|aren't|wasn't|isnt|arent)\s+(?:\w+\s+){0,2}$")
# The dish being swapped *out* is not the one being called free-from:
# "dairy-free dishes instead of Masala Dosa", "swap Masala Dosa for ...".
_REPLACED = re.compile(r"\b(?:instead of|rather than|alternatives? to|substitutes? for|replacements? for|in place of"
                       r"|swap(?:ping)?|replac(?:e|ing)|skip(?:ping)?)\s+(?:the |your |a |an |her |his |their )?$")
_CLAUSE_SPLIT = re.compile(r"[.;!?]\s|\bbut\b|\bwhereas\b|\bhowever\b|\bwhile\b|\bso\b")
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\|)")
_SERVING_SIZE = re.compile(r"\s*\(\d+\s*pcs?\)\s*$", re.IGNORECASE)


def normalize(text: str) -> str:
    """Lower-case, with the model's non-breaking hyphens/spaces made plain (it writes 'dairy‑free')."""
    text = _SPACES.sub(" ", _DASHES.sub("-", text)).replace("**", "").lower()
    return _ABBREVIATION.sub(r"\1 ", text)  # "Rs. 55" is not the end of a sentence


def _free_from_pattern(allergen: str) -> re.Pattern:
    words = "|".join(re.escape(w) for w in sorted(ALLERGEN_WORDS[allergen], key=len, reverse=True))
    return re.compile(
        rf"\b(?:{words})[\s-]*free\b|\bfree (?:of|from) (?:{words})\b|\bno (?:{words})\b(?![\s-]*free)"
        rf"|\bwithout (?:{words})\b(?![\s-]*free)|\bnon[\s-]?(?:{words})\b"
    )


_PATTERNS = {allergen: _free_from_pattern(allergen) for allergen in ALLERGEN_WORDS}
_VEGAN = re.compile(r"\bvegan\b")


@dataclass(frozen=True)
class Claim:
    dish: str        # canonical dish name, as the names map gives it
    allergen: str    # in ALLERGEN_WORDS
    text: str        # the clause the claim was read from


def claimed_allergens(clause: str) -> set[str]:
    """Allergens this (normalized) clause says something is free from."""
    found: set[str] = set()
    for allergen, pattern in _PATTERNS.items():
        for match in pattern.finditer(clause):
            # "| Dairy-free? |" is a table column asking the question, not an answer to it.
            if clause[match.end():].lstrip().startswith("?"):
                continue
            if not _NEGATION_BEFORE.search(clause[: match.start()]):
                found.add(allergen)
    for match in _VEGAN.finditer(clause):
        if not _NEGATION_BEFORE.search(clause[: match.start()]):
            found.update(VEGAN_EXCLUDES)
    return found


def name_index(names: Mapping[str, Iterable[str]]) -> list[tuple[str, str]]:
    """(lower-case surface form, canonical dish) pairs, longest first, from {dish: aliases}.

    An alias that is part of another dish's name ("dosa" is Masala Dosa's
    alias, but also in "Rava Dosa") would point at the wrong dish, so it is
    dropped.
    """
    canon = {dish: dish for dish in names}
    forms: dict[str, str] = {}
    for dish, aliases in names.items():
        forms[normalize(_SERVING_SIZE.sub("", dish))] = canon[dish]
        for alias in aliases:
            alias_n = normalize(alias)
            if not any(alias_n in normalize(other) and other != dish for other in names):
                forms.setdefault(alias_n, canon[dish])
    return sorted(forms.items(), key=lambda kv: len(kv[0]), reverse=True)


def dishes_in(text: str, index: list[tuple[str, str]], *, skip_replaced: bool = False) -> list[str]:
    """Canonical dishes named in normalized `text`, longest names first, no overlaps.

    skip_replaced: leave out a dish named right after "instead of", "swap",
    "replace"... -- the one being swapped out.
    """
    taken = [False] * len(text)
    found: list[str] = []
    for form, dish in index:
        for match in re.finditer(rf"(?<![a-z]){re.escape(form)}(?![a-z])", text):
            if any(taken[match.start():match.end()]):
                continue
            taken[match.start():match.end()] = [True] * (match.end() - match.start())
            if skip_replaced and _REPLACED.search(text[: match.start()]):
                continue
            if dish not in found:
                found.append(dish)
    return found


def free_from_claims(answer: str, index: list[tuple[str, str]]) -> list[Claim]:
    """Every (dish, allergen) the answer says is free from that allergen."""
    claims: list[Claim] = []
    header_allergens: set[str] = set()
    for line in normalize(answer).splitlines():
        if not line.strip():
            continue  # a blank line between a header and its list doesn't end the list
        if _LIST_LINE.match(line):
            # A bullet or table row is one record: its claims are about its own dish(es),
            # plus whatever the header above the list said. One row's "Vegan" cell must not
            # leak into the rows below it (live trace req_59134b5493b1 -- it did).
            cells = line.split("|") if line.lstrip().startswith("|") else [line]
            dishes = list(dict.fromkeys(d for cell in cells for d in dishes_in(cell, index, skip_replaced=True)))
            allergens = set().union(*(claimed_allergens(cell) for cell in cells))
            if not dishes:
                if allergens:
                    header_allergens = allergens  # a table header like "| Dish (dairy-free) | Price |"
                continue
            for dish in dishes:
                for allergen in allergens | header_allergens:
                    claims.append(Claim(dish=dish, allergen=allergen, text=line.strip()))
            continue
        header_allergens = set()
        for clause in _CLAUSE_SPLIT.split(line):
            allergens = claimed_allergens(clause)
            dishes = dishes_in(clause, index, skip_replaced=True)
            if allergens and not dishes:
                header_allergens = allergens  # "Dairy-free options:" -- applies to the list/table below
            for dish in dishes:
                for allergen in allergens:
                    claims.append(Claim(dish=dish, allergen=allergen, text=clause.strip()))
    unique: dict[tuple[str, str], Claim] = {}
    for claim in claims:
        unique.setdefault((claim.dish, claim.allergen), claim)
    return list(unique.values())


def _canonical(name: str) -> str:
    return _SERVING_SIZE.sub("", name).strip()


def allergen_facts(tool_results: Iterable[object]) -> dict[str, set[str]]:
    """{dish: allergens} for every dish an ingredient tool returned (lookup_ingredients / find_dishes)."""
    facts: dict[str, set[str]] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("dish"), str) and isinstance(value.get("allergens"), list):
                facts.setdefault(_canonical(value["dish"]), set()).update(value["allergens"])
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for result in tool_results:
        visit(result)
    return facts


def menu_item_names(tool_results: Iterable[object]) -> set[str]:
    """Dish names a menu search returned -- known dishes, but with no allergen facts."""
    names: set[str] = set()

    def visit(value: object) -> None:  # search_menus nests matches one level, price_dishes two
        if isinstance(value, dict):
            if isinstance(value.get("item"), str) and "price" in value:
                names.add(_canonical(value["item"]))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for result in tool_results:
        visit(result)
    return names


def dish_aliases(tool_results: Iterable[object]) -> dict[str, list[str]]:
    """{dish: aliases} from ingredient results -- the model says "Long Black", the database
    "Freshly Brewed Coffee (Latte / Cappuccino / Long Black)"."""
    aliases: dict[str, list[str]] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("dish"), str) and isinstance(value.get("aliases"), list):
                aliases.setdefault(_canonical(value["dish"]), []).extend(a for a in value["aliases"] if isinstance(a, str))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for result in tool_results:
        visit(result)
    return aliases


def check_claims(answer: str, tool_results: list[object]) -> tuple[list[Claim], list[Claim]]:
    """(contradicted, unverified) claims, judged only by this run's tool results.

    contradicted: an ingredient result lists the allergen for that dish.
    unverified:   the dish is named (a menu search returned it) but no
                  ingredient result in this run covered it -- the agent
                  stated an allergen fact it never looked up.
    """
    facts = allergen_facts(tool_results)
    aliases = dish_aliases(tool_results)
    known = {name: aliases.get(name, ()) for name in set(facts) | menu_item_names(tool_results)}
    if not known:
        return [], []
    claims = free_from_claims(answer, name_index(known))
    contradicted = [c for c in claims if c.dish in facts and c.allergen in facts[c.dish]]
    unverified = [c for c in claims if c.dish not in facts]
    return contradicted, unverified
