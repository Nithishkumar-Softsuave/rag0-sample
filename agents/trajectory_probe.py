"""Adversarial probe for an outcome-vs-trajectory gap.

agents/testset.py's RACE_CASES were chosen (Week 7) to be cleanly answerable
by both paths -- every item name in those questions really is the same dish
at every shop that sells it. This probe asks a question built to break that
assumption instead.

The bug: agents/tools.py's search_menus() matches any item whose name
contains the query as a substring (or vice versa). That is correct when
every match is genuinely the same dish ("Filter Coffee"), but two different
dishes can share a substring by accident -- "Kaveri Special Thali" (a
non-vegetarian banana-leaf meal, Sri Kaveri Bhavan) and "Meals (Veg Thali)"
(a separate, plain vegetarian dish, sold under that same name at both Sri
Lakshmi Vilas and Vaigai Tiffin Center) all match the query "Thali". The
agent's tool-call trajectory for this question -- one search_menus call,
correct tool, correct count -- looks identical to a clean case (agents/
trajectory.py's check_trajectory would pass it), and the final answer reads
confidently. But it silently treats three non-identical dishes as one
comparable item and names a "cheapest" -- not a well-formed comparison, and
not something the tool-choice check can catch, because the tools called
were the right ones; what's wrong is what got compared once the results
came back.

Week 8 fix: agents/tools.py's search_menus_for_agent() now attaches a
`warning` when the matches are different dishes, and the hardened system
prompt says what to do with it. `--policy baseline` reproduces the bug;
`--policy hardened` (default) shows the fix. The same question is case 101
in agents/adversarial_testset.py, measured at scale by
agents/trajectory_eval.py -- this script is the quick single-question view.

Usage:
    python -m agents.trajectory_probe --policy baseline
    python -m agents.trajectory_probe --policy hardened
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agents.agent import run_agent
from agents.failure_modes import AMBIGUOUS_COMPARISON, classify_run
from agents.trajectory_eval import POLICIES

RESULTS_DIR = Path(__file__).resolve().parent / "results"

PROBE_QUESTION = "Which shop has the cheapest Thali?"


def _matches(result) -> list[dict]:
    return result.get("matches", []) if isinstance(result, dict) else result or []


def run_probe(policy_name: str, repeat: int = 3) -> list[dict]:
    """Run the probe question `repeat` times; flag each trial that compared
    different dishes as one item (agents/failure_modes.py's rule)."""
    trials = []
    config = POLICIES[policy_name]
    for _ in range(repeat):
        result = run_agent(PROBE_QUESTION, **config)
        distinct_items = sorted({
            record["item"]
            for step in result.steps if step.tool == "search_menus" and not step.blocked
            for record in _matches(step.result)
        })
        conflated = AMBIGUOUS_COMPARISON in classify_run(result, ("search_menus",), config["system_prompt"])
        trials.append({
            "question": PROBE_QUESTION, "policy": policy_name,
            "steps": [asdict(step) for step in result.steps],
            "answer": result.answer,
            "distinct_items_compared": distinct_items,
            "conflated_different_dishes": conflated,
        })
    return trials


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Probe the 'cheapest Thali' outcome-vs-trajectory gap.")
    parser.add_argument("--policy", choices=list(POLICIES), default="hardened")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    trials = run_probe(args.policy, args.repeat)
    for i, trial in enumerate(trials):
        print(f"--- trial {i} --- search matched {trial['distinct_items_compared']}")
        print(f"conflated different dishes: {trial['conflated_different_dishes']}")
        print("answer:", trial["answer"])
        print()
    print(f"Conflated: {sum(t['conflated_different_dishes'] for t in trials)}/{len(trials)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"trajectory_probe_{args.policy}.json"
    path.write_text(json.dumps(trials, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
