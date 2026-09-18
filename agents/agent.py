"""The hand-built agent: think, call a tool, look at the result, repeat.

No framework -- this is ~50 lines against the raw chat.completions API's
native tool-calling, so every step is something you can read and log
yourself instead of a framework deciding it for you. Three independent stop
conditions (steps, cost, wall-clock time) so a stuck loop always ends.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from agents.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
from rag_chat.client import create_chat_completion, get_usage, record_usage, reset_usage, resolve_max_tokens
from rag_chat.config import get_settings

AGENT_SYSTEM_PROMPT = (
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


@dataclass
class AgentStep:
    tool: str | None
    args: dict | None
    result: object


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    stopped_reason: str = "done"  # "done", "max_steps", "max_cost", "max_seconds"
    elapsed_seconds: float = 0.0
    cost_usd: float = 0.0
    tool_calls: int = 0


def run_agent(
    question: str,
    *,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
    tool_schemas: list[dict] = TOOL_SCHEMAS,
    tool_functions: dict = TOOL_FUNCTIONS,
    max_steps: int = 6,
    max_cost_usd: float = 0.02,
    max_seconds: float = 30.0,
) -> AgentResult:
    """Run the think -> act -> observe loop until an answer or a limit hits.

    The loop itself never changes; only which tools it's given does. Week
    7's own test harness (agents/race.py) uses the default menu tools
    (agents/tools.py, backed by the fixed news_articles/ corpus with known
    ground truth); the live Streamlit "Agent" tab passes
    agents/live_tools.py instead, which searches whatever is actually
    indexed -- uploads included -- via the same retrieve() the Chat tab
    uses. See docs/week7-agents.md.
    """
    reset_usage()
    start = time.perf_counter()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    steps: list[AgentStep] = []
    model = get_settings().chat_model

    for _ in range(max_steps):
        if time.perf_counter() - start > max_seconds:
            return _stopped("Stopped: exceeded time limit.", "max_seconds", steps, start)
        if get_usage().cost_usd > max_cost_usd:
            return _stopped("Stopped: exceeded cost budget.", "max_cost", steps, start)

        response = create_chat_completion(
            model=model, messages=messages, tools=tool_schemas, max_tokens=resolve_max_tokens(400),
        )
        record_usage(response.usage, model)
        message = response.choices[0].message

        if not message.tool_calls:
            steps.append(AgentStep(tool=None, args=None, result=message.content))
            return AgentResult(
                answer=message.content or "I could not generate an answer.",
                steps=steps, stopped_reason="done",
                elapsed_seconds=time.perf_counter() - start, cost_usd=get_usage().cost_usd,
                tool_calls=sum(1 for step in steps if step.tool),
            )

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            result = tool_functions[call.function.name](args)
            steps.append(AgentStep(tool=call.function.name, args=args, result=result))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

    return _stopped("Stopped: exceeded step limit before reaching an answer.", "max_steps", steps, start)


def _stopped(message: str, reason: str, steps: list[AgentStep], start: float) -> AgentResult:
    return AgentResult(
        answer=message, steps=steps, stopped_reason=reason,
        elapsed_seconds=time.perf_counter() - start, cost_usd=get_usage().cost_usd,
        tool_calls=sum(1 for step in steps if step.tool),
    )
