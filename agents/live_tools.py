"""Tools for the Streamlit "Agent" tab -- search whatever is actually
indexed in the app's knowledge base (uploads included), not the fixed
news_articles/ corpus that agents/tools.py and agents/race.py use for
Week 7's own testing (that corpus exists for grading, with known ground
truth; it is not what a real user's documents are). See
docs/week7-agents.md for why these are two separate tool sets sharing one
agent loop.
"""
from __future__ import annotations

from agents.tools import TOOL_SCHEMAS as _MENU_TOOL_SCHEMAS
from agents.tools import convert_currency
from rag_chat.retrieval import retrieve

LIVE_AGENT_SYSTEM_PROMPT = (
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


def search_documents(query: str) -> list[str]:
    """Search the live knowledge base -- whatever is indexed right now."""
    chunks, _sources = retrieve(query)
    return chunks  # already "[Source: filename]\n..."-labeled, see rag_chat/retrieval.py


LIVE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the currently indexed/uploaded documents for passages relevant to a "
                "query. Each returned passage is labeled with its source document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. an item name or topic."},
                },
                "required": ["query"],
            },
        },
    },
    _MENU_TOOL_SCHEMAS[1],  # convert_currency's schema -- generic, not menu-specific, reused as-is
]

LIVE_TOOL_FUNCTIONS = {
    "search_documents": lambda args: search_documents(args["query"]),
    "convert_currency": lambda args: convert_currency(args["amount"], args["from_currency"], args["to_currency"]),
}
