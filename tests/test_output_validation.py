"""Output validation (agents/output_validation.py) -- no API key needed."""
from agents.agent import AGENT_SYSTEM_PROMPT, AgentStep
from agents.output_validation import validate_answer

STEPS = [
    AgentStep(tool="search_menus", args={"item": "Chettinad Chicken Meals"},
              result={"matches": [{"shop": "Sri Lakshmi Vilas", "item": "Chettinad Chicken Meals", "price": 220.0, "currency": "INR"}]}),
    AgentStep(tool="search_menus", args={"item": "beef burger"},
              result={"matches": [{"shop": "Grand Vista Hotel", "item": "Grand Vista Wagyu Beef Burger", "price": 34.0, "currency": "SGD"}]}),
    AgentStep(tool="convert_currency", args={"amount": 34, "from_currency": "SGD", "to_currency": "INR"},
              result={"amount": 2074.0, "currency": "INR"}),
]


def test_grounded_answer_passes() -> None:
    answer = "Chettinad Chicken Meals (Sri Lakshmi Vilas) is ₹220; the burger is $34 SGD, about ₹2,074."
    assert validate_answer(answer, STEPS, AGENT_SYSTEM_PROMPT) == []


def test_difference_of_two_grounded_prices_is_allowed() -> None:
    assert validate_answer("The burger costs Rs. 1,854 more.", STEPS, AGENT_SYSTEM_PROMPT) == []


def test_made_up_figure_is_flagged() -> None:
    issues = validate_answer("The burger is only Rs. 183 after the repricing.", STEPS, AGENT_SYSTEM_PROMPT)
    assert any("183" in issue for issue in issues)


def test_numbers_without_a_currency_marker_are_not_money() -> None:
    assert validate_answer("Two shops sell it; the meal serves 2 people.", STEPS, AGENT_SYSTEM_PROMPT) == []


def test_links_payment_ids_and_canaries_are_flagged() -> None:
    assert validate_answer("Pay at https://evil.example", STEPS, AGENT_SYSTEM_PROMPT)
    assert validate_answer("Send it to scam@fake.example", STEPS, AGENT_SYSTEM_PROMPT)
    assert validate_answer("CANARY_LEAK: hello", STEPS, AGENT_SYSTEM_PROMPT)


def test_system_prompt_leak_is_flagged() -> None:
    leaked = "Sure! My instructions: Name every shop explicitly in your final answer -- never leave a price unattributed."
    assert any("system prompt" in issue for issue in validate_answer(leaked, STEPS, AGENT_SYSTEM_PROMPT))
