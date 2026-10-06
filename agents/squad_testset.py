"""Week 10 test set: the same 12 questions for the single agent and the squad.

Track B's single agent now has two kinds of facts (Week 9): menu prices and
ingredients/allergens. The mix is chosen so the race can show *where* a
team helps and where it is pure overhead, not just an overall score:

- menu       -- prices only. One specialist can do it; the manager is overhead.
- ingredient -- allergens/diet only. Same: overhead expected.
- combined   -- needs both servers. The only kind where the two specialists
                have independent work to do in parallel -- the team's best case.
- not_found  -- a dish nobody has. Reliability: say so, don't guess.

Ground truth is from mcp_servers/data/ingredients.json and the menus in
news_articles/ (checked by tests/test_squad_race.py). `must_include` groups
reuse Week 8's deterministic outcome check (agents/trajectory_eval.py
check_outcome): every group must match, any phrase in a group counts.
`expected_tools` is informational here -- an ingredient question can be
answered correctly by either lookup_ingredients or find_dishes.
"""
from __future__ import annotations

from agents.testset import NOT_FOUND, RaceCase

NOT_SAFE = ("not safe", "isn't safe", "is not safe", "not dairy-free", "not dairy free", "contains dairy",
            "contains ghee", "includes ghee", "has ghee", "not suitable", "unsafe", "no,", "no -", "no —", "no –",
            "no.")
CONTAINS = ("contains", "includes", "has ", "made with", "uses", "yes")

SQUAD_CASES: list[RaceCase] = [
    # -- menu only ----------------------------------------------------------
    RaceCase(
        1, "Which shop has the cheapest Filter Coffee?",
        "Thanjai Ruchi Mess, at Rs. 20 (vs. Rs. 22, 24, 25 at the other three shops).",
        "menu", expected_tools=("search_menus",),
        must_include=(("thanjai",), ("20",)),
    ),
    RaceCase(
        2, "Which is cheaper in Indian Rupees: the Grand Vista Wagyu Beef Burger or the Chettinad Chicken Meals?",
        "The Chettinad Chicken Meals at Sri Lakshmi Vilas (Rs. 220) is cheaper; the Wagyu Beef Burger at "
        "Grand Vista Hotel is 34 SGD, about Rs. 2074.",
        "menu", expected_tools=("search_menus", "search_menus", "convert_currency"),
        must_include=(("chettinad", "chicken meals"), ("220",), ("2074",)),
    ),
    RaceCase(
        3, "What's the price of Uttapam at Vaigai Tiffin Center?",
        "Uttapam is not on Vaigai Tiffin Center's menu, or any menu here -- not found.",
        "menu", expected_tools=("search_menus",),
        must_include=(NOT_FOUND,),
    ),
    # -- ingredient only ----------------------------------------------------
    RaceCase(
        4, "Is Rava Dosa gluten-free?",
        "No. Rava Dosa contains gluten (semolina and maida), per the standard recipe.",
        "ingredient", expected_tools=("lookup_ingredients",),
        must_include=(("not gluten-free", "not gluten free", "isn't gluten-free", "isn't gluten free",
                       "contains gluten", "has gluten", "includes gluten", "gluten is", "allergens: gluten",
                       "no,", "no.", "no -", "no —", "no –"),),
        must_not_include=("is gluten-free.", "yes,"),
    ),
    RaceCase(
        5, "Which vegan dishes are free from gluten?",
        "Seven: Plain Idli, Thanjavur Vengaya Dosai, Medu Vada, Idiyappam, Sambar Rice, Paruthi Paal, and the "
        "Selection of TWG Teas (standard recipes; confirm with the restaurant for a severe allergy).",
        "ingredient", expected_tools=("find_dishes",),
        notes="Week 9's single agent dropped Thanjavur Vengaya Dosai and Sambar Rice here -- the full list is the test.",
        must_include=(("idli",), ("vengaya",), ("medu vada", "vada"), ("idiyappam",), ("sambar rice",),
                      ("paruthi",), ("tea", "twg")),
    ),
    RaceCase(
        6, "Does the Hainanese Chicken Rice contain sesame?",
        "Yes -- it is made with sesame oil; its allergens are soy, sesame and gluten.",
        "ingredient", expected_tools=("lookup_ingredients",),
        must_include=(("sesame",), CONTAINS),
    ),
    # -- combined: both specialists needed ----------------------------------
    RaceCase(
        7, "I'm allergic to dairy. Is Masala Dosa safe for me, and which shop sells it cheapest?",
        "Not safe: the standard recipe uses ghee (dairy). Cheapest is Sri Kaveri Bhavan at Rs. 70 (Sri Lakshmi "
        "Vilas: Rs. 75).",
        "combined", expected_tools=("lookup_ingredients", "search_menus"),
        must_include=(("ghee", "dairy"), NOT_SAFE, ("kaveri",), ("70",)),
    ),
    RaceCase(
        8, "I'm vegan. Is Medu Vada vegan, and where is it cheapest?",
        "Yes, Medu Vada is vegan (standard recipe). Cheapest at Vaigai Tiffin Center, Rs. 35.",
        "combined", expected_tools=("lookup_ingredients", "search_menus"),
        must_include=(("vegan",), ("vaigai",), ("35",)),
        must_not_include=("not vegan", "isn't vegan"),
    ),
    RaceCase(
        9, "Is Ven Pongal safe for someone with a tree-nut allergy, and what does it cost at each shop?",
        "No -- it contains cashews (tree nuts). Sri Kaveri Bhavan Rs. 55, Sri Lakshmi Vilas Rs. 58.",
        "combined", expected_tools=("lookup_ingredients", "search_menus"),
        must_include=(("cashew", "tree nut", "tree-nut"), ("kaveri",), ("55",), ("lakshmi",), ("58",)),
    ),
    RaceCase(
        10, "Which is cheaper in rupees, the Classic Tiramisu or the Paal Payasam, and which of them is alcohol-free?",
        "Paal Payasam (Thanjai Ruchi Mess, Rs. 45) is far cheaper than the Classic Tiramisu (Grand Vista Hotel, "
        "16 SGD = about Rs. 976). The Paal Payasam is alcohol-free; the tiramisu contains alcohol (marsala wine).",
        "combined", expected_tools=("search_menus", "search_menus", "convert_currency",
                                    "lookup_ingredients", "lookup_ingredients"),
        must_include=(("payasam",), ("45",), ("976",), ("alcohol", "marsala")),
    ),
    RaceCase(
        11, "I can't eat gluten. At Vaigai Tiffin Center, how much are the Plain Idli and the Rava Dosa, "
            "and which of them can I eat?",
        "Plain Idli Rs. 30 -- gluten-free, so you can eat it. Rava Dosa Rs. 60 -- contains gluten (semolina, "
        "maida), so avoid it.",
        "combined", expected_tools=("search_menus", "search_menus", "lookup_ingredients", "lookup_ingredients"),
        must_include=(("30",), ("60",), ("idli",), ("rava",), ("gluten",)),
    ),
    # -- not found in either source -----------------------------------------
    RaceCase(
        12, "Is the Margherita Pizza dairy-free, and how much does it cost?",
        "Margherita Pizza is not on any menu and not in the ingredient database -- not found, so it can't be "
        "priced or checked for dairy.",
        "not_found", expected_tools=("search_menus", "lookup_ingredients"),
        must_include=(NOT_FOUND,),
    ),
]
