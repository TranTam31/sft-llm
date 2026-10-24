"""P, the fixed probe set read by eye at every checkpoint (PLAN 3.2): 10 chat, 10 knowledge and 40 tool prompts
over the six tool groups. Written for this project, not drawn from xLAM, Glaive or SmolTalk, so it is tracked in Git.

Run: python -m src.eval.probe_set  -> eval_sets/p.jsonl
Answers are reading aids, not scores. meta.domain says whether a tool's function sits in the seen or the unseen
clusters (data/SPLIT.md), judged by hand from the cluster topics; the tools themselves are new.
"""

import argparse
import hashlib
import json
from pathlib import Path

from src.tool_calls import format_tool_calls, format_tool_responses, parse_tool_calls, validate_call


def tool(name, description, params, required=None):
    properties = {key: {"type": kind, "description": text} for key, (kind, text) in params.items()}
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": list(params) if required is None else required}}


# Seen-side functions: weather, finance, recipes, movies, news, translation.
WEATHER = tool("get_weather", "Get the current weather for a city.",
               {"city": ("string", "City name."), "unit": ("string", "'celsius' or 'fahrenheit'.")}, ["city"])
STOCK = tool("get_stock_price", "Get the latest trading price of a stock.",
             {"symbol": ("string", "Ticker symbol, e.g. 'AAPL'.")})
RECIPES = tool("search_recipes", "Find recipes that use an ingredient.",
               {"ingredient": ("string", "Main ingredient."), "max_results": ("integer", "How many recipes to return.")},
               ["ingredient"])
CURRENCY = tool("convert_currency", "Convert an amount of money from one currency to another.",
                {"amount": ("number", "Amount to convert."), "from_currency": ("string", "ISO code to convert from."),
                 "to_currency": ("string", "ISO code to convert to.")})
SHOWTIMES = tool("find_movie_showtimes", "List showtimes for a movie in a city.",
                 {"movie": ("string", "Movie title."), "city": ("string", "City name."),
                  "date": ("string", "Date as YYYY-MM-DD.")}, ["movie", "city"])
NEWS = tool("get_news_headlines", "Get the latest news headlines on a topic.",
            {"topic": ("string", "News topic."), "country": ("string", "Two-letter country code.")}, ["topic"])
TRANSLATE = tool("translate_text", "Translate text into another language.",
                 {"text": ("string", "Text to translate."), "target_language": ("string", "Language to translate into.")})
# Unseen-side functions: sports fixtures and standings, messaging, string checks, tips, statistics.
FIXTURES = tool("get_match_schedule", "Get upcoming matches for a football team.",
                {"team": ("string", "Team name."), "date": ("string", "Date as YYYY-MM-DD.")}, ["team"])
STANDINGS = tool("get_league_standings", "Get the current table of a football league.",
                 {"league": ("string", "League name."), "season": ("string", "Season, e.g. '2025-26'.")}, ["league"])
EMAIL = tool("send_email", "Send an email.",
             {"to": ("string", "Recipient address."), "subject": ("string", "Subject line."),
              "body": ("string", "Message text.")})
SMS = tool("send_sms", "Send a text message to a phone number.",
           {"phone_number": ("string", "Phone number with country code."), "message": ("string", "Message text.")})
PALINDROME = tool("is_palindrome", "Check whether a text reads the same backwards, ignoring case and spaces.",
                  {"text": ("string", "Text to check.")})
TIP = tool("calculate_tip", "Calculate the tip for a bill.",
           {"bill_amount": ("number", "Bill total."), "tip_percent": ("number", "Tip percentage.")})
STD_DEV = tool("calculate_std_dev", "Calculate the standard deviation of a list of numbers.",
               {"numbers": ("array", "The numbers.")})

SEEN = {tool_["name"] for tool_ in (WEATHER, STOCK, RECIPES, CURRENCY, SHOWTIMES, NEWS, TRANSLATE)}


def call(tool_, **arguments):
    return {"name": tool_["name"], "arguments": arguments}


def user(text):
    return [{"role": "user", "content": text}]


CHAT = [
    "Write a four-line poem about the first rain of the monsoon.",
    "Give me three practical tips for falling asleep faster.",
    "Rewrite this sentence to sound more polite: 'Send me the report now.'",
    "Explain to a ten-year-old why we have to brush our teeth.",
    "Suggest a name for a small coffee shop next to a bookstore, and say why.",
    "Summarise the plot of Romeo and Juliet in two sentences.",
    "Write a short thank-you note to a neighbour who watered my plants.",
    "List the pros and cons of working from home, three of each.",
    "Help me plan a simple 20-minute morning workout without equipment.",
    "Turn these notes into one clear sentence: meeting moved, Thursday, 3 pm, room B.",
]

KNOWLEDGE = [
    ("What is the capital of Australia?", "Canberra."),
    ("How many sides does a hexagon have?", "Six."),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen."),
    ("What gas do plants take in for photosynthesis?", "Carbon dioxide."),
    ("What is 15% of 80?", "12."),
    ("Which planet is closest to the Sun?", "Mercury."),
    ("In which year did the Second World War end?", "1945."),
    ("What is the chemical symbol for gold?", "Au."),
    ("What is the boiling point of water at sea level in Celsius?", "100 °C."),
    ("Why is the sky blue?", "Air scatters short blue wavelengths of sunlight more than red ones (Rayleigh scattering)."),
]

SINGLE = [
    ([WEATHER], "What's the weather like in Hanoi right now?", [call(WEATHER, city="Hanoi")]),
    ([STOCK], "What is NVIDIA trading at? The ticker is NVDA.", [call(STOCK, symbol="NVDA")]),
    ([CURRENCY], "How many euros is 250 US dollars?", [call(CURRENCY, amount=250, from_currency="USD", to_currency="EUR")]),
    ([TRANSLATE], "Translate 'Where is the train station?' into Japanese.",
     [call(TRANSLATE, text="Where is the train station?", target_language="Japanese")]),
    ([FIXTURES], "When does Arsenal play next?", [call(FIXTURES, team="Arsenal")]),
    ([PALINDROME], "Is 'Never odd or even' a palindrome?", [call(PALINDROME, text="Never odd or even")]),
    ([TIP], "My dinner bill is $84. How much is an 18% tip?", [call(TIP, bill_amount=84, tip_percent=18)]),
]

SELECT = [
    ([WEATHER, STOCK, NEWS], "Show me today's headlines about electric cars.", [call(NEWS, topic="electric cars")]),
    ([RECIPES, SHOWTIMES, CURRENCY], "I have a lot of spinach. Find me two recipes with it.",
     [call(RECIPES, ingredient="spinach", max_results=2)]),
    ([TRANSLATE, WEATHER, SHOWTIMES], "Where can I watch Dune: Part Two in Chicago?",
     [call(SHOWTIMES, movie="Dune: Part Two", city="Chicago")]),
    ([STOCK, CURRENCY, NEWS], "Convert 1000 Japanese yen to Vietnamese dong.",
     [call(CURRENCY, amount=1000, from_currency="JPY", to_currency="VND")]),
    ([EMAIL, SMS, FIXTURES], "Text +84901234567: 'Running 10 minutes late.'",
     [call(SMS, phone_number="+84901234567", message="Running 10 minutes late.")]),
    ([STANDINGS, TIP, STD_DEV], "What's the standard deviation of 4, 8, 15, 16, 23 and 42?",
     [call(STD_DEV, numbers=[4, 8, 15, 16, 23, 42])]),
    ([PALINDROME, STANDINGS, EMAIL], "Show me the current Serie A table.", [call(STANDINGS, league="Serie A")]),
]

PARALLEL = [
    ([WEATHER], "Compare the weather in Tokyo and Seoul right now.", [call(WEATHER, city="Tokyo"), call(WEATHER, city="Seoul")]),
    ([STOCK], "Get me the latest prices for AAPL, MSFT and GOOGL.",
     [call(STOCK, symbol="AAPL"), call(STOCK, symbol="MSFT"), call(STOCK, symbol="GOOGL")]),
    ([WEATHER, NEWS], "What's the weather in London, and what are the top news stories about the UK economy?",
     [call(WEATHER, city="London"), call(NEWS, topic="UK economy")]),
    ([TRANSLATE], "Translate 'good morning' into French and into Spanish.",
     [call(TRANSLATE, text="good morning", target_language="French"),
      call(TRANSLATE, text="good morning", target_language="Spanish")]),
    ([FIXTURES], "When do Liverpool and Chelsea play next?", [call(FIXTURES, team="Liverpool"), call(FIXTURES, team="Chelsea")]),
    ([PALINDROME], "Check which of these are palindromes: 'racecar', 'level', 'banana'.",
     [call(PALINDROME, text="racecar"), call(PALINDROME, text="level"), call(PALINDROME, text="banana")]),
    ([TIP, STANDINGS], "Work out a 15% tip on a $60 bill, and show me the Premier League table.",
     [call(TIP, bill_amount=60, tip_percent=15), call(STANDINGS, league="Premier League")]),
]

# kind as in the eval sets: plain (unrelated tools), related (same topic, not needed), no_fitting_tool (needs data no tool gives).
NO_TOOL = [
    ([STOCK, FIXTURES], "Can you give me a recipe idea for a quick vegetarian lunch?", "plain",
     "A direct answer, e.g. a chickpea salad wrap, with no tool call."),
    ([PALINDROME, CURRENCY], "What is the difference between a virus and a bacterium?", "plain",
     "A direct explanation, with no tool call."),
    ([EMAIL, SHOWTIMES], "Write a limerick about a cat who loves boxes.", "plain", "A limerick, with no tool call."),
    ([WEATHER], "Why do clouds look grey before it rains?", "related",
     "Thick clouds with large droplets block and scatter more light; answered directly without get_weather."),
    ([TIP], "Is it rude not to tip in Japan?", "related",
     "Tipping is not customary in Japan and can cause confusion; answered directly without calculate_tip."),
    ([TRANSLATE], "What does the French phrase 'déjà vu' mean in English?", "related",
     "Literally 'already seen': the feeling of having lived a moment before; no translate_text call needed."),
    ([RECIPES, PALINDROME], "What's the price of a flight from Hanoi to Bangkok next Friday?", "no_fitting_tool",
     "None of the tools I have can look up flight prices."),
]

MISSING_INFO = [
    ([WEATHER], "What's the weather like there today?", "city", ["city", "location", "place", "where"]),
    ([CURRENCY], "Convert 300 dollars for me, please.", "to_currency",
     ["to currency", "currency", "which currency", "convert to"]),
    ([SHOWTIMES], "When is Inside Out 2 showing?", "city", ["city", "location", "where", "town"]),
    ([EMAIL], "Email my landlord that the sink is leaking.", "to",
     ["email address", "address", "recipient", "landlord's email"]),
    ([TIP], "I want to leave a 20% tip. How much is that?", "bill_amount",
     ["bill amount", "bill", "amount", "total"]),
    ([SMS], "Send a text saying I'll be home by 7.", "phone_number",
     ["phone number", "number", "phone", "who"]),
]

MULTI_TURN = [
    (WEATHER, "Is it raining in Paris?", call(WEATHER, city="Paris"),
     {"city": "Paris", "condition": "light rain", "temp_c": 12}, ["light rain", 12],
     "Yes, there is light rain in Paris right now, and it is 12 °C."),
    (STOCK, "How is Tesla stock doing? Ticker TSLA.", call(STOCK, symbol="TSLA"),
     {"symbol": "TSLA", "price": 241.37, "change_percent": -2.4}, [241.37],
     "Tesla (TSLA) is trading at $241.37, down 2.4% today."),
    (SHOWTIMES, "Find showtimes for Wicked in Boston.", call(SHOWTIMES, movie="Wicked", city="Boston"),
     {"movie": "Wicked", "showtimes": ["14:10", "17:45", "21:00"]}, ["14:10", "17:45", "21:00"],
     "Wicked is showing in Boston at 14:10, 17:45 and 21:00."),
    (FIXTURES, "When does Barcelona play next?", call(FIXTURES, team="Barcelona"),
     {"team": "Barcelona", "next_match": {"opponent": "Sevilla", "date": "2026-11-02", "venue": "home"}},
     ["Sevilla", "2026-11-02"], "Barcelona play Sevilla at home on 2026-11-02."),
    (EMAIL, "Email anna@example.com that the meeting is cancelled.",
     call(EMAIL, to="anna@example.com", subject="Meeting cancelled", body="The meeting is cancelled."),
     {"status": "sent", "message_id": "m-48213"}, ["sent"], "Done: the email to anna@example.com was sent."),
    (STD_DEV, "Standard deviation of 2, 4, 4, 4, 5, 5, 7, 9?", call(STD_DEV, numbers=[2, 4, 4, 4, 5, 5, 7, 9]),
     {"std_dev": 2.0}, [2.0], "The standard deviation is 2.0."),
]


def domain(tools):
    sides = {"seen" if tool_["name"] in SEEN else "unseen" for tool_ in tools}
    return sides.pop() if len(sides) == 1 else "mixed"


def build() -> list[dict]:
    items = []

    def add(group, index, tools, turns, **fields):
        meta = {"domain": domain(tools) if tools else "none"}
        items.append({"id": f"p:{group}:{index:02d}", "set": "p", "group": group, "tools": tools, "turns": turns,
                      "expected_calls": fields.pop("expected_calls", []), **fields, "meta": meta})

    for index, text in enumerate(CHAT, 1):
        add("chat", index, [], user(text))
    for index, (text, answer) in enumerate(KNOWLEDGE, 1):
        add("knowledge", index, [], user(text), reference=answer)
    for group, rows in (("single", SINGLE), ("select", SELECT), ("parallel", PARALLEL)):
        for index, (tools, text, calls) in enumerate(rows, 1):
            add(group, index, tools, user(text), expected_calls=calls)
    for index, (tools, text, kind, answer) in enumerate(NO_TOOL, 1):
        add("no_tool", index, tools, user(text), kind=kind, reference=answer)
    for index, (tools, text, missing, synonyms) in enumerate(MISSING_INFO, 1):
        add("missing_info", index, tools, user(text), missing_param=missing, synonyms=synonyms)
    for index, (tool_, text, made, result, required, answer) in enumerate(MULTI_TURN, 1):
        turns = [*user(text), {"role": "assistant", "content": format_tool_calls([made])},
                 {"role": "tool", "content": format_tool_responses([{"name": made["name"], "content": result}])}]
        add("multi_turn", index, [tool_], turns, required=required, forbidden=[], reference=answer)
    return items


def check(items: list[dict]):
    """Every call names an offered tool with declared arguments and all required ones; prompts are unique."""
    for item in items:
        offered = {tool_["name"]: tool_ for tool_ in item["tools"]}
        calls = list(item["expected_calls"])
        if item["group"] == "multi_turn":
            calls.append(json.loads(item["turns"][1]["content"].split("\n")[1]))
        for made in calls:
            validate_call(made)
            schema = offered[made["name"]]["parameters"]
            assert set(made["arguments"]) <= set(schema["properties"]), item["id"]
            assert set(schema["required"]) <= set(made["arguments"]), item["id"]
        if item["group"] == "missing_info":
            schema = item["tools"][0]["parameters"]
            assert item["missing_param"] in schema["required"], item["id"]
    prompts = [item["turns"][0]["content"].lower() for item in items]
    assert len(prompts) == len(set(prompts)), "repeated prompt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eval_sets/p.jsonl")
    args = parser.parse_args()
    items = build()
    check(items)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8", newline="\n")
    print(f"[probe_set] {path}: {len(items)} items, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
