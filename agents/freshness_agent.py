"""Classify whether a query depends on information that can change over time.

The deterministic rules are intentionally conservative: strong temporal or
volatile-data cues always trigger retrieval. The LLM classifier catches
implicit cases that are difficult to enumerate (for example, asking who
currently leads an organization without saying "current").
"""

import re
from bifrost_client import chat_text
from agents.json_utils import parse_json

# Explicit time language plus common volatile-data concepts. Keep this list
# domain-neutral; entity names and tickers must never be hardcoded here.
TEMPORAL_CUES = re.compile(
    r"\b(?:latest|current(?:ly)?|right now|as of now|today|tonight|"
    r"this (?:morning|afternoon|evening|week|month|quarter|year)|"
    r"recent(?:ly)?|newest|up[- ]to[- ]date|live|real[- ]time|"
    r"breaking|this year|202\d)\b",
    re.IGNORECASE,
)

VOLATILE_DATA_CUES = re.compile(
    r"\b(?:stock|share|crypto|bitcoin|exchange rate|currency rate|"
    r"market price|stock price|share price|stock quote|price of|"
    r"weather|forecast|temperature|news|announcement|release|"
    r"version|availability|inventory|flight status|traffic|"
    r"outage|service status|election result|score|standings|"
    r"schedule|opening hours|interest rate|inflation|population)\b",
    re.IGNORECASE,
)


def _heuristic_requires_freshness(query: str) -> bool:
    text = query or ""
    return bool(TEMPORAL_CUES.search(text) or VOLATILE_DATA_CUES.search(text))


def run(query: str) -> dict:
    """Return a stable classification contract consumed by ``agent.py``."""
    query = (query or "").strip()
    heuristic = _heuristic_requires_freshness(query)

    prompt = f"""Classify whether answering this query accurately requires
information that may have changed since the model's training data.

Consider facts that change frequently even when the user does not explicitly
say latest/current: market quotes, weather, availability, operational status,
software releases, current office holders, recent events and schedules.
Do not classify stable explanations, definitions or timeless how-to questions
as time-sensitive unless the user asks for a current state.

Return JSON only with this exact shape:
{{"time_sensitive": true, "reason": "short explanation",
  "search_query": "a concise web search query preserving the user's entity and intent"}}

User query: {query}"""

    model_data = {}
    try:
        raw = chat_text([
            {"role": "system", "content": "You classify information freshness needs. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ], temperature=0)
        model_data = parse_json(raw, {})
    except Exception as exc:
        # A classifier outage must not disable deterministic freshness rules.
        model_data = {"reason": f"LLM classification unavailable: {exc}"}

    model_flag = model_data.get("time_sensitive") is True
    time_sensitive = heuristic or model_flag
    return {
        "time_sensitive": time_sensitive,
        "heuristic_triggered": heuristic,
        "model_triggered": model_flag,
        "reason": str(model_data.get("reason") or (
            "A deterministic temporal/volatile-data cue was detected."
            if heuristic else "No strong freshness requirement was detected."
        )),
        "search_query": str(model_data.get("search_query") or query),
    }
