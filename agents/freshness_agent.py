import re
from bifrost_client import chat_text
from agents.json_utils import parse_json

# Strong temporal intent should not be overridden by an LLM classification.
TEMPORAL_CUES = re.compile(
    r"\b(latest|currently|current|today|tonight|this week|this month|recent|"
    r"newest|up[- ]to[- ]date|as of now|right now|202[5-9]|202[0-9]|"
    r"release date|version|price|stock|availability|outage|status|forecast|"
    r"news|breaking|election results|who is the (current|latest))\b", re.I
)

def run(query):
    heuristic = bool(TEMPORAL_CUES.search(query or ""))
    print(f"Heuristic: {heuristic}")
    prompt = f"""Does this query need up-to-date information?
Return JSON only: {{"time_sensitive": boolean, "reason": string, "search_query": string}}.
Query: {query}"""
    raw = chat_text([
        {"role": "system", "content": "Classify freshness requirements. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ], temperature=0)
    data = parse_json(raw, {})
    model_flag = bool(data.get("time_sensitive", False))
    print(f"Model flag: {model_flag}")
    return {
        "time_sensitive": heuristic or model_flag,
        "heuristic_triggered": heuristic,
        "reason": str(data.get("reason", "")),
        "search_query": str(data.get("search_query") or query),
    }
