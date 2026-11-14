import os
from dotenv import load_dotenv
from tavily import TavilyClient
import re
from datetime import datetime, timedelta, timezone
load_dotenv()

DATE_REGEX = (
    r"\d{4}-\d{2}-\d{2}"
    r"|[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+[A-Za-z]+\s+\d{4}"
)

DATE_FORMATS = [
    "%Y-%m-%d",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d %B %Y",
    "%d %b %Y",
]


def normalize_date(value):
    if not value:
        return None
    
    value = value.strip()
    
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue

    return None


def extract_all_dates(content):
    if not content:
        return []

    now = datetime.now(timezone.utc)
    extracted = []

    # Absolute date patterns with context
    absolute_patterns = [
        ("release", rf"\brelease\s+date\s*:?\s*({DATE_REGEX})"),
        ("release", rf"\breleased\s*:?\s*({DATE_REGEX})"),
        ("published", rf"\bpublished\s*:?\s*({DATE_REGEX})"),
        ("updated", rf"\bupdated\s*:?\s*({DATE_REGEX})"),
        ("posted", rf"\bposted\s*:?\s*({DATE_REGEX})"),
        ("build", rf"\bbuild\s+time\s*:?\s*({DATE_REGEX})"),
        ("generic", rf"({DATE_REGEX})"),
    ]

    occupied = []

    for date_type, pattern in absolute_patterns:
        for match in re.finditer(
            pattern, content, re.IGNORECASE
        ):
            start, end = match.span(1)

            # Avoid extracting the same date multiple times
            if any(start < old_end and end > old_start
                   for old_start, old_end in occupied):
                continue

            raw_date = match.group(1)
            normalized = normalize_date(raw_date)

            if not normalized:
                continue

            # Capture surrounding context
            context = content[
                max(0, start - 100):
                min(len(content), end + 100)
            ].strip()

            extracted.append({
                "date": normalized,
                "raw_date": raw_date,
                "date_type": date_type,
                "date_source": "content_extraction",
                "context": context,
            })

            occupied.append((start, end))

    # Relative dates: 5d ago, 2mo ago, 1y ago
    relative_pattern = (
        r"\b(\d+)\s*(d|day|days|w|week|weeks|"
        r"mo|month|months|y|year|years|h|hour|hours)"
        r"\s*ago\b"
    )

    for match in re.finditer(
        relative_pattern, content, re.IGNORECASE
    ):
        value = int(match.group(1))
        unit = match.group(2).lower()

        if unit in ("d", "day", "days"):
            date = now - timedelta(days=value)
        elif unit in ("w", "week", "weeks"):
            date = now - timedelta(weeks=value)
        elif unit in ("mo", "month", "months"):
            date = now - timedelta(days=value * 30)
        elif unit in ("y", "year", "years"):
            date = now - timedelta(days=value * 365)
        elif unit in ("h", "hour", "hours"):
            date = now - timedelta(hours=value)
        else:
            continue

        start, end = match.span()

        context = content[
            max(0, start - 100):
            min(len(content), end + 100)
        ].strip()

        extracted.append({
            "date": date.date().isoformat(),
            "raw_date": match.group(0),
            "date_type": "relative",
            "date_source": "content_extraction",
            "context": context,
        })

    # Sort by position in the original content
    extracted.sort(
        key=lambda x: content.find(x["raw_date"])
    )

    return extracted

def search_web(
    query: str,
    max_results: int = 6,
    time_range: str | None = None,
):
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing")

    client = TavilyClient(api_key=api_key)

    search_kwargs = {
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": "basic",
        "include_raw_content": False,
        "include_published_date": True
    }

    if time_range in {"day", "week", "month", "year"}:
        search_kwargs["time_range"] = time_range

    data = client.search(**search_kwargs)
    results = []
    for item in data.get("results", []):

        dates = extract_all_dates(
            item.get("content", "")
        )
        #print(f"content: {item.get('content', '')}")
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
            "score": item.get("score"),

            "published_date": (
                item.get("published_date")
                or item.get("publishedDate")
            ),
        })
    return {
        "query": query,
        "answer": data.get("answer", ""),
        "time_range": time_range,
        "results": results,
    }
    '''
    return {
        "query": query,
        "time_range": time_range,
        "results": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score"),
                "published_date": (
                    item.get("published_date")
                    or item.get("publishedDate")
                ),
            }
            for item in data.get("results", [])
        ],
    }
    '''