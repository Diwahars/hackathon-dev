import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


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
    }

    if time_range in {"day", "week", "month", "year"}:
        search_kwargs["time_range"] = time_range

    data = client.search(**search_kwargs)

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