import os
import httpx
from dotenv import load_dotenv

load_dotenv()

TAVILY_URL = "https://api.tavily.com/search"


def search_web(query: str, max_results: int = 5):
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    response = httpx.post(
        TAVILY_URL,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    return {
        "query": query,
        "results": [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
                "published_date": item.get("published_date"),
            }
            for item in data.get("results", [])
        ],
    }