import json
import re

from datetime import datetime, timezone

from bifrost_client import chat
from tavily_client import search_web
from observability import Trace

STALE_THRESHOLD = 0.70


def classify_freshness(query: str):
    response = chat(
        messages=[
            {
                "role": "system",
                "content": """
Classify whether a user question requires
current information.

Return JSON only:
{
  "time_sensitive": true,
  "reason": "short reason",
  "search_query": "web search query"
}

Mark true for current prices, recent events,
latest versions, current policies, or changing facts.
Mark false for timeless concepts and stable facts.
""",
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    return json.loads(content)


def extract_claims(answer: str):
    response = chat(
        messages=[
            {
                "role": "system",
                "content": """
Extract factual claims that may become outdated.

Return JSON only:
{
  "claims": [
    {
      "claim": "claim text",
      "search_query": "query to verify"
    }
  ]
}

Extract claims about dates, releases, prices,
policies, availability, and changing information.
Do not extract opinions or timeless facts.
""",
            },
            {
                "role": "user",
                "content": answer,
            },
        ],
        temperature=0,
    )

    return json.loads(
        response.choices[0].message.content
    )


def verify_claims(claims, evidence):
    response = chat(
        messages=[
            {
                "role": "system",
                "content": """
You are a factual freshness evaluator.

Compare each claim against the supplied evidence.

Return JSON only:
{
  "results": [
    {
      "claim": "claim text",
      "status": "supported",
      "confidence": 0.95,
      "explanation": "short explanation"
    }
  ]
}

Allowed statuses:
supported
stale
contradicted
insufficient_evidence

Rules:
- Do not treat missing evidence as contradiction.
- Do not assume search results are authoritative.
- Consider publication dates and source quality.
- Distinguish old but historically correct facts
  from claims presented as current.
- Use insufficient_evidence when unsure.
""",
            },
            {
                "role": "user",
                "content": json.dumps({
                    "claims": claims,
                    "evidence": evidence,
                    "current_date": datetime.now(
                        timezone.utc
                    ).date().isoformat(),
                }),
            },
        ],
        temperature=0,
    )

    return json.loads(
        response.choices[0].message.content
    )


def detect_stale_answer(
    query,
    answer,
    trace=None,
    parent_span=None,
):
    """
    Run freshness verification with trace spans.

    If trace is None, create a standalone trace.
    """

    if trace is None:
        trace = Trace(query)

        parent_span = trace.start_span(
            name="freshness.evaluate",
            kind="freshness",
        )

    # ------------------------------------------
    # 1. Classify freshness requirements
    # ------------------------------------------

    with trace.span(
        name="freshness.classify",
        kind="llm",
        parent_span=parent_span,
        attributes={
            "llm.provider": "bifrost",
        },
    ) as classify_span:

        classification = classify_freshness(query)

        classify_span.set_attributes(
            **{
                "freshness.time_sensitive": (
                    classification.get(
                        "time_sensitive",
                        False,
                    )
                ),
            }
        )

    if not classification.get("time_sensitive"):

        result = {
            "time_sensitive": False,
            "stale": False,
            "score": 0,
            "results": [],
        }

        parent_span.set_attributes(
            **{
                "freshness.stale": False,
                "freshness.score": 0,
            }
        )

        return result

    # ------------------------------------------
    # 2. Extract time-sensitive claims
    # ------------------------------------------

    with trace.span(
        name="freshness.extract_claims",
        kind="llm",
        parent_span=parent_span,
        attributes={
            "llm.provider": "bifrost",
        },
    ) as extraction_span:

        extracted = extract_claims(answer)

        claims = extracted.get("claims", [])

        extraction_span.set_attributes(
            **{
                "freshness.claim_count": len(claims),
            }
        )

    if not claims:

        result = {
            "time_sensitive": True,
            "stale": False,
            "score": 0,
            "results": [],
        }

        parent_span.set_attributes(
            **{
                "freshness.stale": False,
                "freshness.score": 0,
            }
        )

        return result

    # ------------------------------------------
    # 3. Retrieve current evidence
    # ------------------------------------------

    evidence = []

    for index, claim in enumerate(claims):

        search_query = claim.get("search_query")

        if not search_query:
            continue

        with trace.span(
            name="freshness.tavily_search",
            kind="tool",
            parent_span=parent_span,
            attributes={
                "tool.name": "tavily",
                "tool.query": search_query,
                "freshness.claim_index": index,
            },
        ) as search_span:

            try:

                from tavily_client import search_web

                search_result = search_web(
                    search_query
                )

                results = search_result.get(
                    "results",
                    [],
                )

                evidence.extend(results)

                search_span.set_attributes(
                    **{
                        "tool.result_count": len(results),
                        "tool.status": "success",
                    }
                )

            except Exception as exc:

                search_span.set_attributes(
                    **{
                        "tool.status": "error",
                        "error.message": str(exc),
                    }
                )

    # ------------------------------------------
    # 4. Verify claims against evidence
    # ------------------------------------------

    with trace.span(
        name="freshness.verify_claims",
        kind="llm",
        parent_span=parent_span,
        attributes={
            "llm.provider": "bifrost",
            "freshness.claim_count": len(claims),
            "freshness.evidence_count": len(evidence),
        },
    ) as verification_span:

        verification = verify_claims(
            claims=claims,
            evidence=evidence,
        )

        results = verification.get(
            "results",
            [],
        )

    # ------------------------------------------
    # 5. Individual claim spans
    # ------------------------------------------

    for index, item in enumerate(results):

        with trace.span(
            name="freshness.claim_verification",
            kind="evaluation",
            parent_span=verification_span,
            attributes={
                "freshness.claim_index": index,
                "freshness.claim": item.get("claim"),
                "freshness.claim_status": (
                    item.get("status")
                ),
                "freshness.confidence": (
                    item.get("confidence")
                ),
            },
        ):
            pass

    # ------------------------------------------
    # 6. Calculate stale score
    # ------------------------------------------

    stale_results = [
        item for item in results
        if item.get("status") in (
            "stale",
            "contradicted",
        )
    ]

    stale_score = (
        len(stale_results) / len(results)
        if results
        else 0
    )

    stale = stale_score >= 0.70

    result = {
        "time_sensitive": True,
        "stale": stale,
        "score": round(stale_score, 3),
        "results": results,
        "evidence": evidence,
    }

    parent_span.set_attributes(
        **{
            "freshness.stale": stale,
            "freshness.score": round(
                stale_score,
                3,
            ),
            "freshness.verified_claims": len(results),
            "freshness.stale_claims": len(
                stale_results
            ),
        }
    )

    return result

    classification = classify_freshness(query)

    if not classification.get("time_sensitive"):
        return {
            "time_sensitive": False,
            "stale": False,
            "score": 0,
            "results": [],
        }

    extracted = extract_claims(answer)

    claims = extracted.get("claims", [])

    if not claims:
        return {
            "time_sensitive": True,
            "stale": False,
            "score": 0,
            "results": [],
        }

    evidence = []

    for item in claims:
        search_query = item.get("search_query")

        if not search_query:
            continue

        result = search_web(search_query)

        evidence.extend(
            result.get("results", [])
        )

    verification = verify_claims(
        claims=claims,
        evidence=evidence,
    )

    results = verification.get("results", [])

    if not results:
        return {
            "time_sensitive": True,
            "stale": False,
            "score": 0,
            "results": [],
        }

    stale_results = [
        item for item in results
        if item.get("status") in (
            "stale",
            "contradicted",
        )
    ]

    stale_score = (
        len(stale_results) / len(results)
    )

    return {
        "time_sensitive": True,
        "stale": stale_score >= STALE_THRESHOLD,
        "score": round(stale_score, 3),
        "results": results,
        "evidence": evidence,
    }