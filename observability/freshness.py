import json
import re
from datetime import datetime, timezone

from bifrost_client import chat
from tavily_client import search_web
from observability import Trace


STALE_THRESHOLD = 0.70


def _parse_json(content: str) -> dict:
    """
    Parse JSON returned by the LLM.
    Handles plain JSON and JSON wrapped in Markdown fences.
    """
    if not content:
        raise ValueError("LLM returned an empty response.")

    content = content.strip()

    # Remove Markdown JSON fences if present.
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Attempt to extract the first JSON object.
        match = re.search(r"\{.*\}", content, re.DOTALL)

        if not match:
            raise ValueError(
                f"Could not parse JSON from LLM response: {content}"
            )

        return json.loads(match.group(0))


def classify_freshness(query: str) -> dict:
    """
    Classify whether a query depends on current information.
    """
    response = chat(
        messages=[
            {
                "role": "system",
                "content": """
Classify whether a user question requires current information.

Return JSON only:
{
  "time_sensitive": true,
  "reason": "short reason",
  "search_query": "web search query"
}

Mark true for current prices, recent events, latest versions,
current policies, or changing facts.

Mark false for timeless concepts and stable facts.

Do not wrap the JSON in Markdown fences.
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
    result = _parse_json(content)

    return {
        "time_sensitive": bool(
            result.get("time_sensitive", False)
        ),
        "reason": str(result.get("reason", "")),
        "search_query": str(result.get("search_query", query)),
    }


def extract_claims(answer: str) -> dict:
    """
    Extract factual claims from the answer that may become outdated.
    """
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

Extract claims about dates, releases, prices, policies,
availability, and changing information.

Do not extract opinions or timeless facts.
Do not wrap the JSON in Markdown fences.
""",
            },
            {
                "role": "user",
                "content": answer,
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    result = _parse_json(content)

    claims = result.get("claims", [])

    if not isinstance(claims, list):
        return {"claims": []}

    return {"claims": claims}


def verify_claims(claims: list, evidence: list) -> dict:
    """
    Compare extracted claims against current web evidence.
    """
    current_date = datetime.now(timezone.utc).date().isoformat()

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
- supported
- stale
- contradicted
- insufficient_evidence

Rules:
- Do not treat missing evidence as contradiction.
- Do not assume search results are authoritative.
- Consider publication dates and source quality.
- Distinguish old but historically correct facts
  from claims presented as current.
- Use insufficient_evidence when unsure.
- Do not wrap the JSON in Markdown fences.
""",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "claims": claims,
                        "evidence": evidence,
                        "current_date": current_date,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    result = _parse_json(content)

    results = result.get("results", [])

    if not isinstance(results, list):
        return {"results": []}

    return {"results": results}


def detect_stale_answer(
    query,
    answer,
    trace=None,
    parent_span=None,
):
    """
    Run freshness verification with trace spans.

    If trace is None, create a standalone trace.
    Returns:
        time_sensitive: bool
        stale: bool
        score: float
        results: list
        evidence: list
    """

    # Create a trace and parent span when called standalone.
    if trace is None:
        trace = Trace(query)

    if parent_span is None:
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

        time_sensitive = classification.get(
            "time_sensitive",
            False,
        )

        classify_span.set_attributes(
            **{
                "freshness.time_sensitive": time_sensitive,
                "freshness.reason": classification.get(
                    "reason",
                    "",
                ),
            }
        )

    if not time_sensitive:
        result = {
            "time_sensitive": False,
            "stale": False,
            "score": 0.0,
            "results": [],
            "evidence": [],
            "evaluation_status": "not_time_sensitive",
        }

        parent_span.set_attributes(
            **{
                "freshness.stale": False,
                "freshness.score": 0.0,
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
            "score": 0.0,
            "results": [],
            "evidence": [],
            "evaluation_status": "no_time_sensitive_claims",
        }

        parent_span.set_attributes(
            **{
                "freshness.stale": False,
                "freshness.score": 0.0,
            }
        )

        return result

    # ------------------------------------------
    # 3. Retrieve current evidence from Tavily
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
                search_result = search_web(search_query)

                results = search_result.get("results", [])

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

        if evidence:
            verification = verify_claims(
                claims=claims,
                evidence=evidence,
            )

            results = verification.get("results", [])
        else:
            # No evidence is not proof that the answer is current.
            results = [
                {
                    "claim": claim.get("claim", ""),
                    "status": "insufficient_evidence",
                    "confidence": 0.0,
                    "explanation": (
                        "No current web evidence was retrieved "
                        "to verify this claim."
                    ),
                }
                for claim in claims
            ]

        verification_span.set_attributes(
            **{
                "freshness.verified_claims": len(results),
            }
        )

    # ------------------------------------------
    # 5. Individual claim verification spans
    # ------------------------------------------

    for index, item in enumerate(results):
        with trace.span(
            name="freshness.claim_verification",
            kind="evaluation",
            parent_span=verification_span,
            attributes={
                "freshness.claim_index": index,
                "freshness.claim": item.get("claim", ""),
                "freshness.claim_status": item.get("status", ""),
                "freshness.confidence": item.get(
                    "confidence",
                    0.0,
                ),
            },
        ):
            pass

    # ------------------------------------------
    # 6. Calculate stale score
    # ------------------------------------------

    stale_results = [
        item
        for item in results
        if item.get("status") in ("stale", "contradicted")
    ]

    insufficient_results = [
        item
        for item in results
        if item.get("status") == "insufficient_evidence"
    ]

    # Only calculate the stale ratio from claims that were
    # actually verified. Insufficient evidence is not stale evidence.
    verified_results = [
        item
        for item in results
        if item.get("status") in (
            "supported",
            "stale",
            "contradicted",
        )
    ]

    stale_score = (
        len(stale_results) / len(verified_results)
        if verified_results
        else 0.0
    )

    stale = stale_score >= STALE_THRESHOLD

    if not evidence or not verified_results:
        evaluation_status = "insufficient_evidence"
    elif insufficient_results:
        evaluation_status = "partially_verified"
    else:
        evaluation_status = "verified"

    result = {
        "time_sensitive": True,
        "stale": stale,
        "score": round(stale_score, 3),
        "results": results,
        "evidence": evidence,
        "evaluation_status": evaluation_status,
    }

    parent_span.set_attributes(
        **{
            "freshness.stale": stale,
            "freshness.score": round(stale_score, 3),
            "freshness.verified_claims": len(verified_results),
            "freshness.stale_claims": len(stale_results),
            "freshness.evaluation_status": evaluation_status,
        }
    )

    return result