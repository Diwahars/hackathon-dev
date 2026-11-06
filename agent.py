"""Coordinator for parallel baseline generation and evidence-grounded refresh."""

from concurrent.futures import ThreadPoolExecutor
import os

from observability.trace import Trace
from agents import research_agent, freshness_agent, evidence_agent
from agents import verification_agent, refresh_agent


def _safe_unverified_answer(reason: str) -> str:
    reason = (reason or "Freshness or evidence could not be established.").strip()
    return (
        "I couldn't verify a sufficiently current answer from the retrieved "
        f"web evidence. {reason} Please try again or provide a trusted, recent source."
    )


def _search_time_range() -> str | None:
    """Read Tavily's coarse time filter without treating it as proof of freshness."""
    value = os.getenv("TAVILY_TIME_RANGE", "month").strip().lower()
    return value if value in {"day", "week", "month", "year"} else None


def _run_static_query(query: str, trace: Trace) -> str:
    with trace.span("research_agent", {"query": query, "mode": "general"}) as span:
        answer = research_agent.run(query)
        span.set_output({"answer": answer})
    return answer


def _generate_baseline(query: str, trace: Trace) -> str:
    """Run the original LLM-only answer branch."""
    with trace.span(
        "original_research_agent",
        {"query": query, "mode": "llm_only_baseline"},
    ) as span:
        answer = research_agent.run_baseline(query)
        span.set_output({"answer": answer})
    return answer


def _retrieve_evidence(search_query: str, time_range: str | None, trace: Trace) -> dict:
    """Run the Tavily evidence branch independently of baseline generation."""
    with trace.span(
        "evidence_search_agent",
        {"query": search_query, "time_range": time_range},
    ) as span:
        try:
            evidence = evidence_agent.run(search_query, time_range=time_range)
        except Exception as exc:
            evidence = {
                "query": search_query,
                "time_range": time_range,
                "results": [],
                "error": str(exc),
            }
        span.set_output(evidence)
    return evidence


def _verify_answer(answer: str, evidence: dict, trace: Trace, span_name: str) -> dict:
    with trace.span(span_name, {"answer_length": len(answer or "")}) as span:
        if not answer:
            result = {
                "claims": [],
                "overall_status": "insufficient_evidence",
                "confidence": 0.0,
                "reason": "No answer was generated.",
            }
        else:
            result = verification_agent.run(answer, evidence)
        span.set_output(result)
    return result


def run_agent(query: str) -> dict:
    """Run the multi-agent workflow.

    For time-sensitive queries, the original LLM-only answer and Tavily
    retrieval execute concurrently. The refreshed LLM answer is then generated
    from Tavily evidence. Both answers are verified independently, and the
    coordinator selects a final answer without conflating the candidates.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    trace = Trace()
    evidence = {"query": "", "time_range": None, "results": []}
    original_answer = None
    refreshed_answer = None
    final_answer = None
    original_verification = None
    refreshed_verification = None
    verification = None
    date_check = None
    was_refreshed = False
    refresh_attempted = False
    final_answer_source = None

    with trace.span("freshness_agent", {"query": query}) as span:
        freshness = freshness_agent.run(query)
        span.set_output(freshness)

    original_freshness = dict(freshness)

    # Stable/general questions do not need a second answer or web retrieval.
    if not freshness.get("time_sensitive", False):
        original_answer = _run_static_query(query, trace)
        final_answer = original_answer
        final_answer_source = "original"
        final_freshness = {
            **freshness,
            "evidence_freshness": None,
            "verification_status": "not_required",
            "confidence": None,
        }
        return {
            "trace_id": trace.trace_id,
            "answer": final_answer,
            "original_answer": original_answer,
            "original_answer_error": None,
            "refreshed_answer": None,
            "original_freshness": original_freshness,
            "freshness": final_freshness,
            "original_verification": None,
            "refreshed_verification": None,
            "verification": None,
            "was_refreshed": False,
            "refresh_attempted": False,
            "final_answer_source": final_answer_source,
            "evidence": evidence,
            "trace": trace.to_dict(),
        }

    search_query = freshness.get("search_query") or query
    time_range = _search_time_range()

    # Parallel branches: (1) LLM-only baseline, (2) Tavily retrieval.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="freshness-branch") as pool:
        baseline_future = pool.submit(_generate_baseline, query, trace)
        evidence_future = pool.submit(
            _retrieve_evidence, search_query, time_range, trace
        )

        baseline_error = None
        try:
            original_answer = baseline_future.result()
        except Exception as exc:
            baseline_error = str(exc)
            original_answer = ""

        # _retrieve_evidence converts Tavily errors into an evidence payload.
        evidence = evidence_future.result()

    # Check evidence freshness independently of either answer's wording.
    with trace.span("evidence_freshness_gate", {"query": query}) as span:
        date_check = verification_agent.evidence_freshness(evidence)
        span.set_output(date_check)

    # Generate refreshed candidate from Tavily evidence on every dynamic query.
    refresh_attempted = True
    with trace.span(
        "refresh_agent",
        {
            "query": query,
            "mode": "tavily_grounded_answer",
            "evidence_count": len(evidence.get("results", [])),
        },
    ) as span:
        try:
            refreshed_answer = refresh_agent.run(
                query,
                original_answer,
                evidence,
                verification=None,
            )
            refreshed_answer = (refreshed_answer or "").strip()
            span.set_output({"answer": refreshed_answer})
        except Exception as exc:
            refreshed_answer = ""
            span.set_output({"error": str(exc), "answer": None})

    # Verify both candidates against the same retrieved evidence.
    original_verification = _verify_answer(
        original_answer, evidence, trace, "original_answer_verification"
    )
    refreshed_verification = _verify_answer(
        refreshed_answer, evidence, trace, "refreshed_answer_verification"
    )

    evidence_is_fresh = date_check.get("status") == "fresh"
    original_is_supported = (
        original_verification.get("overall_status") == "supported"
    )
    refreshed_is_supported = (
        refreshed_verification.get("overall_status") == "supported"
    )

    # Prefer the refreshed answer when fresh evidence supports it.
    if evidence_is_fresh and refreshed_is_supported:
        final_answer = refreshed_answer
        verification = refreshed_verification
        was_refreshed = True
        final_answer_source = "refreshed"

    # If the refreshed candidate fails but the baseline independently matches
    # fresh evidence, it is safe to retain the baseline.
    elif evidence_is_fresh and original_is_supported:
        final_answer = original_answer
        verification = original_verification
        final_answer_source = "original_fallback"

    else:
        reasons = []
        if not evidence_is_fresh:
            reasons.append(
                date_check.get("reason")
                or "Evidence freshness is unverified."
            )
        if not refreshed_is_supported and not original_is_supported:
            reasons.append(
                "Neither the original nor refreshed answer passed claim verification."
            )
        elif not refreshed_is_supported:
            reasons.append("The refreshed answer did not pass claim verification.")
        final_answer = _safe_unverified_answer(" ".join(reasons))
        verification = refreshed_verification
        final_answer_source = "unverified"

    final_freshness = {
        **freshness,
        "evidence_freshness": date_check,
        "verification_status": (
            verification.get("overall_status") if verification else "not_checked"
        ),
        "confidence": verification.get("confidence") if verification else None,
    }

    return {
        "trace_id": trace.trace_id,
        "answer": final_answer,
        "original_answer": original_answer,
        "original_answer_error": baseline_error,
        "refreshed_answer": refreshed_answer,
        "original_freshness": original_freshness,
        "freshness": final_freshness,
        "original_verification": original_verification,
        "refreshed_verification": refreshed_verification,
        "verification": verification,
        "was_refreshed": was_refreshed,
        "refresh_attempted": refresh_attempted,
        "final_answer_source": final_answer_source,
        "evidence": evidence,
        "trace": trace.to_dict(),
    }
