"""Coordinator for parallel baseline generation, retrieval, freshness and QA."""

from concurrent.futures import ThreadPoolExecutor
import os

from observability.trace import Trace
from agents import (
    research_agent,
    freshness_agent,
    evidence_agent,
    verification_agent,
    refresh_agent,
    hallucination_agent,
)


def _safe_unverified_answer(reason: str) -> str:
    reason = (reason or "Freshness or evidence could not be established.").strip()
    return (
        "I couldn't verify a sufficiently current answer from the retrieved "
        f"web evidence. {reason} Please try again or provide a trusted, recent source."
    )


def _search_time_range() -> str | None:
    value = os.getenv("TAVILY_TIME_RANGE", "month").strip().lower()
    return value if value in {"day", "week", "month", "year"} else None


def _run_static_query(query: str, trace: Trace) -> str:
    with trace.span("research_agent", {"query": query, "mode": "general"}) as span:
        answer = research_agent.run(query)
        span.set_output({"answer": answer})
    return answer


def _generate_baseline(query: str, trace: Trace) -> str:
    with trace.span(
        "original_research_agent",
        {"query": query, "mode": "llm_only_baseline"},
    ) as span:
        answer = research_agent.run_baseline(query)
        span.set_output({"answer": answer})
    return answer


def _retrieve_evidence(search_query: str, time_range: str | None, trace: Trace) -> dict:
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


def _detect_hallucinations(
    query: str, answer: str, evidence: dict, trace: Trace, span_name: str
) -> dict:
    with trace.span(
        span_name,
        {
            "answer_length": len(answer or ""),
            "evidence_count": len((evidence or {}).get("results", [])),
            "evaluator": "llm_judge_via_oss_bifrost",
        },
    ) as span:
        try:
            result = hallucination_agent.run(query, answer, evidence)
        except Exception as exc:
            result = {
                "status": "evaluation_error",
                "hallucination_detected": None,
                "confidence": 0.0,
                "claims": [],
                "reason": str(exc),
                "evaluator": "llm_judge_via_oss_bifrost",
            }
        span.set_output(result)
    return result


def _hallucination_is_clear(result: dict | None) -> bool:
    """Return True when the judge did not find a material falsehood.

    `unsupported_claims` is not the same as `likely_hallucination`. The
    independent claim-verification agent remains responsible for deciding
    whether the answer is sufficiently supported. The hallucination judge
    acts as a veto for contradicted/fabricated claims, not as a duplicate
    all-claims-must-be-supported gate.
    """
    return bool(result) and result.get("status") in {
        "grounded", "unsupported_claims"
    } and result.get("hallucination_detected") is False


def run_agent(query: str) -> dict:
    """Run baseline + retrieval concurrently, then evaluate both candidates.

    OSS Bifrost is used as the LLM gateway. Hallucination evaluation is an
    application-level LLM-as-judge agent, not a licensed Bifrost guardrail.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    trace = Trace()
    evidence = {"query": "", "time_range": None, "results": []}
    original_answer = None
    refreshed_answer = None
    original_answer_error = None
    refreshed_answer_error = None
    original_verification = None
    refreshed_verification = None
    original_hallucination = None
    refreshed_hallucination = None
    verification = None
    date_check = None
    was_refreshed = False
    refresh_attempted = False
    final_answer_source = None

    with trace.span("freshness_agent", {"query": query}) as span:
        freshness = freshness_agent.run(query)
        span.set_output(freshness)
    original_freshness = dict(freshness)

    # Stable/general questions use one LLM response. Without evidence, a
    # factual hallucination verdict is intentionally marked unverifiable.
    if not freshness.get("time_sensitive", False):
        original_answer = _run_static_query(query, trace)
        original_hallucination = _detect_hallucinations(
            query, original_answer, evidence, trace, "original_hallucination_evaluation"
        )
        final_freshness = {
            **freshness,
            "evidence_freshness": None,
            "verification_status": "not_required",
            "confidence": None,
        }
        return {
            "trace_id": trace.trace_id,
            "answer": original_answer,
            "original_answer": original_answer,
            "original_answer_error": None,
            "refreshed_answer": None,
            "refreshed_answer_error": None,
            "original_freshness": original_freshness,
            "freshness": final_freshness,
            "original_verification": None,
            "refreshed_verification": None,
            "verification": None,
            "original_hallucination": original_hallucination,
            "refreshed_hallucination": None,
            "was_refreshed": False,
            "refresh_attempted": False,
            "final_answer_source": "original",
            "evidence": evidence,
            "trace": trace.to_dict(),
        }

    search_query = freshness.get("search_query") or query
    time_range = _search_time_range()

    # The baseline LLM response and Tavily retrieval run independently.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="freshness-branch") as pool:
        baseline_future = pool.submit(_generate_baseline, query, trace)
        evidence_future = pool.submit(_retrieve_evidence, search_query, time_range, trace)

        try:
            original_answer = (baseline_future.result() or "").strip()
        except Exception as exc:
            original_answer_error = str(exc)
            original_answer = ""

        evidence = evidence_future.result()

    with trace.span("evidence_freshness_gate", {"query": query}) as span:
        date_check = verification_agent.evidence_freshness(evidence)
        span.set_output(date_check)

    # Generate an evidence-grounded candidate on every dynamic query.
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
            refreshed_answer = (
                refresh_agent.run(query, original_answer, evidence) or ""
            ).strip()
            span.set_output({"answer": refreshed_answer})
        except Exception as exc:
            refreshed_answer_error = str(exc)
            refreshed_answer = ""
            span.set_output({"error": str(exc), "answer": None})

    # Run claim verification and hallucination evaluation for both candidates.
    original_verification = _verify_answer(
        original_answer, evidence, trace, "original_answer_verification"
    )
    refreshed_verification = _verify_answer(
        refreshed_answer, evidence, trace, "refreshed_answer_verification"
    )
    original_hallucination = _detect_hallucinations(
        query, original_answer, evidence, trace, "original_hallucination_evaluation"
    )
    refreshed_hallucination = _detect_hallucinations(
        query, refreshed_answer, evidence, trace, "refreshed_hallucination_evaluation"
    )

    evidence_is_fresh = date_check.get("status") == "fresh"
    original_is_supported = (
        original_verification.get("overall_status") == "supported"
        and _hallucination_is_clear(original_hallucination)
    )
    refreshed_is_supported = (
        refreshed_verification.get("overall_status") == "supported"
        and _hallucination_is_clear(refreshed_hallucination)
    )

    decision_checks = {
        "evidence_is_fresh": evidence_is_fresh,
        "original_claims_supported": original_verification.get("overall_status") == "supported",
        "refreshed_claims_supported": refreshed_verification.get("overall_status") == "supported",
        "original_hallucination_clear": _hallucination_is_clear(original_hallucination),
        "refreshed_hallucination_clear": _hallucination_is_clear(refreshed_hallucination),
        "original_hallucination_status": original_hallucination.get("status"),
        "refreshed_hallucination_status": refreshed_hallucination.get("status"),
        "original_verification_status": original_verification.get("overall_status"),
        "refreshed_verification_status": refreshed_verification.get("overall_status"),
    }

    if evidence_is_fresh and refreshed_is_supported:
        final_answer = refreshed_answer
        verification = refreshed_verification
        was_refreshed = True
        final_answer_source = "refreshed"
    elif evidence_is_fresh and original_is_supported:
        final_answer = original_answer
        verification = original_verification
        final_answer_source = "original_fallback"
    else:
        reasons = []
        if not evidence_is_fresh:
            reasons.append(
                date_check.get("reason") or "Evidence freshness is unverified."
            )
        if not refreshed_is_supported and not original_is_supported:
            reasons.append(
                "Neither answer passed both claim verification and hallucination evaluation."
            )
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
        "original_answer_error": original_answer_error,
        "refreshed_answer": refreshed_answer,
        "refreshed_answer_error": refreshed_answer_error,
        "original_freshness": original_freshness,
        "freshness": final_freshness,
        "original_verification": original_verification,
        "refreshed_verification": refreshed_verification,
        "verification": verification,
        "original_hallucination": original_hallucination,
        "refreshed_hallucination": refreshed_hallucination,
        "was_refreshed": was_refreshed,
        "refresh_attempted": refresh_attempted,
        "final_answer_source": final_answer_source,
        "decision_checks": decision_checks,
        "evidence": evidence,
        "trace": trace.to_dict(),
    }
