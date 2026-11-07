"""Coordinator for parallel baseline generation, retrieval, freshness and QA."""

from concurrent.futures import ThreadPoolExecutor
import os
import re

from observability.trace import Trace
from agents import (
    research_agent,
    freshness_agent,
    evidence_agent,
    verification_agent,
    refresh_agent,
    hallucination_agent,
)

_STATIC_VERIFICATION_PATTERNS = (
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\b(?:according to|cited in|source:|study by)\b", re.IGNORECASE),
    re.compile(r"\b(?:19\d{2}|20\d{2})\b"),
    re.compile(r"\b\d+(?:\.\d+)?%\b|\$\d+"),
)


def _needs_static_verification(answer: str | None) -> bool:
    """Return True if a static answer contains URLs, citations, or specific figures."""
    if not answer:
        return False
    return any(pattern.search(answer) for pattern in _STATIC_VERIFICATION_PATTERNS)


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
    """Return True when the judge did not find a material falsehood."""
    return bool(result) and result.get("status") in {
        "grounded", "unsupported_claims"
    } and result.get("hallucination_detected") is False


def run_agent(query: str) -> dict:
    """Run baseline + retrieval concurrently, then evaluate both candidates."""
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

    # Stable/general questions use one LLM response by default, unless the
    # answer includes verifiable specifics (URLs, citations, dates, figures).
    if not freshness.get("time_sensitive", False):
        original_answer = _run_static_query(query, trace)

        if _needs_static_verification(original_answer):
            evidence = _retrieve_evidence(query, None, trace)
            with ThreadPoolExecutor(max_workers=2) as executor:
                ver_fut = executor.submit(
                    _verify_answer,
                    original_answer,
                    evidence,
                    trace,
                    "original_verification_agent",
                )
                hal_fut = executor.submit(
                    _detect_hallucinations,
                    query,
                    original_answer,
                    evidence,
                    trace,
                    "original_hallucination_evaluation",
                )
                original_verification = ver_fut.result()
                original_hallucination = hal_fut.result()

            verification = original_verification
            final_answer = original_answer

            if not _hallucination_is_clear(original_hallucination):
                refresh_attempted = True
                claims = original_verification.get("claims", [])
                with trace.span(
                    "refresh_agent",
                    {
                        "query": query,
                        "claims_count": len(claims),
                        "original_status": original_verification.get("overall_status"),
                    },
                ) as span:
                    try:
                        refreshed_answer = refresh_agent.run(
                            query=query, claims=claims, evidence=evidence
                        )
                        span.set_output({"answer": refreshed_answer})
                    except Exception as exc:
                        refreshed_answer_error = str(exc)
                        span.set_output({"error": refreshed_answer_error})

                if refreshed_answer:
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        r_ver_fut = executor.submit(
                            _verify_answer,
                            refreshed_answer,
                            evidence,
                            trace,
                            "refreshed_verification_agent",
                        )
                        r_hal_fut = executor.submit(
                            _detect_hallucinations,
                            query,
                            refreshed_answer,
                            evidence,
                            trace,
                            "refreshed_hallucination_evaluation",
                        )
                        refreshed_verification = r_ver_fut.result()
                        refreshed_hallucination = r_hal_fut.result()

                    if (
                        refreshed_verification.get("overall_status")
                        in {"verified", "partially_verified"}
                        and _hallucination_is_clear(refreshed_hallucination)
                    ):
                        final_answer = refreshed_answer
                        verification = refreshed_verification
                        was_refreshed = True
                        final_answer_source = "refreshed"

                if not was_refreshed:
                    final_answer = _safe_unverified_answer(
                        "Independent LLM hallucination check flagged contradicted or fabricated specifics in the static response."
                    )
                    final_answer_source = "fallback"
            else:
                final_answer_source = "original"

            final_freshness = {
                **freshness,
                "evidence_freshness": evidence.get("freshness"),
                "verification_status": verification.get("overall_status"),
                "confidence": verification.get("confidence"),
            }
            return {
                "trace_id": trace.trace_id,
                "answer": final_answer,
                "original_answer": original_answer,
                "original_answer_error": None,
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
                "evidence": evidence,
                "trace": trace.to_dict(),
            }

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

    search_query = (
        freshness.get("search_query")
        if isinstance(freshness.get("search_query"), str)
        and freshness.get("search_query").strip()
        else query
    )
    time_range = _search_time_range()

    with ThreadPoolExecutor(max_workers=2) as executor:
        baseline_future = executor.submit(_generate_baseline, query, trace)
        evidence_future = executor.submit(
            _retrieve_evidence, search_query, time_range, trace
        )
        original_answer = baseline_future.result()
        evidence = evidence_future.result()

    with ThreadPoolExecutor(max_workers=2) as executor:
        verification_future = executor.submit(
            _verify_answer,
            original_answer,
            evidence,
            trace,
            "original_verification_agent",
        )
        hallucination_future = executor.submit(
            _detect_hallucinations,
            query,
            original_answer,
            evidence,
            trace,
            "original_hallucination_evaluation",
        )
        original_verification = verification_future.result()
        original_hallucination = hallucination_future.result()

    claims = original_verification.get("claims", [])
    overall_status = original_verification.get("overall_status")
    verification = original_verification
    final_answer = original_answer
    final_answer_source = "original"

    if overall_status != "verified" and _hallucination_is_clear(original_hallucination):
        refresh_attempted = True
        with trace.span(
            "refresh_agent",
            {
                "query": query,
                "claims_count": len(claims),
                "original_status": overall_status,
            },
        ) as span:
            try:
                refreshed_answer = refresh_agent.run(
                    query=query, claims=claims, evidence=evidence
                )
                span.set_output({"answer": refreshed_answer})
            except Exception as exc:
                refreshed_answer_error = str(exc)
                span.set_output({"error": refreshed_answer_error})

        if refreshed_answer:
            with ThreadPoolExecutor(max_workers=2) as executor:
                ver_fut = executor.submit(
                    _verify_answer,
                    refreshed_answer,
                    evidence,
                    trace,
                    "refreshed_verification_agent",
                )
                hal_fut = executor.submit(
                    _detect_hallucinations,
                    query,
                    refreshed_answer,
                    evidence,
                    trace,
                    "refreshed_hallucination_evaluation",
                )
                refreshed_verification = ver_fut.result()
                refreshed_hallucination = hal_fut.result()

            refreshed_status = refreshed_verification.get("overall_status")
            if (
                refreshed_status in {"verified", "partially_verified"}
                and _hallucination_is_clear(refreshed_hallucination)
            ):
                final_answer = refreshed_answer
                verification = refreshed_verification
                was_refreshed = True
                final_answer_source = "refreshed"

    if not was_refreshed and overall_status not in {"verified", "partially_verified"}:
        if not _hallucination_is_clear(original_hallucination):
            final_answer = _safe_unverified_answer(
                "Independent LLM hallucination check flagged contradicted or unverified statements."
            )
        else:
            final_answer = _safe_unverified_answer(
                f"Initial claims evaluation returned status '{overall_status}'."
            )
        final_answer_source = "fallback"

    final_freshness = {
        **freshness,
        "evidence_freshness": evidence.get("freshness"),
        "verification_status": verification.get("overall_status"),
        "confidence": verification.get("confidence"),
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
        "evidence": evidence,
        "trace": trace.to_dict(),
    }
