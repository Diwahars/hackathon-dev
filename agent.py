import os
from observability.trace import Trace
from agents import research_agent, freshness_agent, evidence_agent
from agents import verification_agent, refresh_agent

def _safe_unverified_answer(reason):
    return ("I couldn't verify a sufficiently current answer from the web evidence "
            f"retrieved for this question. {reason} Please try again or provide a trusted, recent source.")

def run_agent(query):
    trace = Trace()
    was_refreshed = False
    evidence = {"query": "", "results": []}
    verification = None
    freshness = {}

    with trace.span("freshness_agent", {"query": query}) as span:
        freshness = freshness_agent.run(query)
        span.set_output(freshness)
        print(f"Freshness: {freshness}")
    # Search first for time-sensitive questions; the initial model response is
    # only a baseline and is never treated as current evidence.
    with trace.span("research_agent", {"query": query}) as span:
        original_answer = research_agent.run(query)
        span.set_output({"answer": original_answer})
    print(f"Original answer: {original_answer}")
    original_freshness = freshness.copy()
    final_answer = original_answer

    if freshness.get("time_sensitive"):
        search_query = freshness.get("search_query") or query
        # Ask Tavily for recent results. Its supported time_range values are
        # coarse; the strict age policy is checked separately below.
        time_range = os.getenv("TAVILY_TIME_RANGE", "month")
        with trace.span("evidence_search_agent", {"query": search_query, "time_range": time_range}) as span:
            evidence = evidence_agent.run(search_query, time_range=time_range)
            span.set_output(evidence)
        print(f"Evidence: {evidence}")
        date_check = verification_agent.evidence_freshness(evidence)
        with trace.span("evidence_freshness_gate", {"max_age_days": date_check["max_age_days"]}) as span:
            span.set_output(date_check)
        print(f"Date check: {date_check}")
        with trace.span("claim_verification_agent", {"answer": original_answer}) as span:
            verification = verification_agent.run(original_answer, evidence)
            span.set_output(verification)
        print(f"Verification: {verification}")

        # Force refresh for non-supported answers OR evidence that is stale /
        # has unknown dates. Never silently pass stale evidence as current.
        must_refresh = (verification.get("overall_status") != "supported"
                        or date_check.get("status") != "fresh")
        print(f"Must refresh: {must_refresh}")
        if must_refresh:
            was_refreshed = True
            with trace.span("refresh_agent", {"query": query}) as span:
                refreshed_answer = refresh_agent.run(query, original_answer, evidence, verification)
                span.set_output({"answer": refreshed_answer})
            print(f"Refreshed answer: {refreshed_answer}")
            with trace.span("re_verification_agent", {"answer": refreshed_answer}) as span:
                refreshed_verification = verification_agent.run(refreshed_answer, evidence)
                span.set_output(refreshed_verification)
            print(f"Refreshed verification: {refreshed_verification}")

            # The refreshed answer is only accepted if evidence passes the age
            # gate and the verifier supports it. Otherwise return an honest limit.
            if (date_check.get("status") == "fresh"
                    and refreshed_verification.get("overall_status") == "supported"):
                final_answer = refreshed_answer
                verification = refreshed_verification
            else:
                reason = date_check.get("reason", "Freshness could not be established.")
                if refreshed_verification.get("overall_status") != "supported":
                    reason += " The refreshed claims also did not pass evidence verification."
                final_answer = _safe_unverified_answer(reason)
                verification = refreshed_verification
        elif date_check.get("status") == "fresh":
            final_answer = original_answer
        else:
            final_answer = _safe_unverified_answer(date_check.get("reason", "Freshness is unknown."))

    final_freshness = {
        **freshness,
        "evidence_freshness": date_check if freshness.get("time_sensitive") else None,
        "verification_status": verification.get("overall_status") if verification else "not_checked",
        "confidence": verification.get("confidence") if verification else None,
    }
    return {
        "trace_id": trace.trace_id, "answer": final_answer,
        "original_answer": original_answer, "original_freshness": original_freshness,
        "was_refreshed": was_refreshed, "freshness": final_freshness,
        "verification": verification, "evidence": evidence, "trace": trace.to_dict(),
    }
