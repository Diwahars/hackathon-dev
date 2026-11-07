"""LLM answer generation for baseline and stable/general questions."""

from bifrost_client import chat_text


def _baseline_messages(query: str):
    return [
        {
            "role": "system",
            "content": """You are a knowledgeable research assistant producing an ORIGINAL BASELINE answer.

Answer using only your pretrained/model knowledge. No web search or external
retrieval has been performed for this answer.

For facts that may change over time:
- Give your best answer based on knowledge available to you.
- Explicitly say that the answer is not live-verified and may be outdated.
- Do not invent an exact 'as of' date, release date, price, or current status.
- If you do not know, say so rather than guessing.

Do not include fabricated URLs or citations. Do not expose private chain-of-thought.""",
        },
        {"role": "user", "content": query},
    ]


def run_baseline(query: str) -> str:
    """Generate an LLM-only baseline answer through OSS Bifrost."""
    query = (query or "").strip()
    return chat_text(_baseline_messages(query), temperature=0.2)


def run(query: str, evidence: dict | None = None) -> str:
    """Answer stable/general questions, or evidence-grounded questions."""
    query = (query or "").strip()
    results = (evidence or {}).get("results", [])

    if evidence is not None and not results:
        return (
            "I couldn't retrieve usable web evidence for this time-sensitive "
            "question, so I can't reliably confirm the current answer."
        )

    if evidence is None:
        system_prompt = (
            "You are a careful research assistant. Answer stable, general "
            "questions clearly and accurately. Do not invent citations or "
            "claim to have checked live information. Do not expose private "
            "chain-of-thought."
        )
        user_prompt = query
    else:
        from agents.verification_agent import format_evidence
        system_prompt = """You are an evidence-grounded research agent.
Use ONLY the supplied web evidence for current or changing facts.
Do not invent prices, versions, dates, statistics, statuses or citations.
Prefer primary/official sources and directly relevant evidence.
If the evidence does not establish the requested fact, say so plainly.
Cite source URLs next to the claims they support.
Do not expose private chain-of-thought."""
        user_prompt = (
            f"Question:\n{query}\n\n"
            f"Retrieved web evidence:\n{format_evidence(evidence)}\n\n"
            "Provide a concise answer grounded in these sources."
        )

    return chat_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1 if evidence else 0.2,
    )
