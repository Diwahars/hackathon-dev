"""Generate a refreshed answer using Tavily-retrieved evidence."""

from bifrost_client import chat_text
from agents.verification_agent import format_evidence


def run(query, original_answer, evidence, verification=None):
    """Generate a new answer grounded in Tavily evidence only.

    The original answer and its verification are supplied as comparison
    context, not as factual sources. The retrieved evidence is authoritative
    for changing/current claims.
    """
    prompt = f"""Produce a refreshed answer to the user's question using ONLY the supplied web evidence.
This is a new evidence-grounded answer, not a paraphrase of the original.

Rules:
- Treat the original answer as untrusted baseline context, not as a source.
- Do not repeat claims from the original answer unless the web evidence supports them.
- Prioritize recent, directly relevant evidence and official/primary sources.
- Include source URLs for key claims.
- If dates are missing, do not assert that a changing fact is current.
- If sources conflict or are insufficient, clearly state the limitation instead of guessing.
- Do not invent facts, dates, versions, statistics, or citations.

QUERY:
{query}

ORIGINAL ANSWER (baseline only; not evidence):
{original_answer or 'No baseline answer was generated.'}

ORIGINAL ANSWER VERIFICATION:
{verification or 'Not available.'}

TAVILY EVIDENCE:
{format_evidence(evidence)}"""

    return chat_text(
        [
            {
                "role": "system",
                "content": (
                    "You produce evidence-grounded refreshed answers. "
                    "Treat the original answer as untrusted and use only the "
                    "provided web evidence for current facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
