from bifrost_client import chat_text
from agents.verification_agent import format_evidence

def run(query, original_answer, evidence, verification):
    prompt = f"""Answer the user's question using ONLY the supplied web evidence.
This is a refresh, not a paraphrase of the old answer.
- Do not repeat claims from the original answer unless evidence supports them.
- Prioritize recent, directly relevant evidence and include source URLs for key claims.
- If dates are missing, do not assert that a changing fact is current.
- If sources conflict or are insufficient, clearly state the limitation instead of guessing.
- Do not invent facts, dates, versions, statistics, or citations.
QUERY: {query}
ORIGINAL ANSWER (unverified): {original_answer}
VERIFICATION: {verification}
EVIDENCE:
{format_evidence(evidence)}"""
    return chat_text([
        {"role": "system", "content": "You produce evidence-grounded refreshed answers. Treat the original answer as untrusted."},
        {"role": "user", "content": prompt},
    ], temperature=0)
