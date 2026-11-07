"""Application-level, evidence-grounded hallucination evaluation.

The evaluator is an LLM-as-judge call routed through OSS Bifrost. It is not a
Bifrost Enterprise guardrail and does not require x-bf-guardrail-id.
"""

from bifrost_client import chat_text
from agents.json_utils import parse_json
from agents.verification_agent import format_evidence


_ALLOWED_CLAIM_STATUSES = {
    "supported", "contradicted", "unsupported", "unverifiable"
}
_ALLOWED_OVERALL = {
    "grounded", "likely_hallucination", "unsupported_claims", "unverifiable"
}


def run(query: str, answer: str, evidence: dict | None) -> dict:
    """Evaluate answer claims against supplied context, conservatively.

    With no independent evidence, factual hallucination cannot be established;
    the result is explicitly `unverifiable`, not a false pass.
    """
    answer = (answer or "").strip()
    results = (evidence or {}).get("results", [])
    if not answer:
        return {
            "status": "unverifiable",
            "hallucination_detected": None,
            "confidence": 0.0,
            "claims": [],
            "reason": "No answer was generated.",
            "evidence_count": len(results),
        }
    if not results:
        return {
            "status": "unverifiable",
            "hallucination_detected": None,
            "confidence": 0.0,
            "claims": [],
            "reason": (
                "No independent evidence was supplied. An LLM self-check "
                "cannot establish whether factual claims are true."
            ),
            "evidence_count": 0,
        }

    prompt = f"""You are a strict factual-grounding and hallucination evaluator.
Evaluate the answer ONLY against the supplied evidence. Do not use your
pretrained knowledge to fill gaps.

Question:
{query}

Answer to evaluate:
{answer}

Evidence:
{format_evidence(evidence)}

Extract material factual claims and evaluate each one.
Claim statuses:
- supported: directly supported by evidence
- contradicted: evidence directly conflicts with the claim
- unsupported: evidence does not substantiate the claim, but does not directly conflict
- unverifiable: evidence is too ambiguous or irrelevant to decide

Also flag fabricated citations, invented quotations, unsupported precise numbers,
dates, versions, named entities, and claims presented as current without dated support.
Do not call a claim hallucinated merely because evidence omits it. Use unsupported.
A hallucination is likely only when a material claim is contradicted or a citation
is fabricated/misrepresented. If evidence is weak, prefer unverifiable.

Return JSON only in this schema:
{{
  "status": "grounded|likely_hallucination|unsupported_claims|unverifiable",
  "hallucination_detected": true,
  "confidence": 0.0,
  "reason": "short explanation",
  "claims": [
    {{
      "claim": "atomic factual claim",
      "status": "supported|contradicted|unsupported|unverifiable",
      "evidence_urls": ["https://..."],
      "reason": "short evidence-based explanation"
    }}
  ],
  "fabricated_citations": []
}}
"""
    raw = chat_text(
        [
            {
                "role": "system",
                "content": (
                    "You are a skeptical factual-grounding evaluator. "
                    "Return valid JSON only. Never infer support from plausibility."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    data = parse_json(raw, {})

    claims = data.get("claims", [])
    normalized_claims = []
    for item in claims if isinstance(claims, list) else []:
        if not isinstance(item, dict):
            continue
        status = item.get("status", "unverifiable")
        if status not in _ALLOWED_CLAIM_STATUSES:
            status = "unverifiable"
        normalized_claims.append({
            "claim": str(item.get("claim", "")),
            "status": status,
            "evidence_urls": item.get("evidence_urls", []),
            "reason": str(item.get("reason", "")),
        })

    overall = data.get("status", "unverifiable")
    if overall not in _ALLOWED_OVERALL:
        overall = "unverifiable"
    # Derive a conservative status from the individual claims as a backstop.
    claim_statuses = {item["status"] for item in normalized_claims}
    if "contradicted" in claim_statuses:
        overall = "likely_hallucination"
    elif "unsupported" in claim_statuses and overall == "grounded":
        overall = "unsupported_claims"
    elif not normalized_claims and not data.get("fabricated_citations"):
        overall = "unverifiable"
    elif data.get("hallucination_detected") is True and overall == "grounded":
        overall = "likely_hallucination"

    detected = overall == "likely_hallucination"
    return {
        "status": overall,
        "hallucination_detected": detected,
        "confidence": data.get("confidence", 0.0),
        "reason": str(data.get("reason", "")),
        "claims": normalized_claims,
        "fabricated_citations": data.get("fabricated_citations", []),
        "evidence_count": len(results),
        "evaluator": "llm_judge_via_oss_bifrost",
    }
