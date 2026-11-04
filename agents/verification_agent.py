from datetime import datetime, timezone
import os
from bifrost_client import chat_text
from agents.json_utils import parse_json

def _parse_date(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None

def get_evidence_date(item):
    """Extract a date from common metadata fields."""
    date_fields = (
        "published_date",
        "publishedDate",
        "release_date",
        "releaseDate",
        "date",
        "updated_date",
        "updatedDate",
        "last_updated",
        "lastUpdated",
    )

    for field in date_fields:
        value = item.get(field)

        if value:
            parsed = _parse_date(value)
            if parsed is not None:
                return parsed

    return _parse_date(item.get("content"))

def evidence_freshness(evidence, max_age_days=None):
    """Conservative age check. Unknown dates are not considered verified fresh."""
    max_age_days = max_age_days or int(os.getenv("FRESHNESS_MAX_AGE_DAYS", "30"))
    now = datetime.now(timezone.utc)
    dated, stale = 0, 0
    results = evidence.get("results", [])
    if not results:
        return {"status": "insufficient_evidence", "dated_sources": 0, "stale_sources": 0,
                "max_age_days": max_age_days, "reason": "No web results were retrieved."}
    for item in results:
        dt = _parse_date(item.get("date"))
        if dt is None:
            continue
        dated += 1
        if (now - dt).days > max_age_days:
            stale += 1
    if dated == 0:
        return {"status": "unknown_date", "dated_sources": 0, "stale_sources": 0,
                "max_age_days": max_age_days, "reason": "Retrieved sources have no parseable publication dates."}
    if stale == dated:
        return {"status": "stale", "dated_sources": dated, "stale_sources": stale,
                "max_age_days": max_age_days, "reason": "All dated sources exceed the freshness limit."}
    return {"status": "fresh", "dated_sources": dated, "stale_sources": stale,
            "max_age_days": max_age_days, "reason": "At least one dated source is within the freshness limit."}

def format_evidence(evidence):
    chunks = []
    for i, item in enumerate(evidence.get("results", []), 1):
        chunks.append(
            f"[{i}] {item.get('title','')}\nURL: {item.get('url','')}\n"
            f"Published: {item.get('published_date') or 'unknown'}\n"
            f"Content: {item.get('content','')}"
        )
    return "\n\n".join(chunks) or "No evidence returned."

def run(answer, evidence):
    prompt = f"""Verify each factual claim in the answer using ONLY supplied evidence.
Treat publication dates as relevant. Old or undated evidence cannot establish that a changing fact is current.
Return JSON: {{"claims":[{{"claim":"...", "status":"supported|contradicted|stale|insufficient_evidence",
"evidence_urls":["..."], "reason":"..."}}],
"overall_status":"supported|contradicted|stale|insufficient_evidence","confidence":0.0}}
No evidence or inconclusive evidence => insufficient_evidence. Do not use outside knowledge.
ANSWER:
{answer}
EVIDENCE:
{format_evidence(evidence)}"""
    raw = chat_text([
        {"role": "system", "content": "You are a skeptical verifier. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ], temperature=0)
    data = parse_json(raw, {})
    allowed = {"supported", "contradicted", "stale", "insufficient_evidence"}
    status = data.get("overall_status", "insufficient_evidence")
    if status not in allowed:
        status = "insufficient_evidence"
    return {"claims": data.get("claims", []), "overall_status": status,
            "confidence": data.get("confidence", 0.0)}
