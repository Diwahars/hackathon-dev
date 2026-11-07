import sys
import types
import unittest
from unittest.mock import patch

try:
    from openai import OpenAI  # noqa: F401
except ImportError:
    fake_openai = types.ModuleType("openai")

    class _UnusedCompletions:
        def create(self, **kwargs):
            raise RuntimeError("Network call not expected in offline tests")

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_UnusedCompletions())

    fake_openai.OpenAI = _FakeOpenAI
    sys.modules["openai"] = fake_openai

import agent
from agents import hallucination_agent


class ParallelWorkflowTests(unittest.TestCase):
    def test_dynamic_query_runs_hallucination_evaluation_for_both_answers(self):
        evidence = {
            "query": "latest sample version",
            "time_range": "month",
            "results": [{
                "title": "Official release",
                "url": "https://example.com/release",
                "published_date": "2026-11-05T00:00:00Z",
                "content": "Version 9.2 was released.",
            }],
        }
        supported = {"claims": [], "overall_status": "supported", "confidence": 0.99}
        grounded = {
            "status": "grounded",
            "hallucination_detected": False,
            "confidence": 0.98,
            "claims": [],
            "reason": "Claims supported by supplied evidence.",
        }

        with patch.object(agent.freshness_agent, "run", return_value={
            "time_sensitive": True,
            "search_query": "latest sample version",
            "reason": "Latest version changes.",
        }), patch.object(
            agent.research_agent, "run_baseline", return_value="Baseline answer."
        ), patch.object(
            agent.evidence_agent, "run", return_value=evidence
        ), patch.object(
            agent.verification_agent, "evidence_freshness",
            return_value={"status": "fresh", "reason": "Recent source."},
        ), patch.object(
            agent.refresh_agent, "run", return_value="Refreshed answer version 9.2."
        ), patch.object(
            agent.verification_agent, "run", return_value=supported
        ), patch.object(
            agent.hallucination_agent, "run", return_value=grounded
        ) as hall:
            result = agent.run_agent("What is the latest sample version?")

        self.assertEqual(result["original_answer"], "Baseline answer.")
        self.assertEqual(result["refreshed_answer"], "Refreshed answer version 9.2.")
        self.assertEqual(result["answer"], result["refreshed_answer"])
        self.assertEqual(result["final_answer_source"], "refreshed")
        self.assertEqual(result["original_hallucination"]["status"], "grounded")
        self.assertEqual(result["refreshed_hallucination"]["status"], "grounded")
        self.assertEqual(hall.call_count, 2)

    def test_unsupported_claim_label_does_not_override_supported_verification(self):
        evidence = {
            "results": [{
                "title": "Official source",
                "url": "https://example.com",
                "published_date": "2026-11-05T00:00:00Z",
                "content": "The release is version 9.2.",
            }]
        }
        supported = {"claims": [], "overall_status": "supported", "confidence": 0.9}
        no_detected_hallucination = {
            "status": "unsupported_claims", "hallucination_detected": False,
            "confidence": 0.8, "claims": [{"status": "unsupported"}],
        }
        with patch.object(agent.freshness_agent, "run", return_value={"time_sensitive": True}), \
             patch.object(agent.research_agent, "run_baseline", return_value="Baseline."), \
             patch.object(agent.evidence_agent, "run", return_value=evidence), \
             patch.object(agent.verification_agent, "evidence_freshness", return_value={"status": "fresh"}), \
             patch.object(agent.refresh_agent, "run", return_value="Refreshed."), \
             patch.object(agent.verification_agent, "run", return_value=supported), \
             patch.object(agent.hallucination_agent, "run", side_effect=[no_detected_hallucination, no_detected_hallucination]):
            result = agent.run_agent("Latest release?")
        self.assertEqual(result["final_answer_source"], "refreshed")
        self.assertEqual(result["answer"], "Refreshed.")
        self.assertTrue(result["decision_checks"]["refreshed_hallucination_clear"])

    def test_hallucination_finding_prevents_candidate_acceptance(self):
        evidence = {
            "results": [{
                "title": "Official source",
                "url": "https://example.com",
                "published_date": "2026-11-05T00:00:00Z",
                "content": "The release is version 9.2.",
            }]
        }
        supported = {"claims": [], "overall_status": "supported", "confidence": 0.9}
        grounded = {
            "status": "grounded", "hallucination_detected": False,
            "confidence": 0.9, "claims": [],
        }
        hallucinated = {
            "status": "likely_hallucination", "hallucination_detected": True,
            "confidence": 0.99, "claims": [{"status": "contradicted"}],
        }
        with patch.object(agent.freshness_agent, "run", return_value={"time_sensitive": True}), \
             patch.object(agent.research_agent, "run_baseline", return_value="Baseline."), \
             patch.object(agent.evidence_agent, "run", return_value=evidence), \
             patch.object(agent.verification_agent, "evidence_freshness", return_value={"status": "fresh"}), \
             patch.object(agent.refresh_agent, "run", return_value="Refreshed."), \
             patch.object(agent.verification_agent, "run", return_value=supported), \
             patch.object(agent.hallucination_agent, "run", side_effect=[grounded, hallucinated]):
            result = agent.run_agent("Latest release?")
        self.assertEqual(result["final_answer_source"], "original_fallback")
        self.assertEqual(result["answer"], "Baseline.")

    def test_static_query_marks_hallucination_unverifiable_without_context(self):
        with patch.object(agent.freshness_agent, "run", return_value={
            "time_sensitive": False,
            "reason": "Stable concept.",
        }), patch.object(
            agent.research_agent, "run", return_value="Stable answer."
        ), patch.object(
            agent.hallucination_agent, "run", return_value={
                "status": "unverifiable",
                "hallucination_detected": None,
                "confidence": 0.0,
                "claims": [],
                "reason": "No independent evidence.",
            }
        ):
            result = agent.run_agent("Explain a stable concept.")
        self.assertEqual(result["answer"], "Stable answer.")
        self.assertEqual(result["original_hallucination"]["status"], "unverifiable")
        self.assertIsNone(result["refreshed_answer"])


class HallucinationAgentTests(unittest.TestCase):
    def test_no_evidence_is_unverifiable_not_pass(self):
        result = hallucination_agent.run("Question?", "Answer.", {"results": []})
        self.assertEqual(result["status"], "unverifiable")
        self.assertIsNone(result["hallucination_detected"])

    def test_judge_output_is_normalized(self):
        evidence = {"results": [{
            "title": "Source", "url": "https://example.com",
            "published_date": "2026-11-05", "content": "Fact is true.",
        }]}
        with patch.object(
            hallucination_agent, "chat_text", return_value=(
                '{"status":"grounded","hallucination_detected":false,'
                '"confidence":0.9,"claims":[{"claim":"Fact is true",'
                '"status":"supported","evidence_urls":["https://example.com"]}]}'
            )
        ):
            result = hallucination_agent.run("Question?", "Fact is true.", evidence)
        self.assertEqual(result["status"], "grounded")
        self.assertFalse(result["hallucination_detected"])
        self.assertEqual(result["claims"][0]["status"], "supported")


if __name__ == "__main__":
    unittest.main()
