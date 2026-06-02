"""
Layer 7 — Retrieval-Augmented Generation (RAG) Layer

Combines the model's understanding (from the cross-modal transformer and
multi-vector matching) with retrieved evidence from the knowledge base to
produce a grounded, structured diagnosis.

Output format:
    O = {diagnosis, explanation, recommendation, confidence}

This layer reduces hallucination by conditioning generation on real
agricultural evidence retrieved from the FAISS index.
"""

import numpy as np


class RAGGenerator:
    """
    Template-based retrieval-augmented generator.

    Takes:
      - Retrieved evidence documents (from Layer 6)
      - Similarity scores
      - Original query context

    Produces a structured diagnosis grounded in the retrieved knowledge.
    """

    def __init__(self):
        pass

    def generate(
        self,
        query_text: str,
        retrieved_evidence: list[dict],
        fusion_score: float = 0.0,
    ) -> dict:
        """
        Generate a structured diagnosis from retrieved evidence.

        Parameters
        ----------
        query_text          : original farmer query text
        retrieved_evidence  : list of dicts from FAISS retrieval, each with
                              'disease', 'crop', 'symptoms', 'cause',
                              'treatment', 'prevention', 'severity', 'score'
        fusion_score        : overall fusion confidence from cross-modal layer

        Returns
        -------
        dict with keys: diagnosis, explanation, recommendation,
                        confidence, evidence_sources, severity
        """
        if not retrieved_evidence:
            return {
                "diagnosis": "Insufficient data for diagnosis",
                "explanation": "No matching conditions found in the knowledge base. Please provide a clearer image or more detailed description.",
                "recommendation": "Consult a local agricultural extension officer for in-person assessment.",
                "confidence": 0.0,
                "severity": "unknown",
                "evidence_sources": [],
            }

        # ── Primary diagnosis: highest-scoring match ──────────
        primary = retrieved_evidence[0]
        secondary = retrieved_evidence[1:3] if len(retrieved_evidence) > 1 else []

        # ── Confidence computation ────────────────────────────
        primary_score = primary.get("score", 0.0)
        confidence = self._compute_confidence(
            primary_score, fusion_score, retrieved_evidence
        )

        # ── Build structured output ───────────────────────────
        diagnosis = self._build_diagnosis(primary, secondary, confidence)
        explanation = self._build_explanation(primary, query_text)
        recommendation = self._build_recommendation(primary, confidence)
        severity = primary.get("severity", "unknown")

        evidence_sources = [
            {
                "disease": ev.get("disease", "Unknown"),
                "crop": ev.get("crop", "Unknown"),
                "score": round(ev.get("score", 0.0), 4),
            }
            for ev in retrieved_evidence
        ]

        return {
            "diagnosis": diagnosis,
            "explanation": explanation,
            "recommendation": recommendation,
            "confidence": round(confidence, 4),
            "severity": severity,
            "evidence_sources": evidence_sources,
        }

    # ── Private helpers ────────────────────────────────────────

    def _compute_confidence(
        self,
        primary_score: float,
        fusion_score: float,
        evidence: list[dict],
    ) -> float:
        """
        Compute confidence from multiple signals:
          - Primary retrieval similarity (cosine / inner product)
          - Gap between top-1 and top-2 scores (distinctiveness)
          - Fusion score contribution
        """
        # Base confidence from retrieval similarity (0-1 range)
        base = max(0.0, min(1.0, (primary_score + 1.0) / 2.0))  # map [-1,1] → [0,1]

        # Score gap bonus: bigger gap = more confident
        if len(evidence) >= 2:
            gap = primary_score - evidence[1].get("score", 0.0)
            gap_bonus = min(0.15, gap * 0.3)
        else:
            gap_bonus = 0.05

        # Fusion contribution
        fusion_bonus = max(0.0, fusion_score * 0.1)

        confidence = min(1.0, base + gap_bonus + fusion_bonus)
        return confidence

    def _build_diagnosis(
        self,
        primary: dict,
        secondary: list[dict],
        confidence: float,
    ) -> str:
        """Build the diagnosis string."""
        disease = primary.get("disease", "Unknown Condition")
        crop = primary.get("crop", "Unknown Crop")

        if confidence >= 0.8:
            prefix = f"High-confidence diagnosis: **{disease}** in {crop}"
        elif confidence >= 0.5:
            prefix = f"Likely diagnosis: **{disease}** in {crop}"
        else:
            prefix = f"Possible diagnosis: **{disease}** in {crop} (low confidence)"

        # Mention differentials
        if secondary:
            alts = ", ".join(s.get("disease", "?") for s in secondary)
            prefix += f"\n\nDifferential diagnoses to consider: {alts}"

        return prefix

    def _build_explanation(self, primary: dict, query: str) -> str:
        """Build the explanation grounded in retrieved evidence."""
        symptoms = primary.get("symptoms", "No symptom data available.")
        cause = primary.get("cause", "Cause not documented.")

        explanation = (
            f"**Matching Symptoms:**\n{symptoms}\n\n"
            f"**Cause:**\n{cause}\n\n"
            f"This diagnosis was identified by matching your input "
            f"(query: \"{query[:100]}{'…' if len(query) > 100 else ''}\") "
            f"against a curated agricultural disease knowledge base using "
            f"multimodal embedding fusion and cross-modal transformer analysis."
        )
        return explanation

    def _build_recommendation(self, primary: dict, confidence: float) -> str:
        """Build actionable treatment + prevention recommendation."""
        treatment = primary.get("treatment", "No treatment data available.")
        prevention = primary.get("prevention", "No prevention data available.")
        severity = primary.get("severity", "unknown")

        severity_note = {
            "critical": "⚠️ **CRITICAL SEVERITY** — Immediate action required!",
            "high": "🔴 **HIGH SEVERITY** — Take action promptly.",
            "medium": "🟡 **MEDIUM SEVERITY** — Monitor closely and treat.",
            "low": "🟢 **LOW SEVERITY** — Manageable with standard practices.",
        }.get(severity, "Severity not assessed.")

        recommendation = (
            f"{severity_note}\n\n"
            f"**Immediate Treatment:**\n{treatment}\n\n"
            f"**Prevention for Future:**\n{prevention}"
        )

        if confidence < 0.5:
            recommendation += (
                "\n\n⚠️ *Note: Confidence is below 50%. "
                "We strongly recommend consulting a local agricultural expert "
                "or extension officer for verification.*"
            )

        return recommendation
