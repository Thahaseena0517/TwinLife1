"""
Module 7 -- narration_agent.py

Narration layer for TwinLife AI.

Uses the Gemini API to generate warm, plain-language explanations that
combine computed scores, SHAP feature contributions, and RAG guideline
chunks. Also generates ranked what-if scenario comparisons.

Usage:

    explanation = generate_explanation(twin_outputs, shap_features, rag_chunks)
    whatif_text = generate_ranked_whatifs(scenarios)
"""

from __future__ import annotations

import os
import json
import time
import warnings

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# GEMINI CLIENT HELPER
# ---------------------------------------------------------------------------

def _get_gemini_client() -> genai.Client:
    """Create a Gemini API client from the environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def _call_gemini(
    client: genai.Client,
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-3.5-flash-lite",
    max_retries: int = 3,
) -> str:
    """Call Gemini API with automatic retry on rate-limit errors.

    Args:
        client: Gemini API client.
        system_prompt: System instruction for the model.
        user_prompt: User message content.
        model: Model name to use.
        max_retries: Maximum number of retry attempts.

    Returns:
        Generated text response.
    """
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_prompt)],
                    ),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.4,
                    max_output_tokens=1500,
                ),
            )
            return response.candidates[0].content.parts[0].text.strip()
        except Exception as e:
            error_str = str(e)
            if (
                "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            ) and attempt < max_retries:
                wait = 30 * (attempt + 1)
                print(
                    f"    [Rate limited — waiting {wait}s before retry "
                    f"{attempt + 1}/{max_retries}...]"
                )
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# GENERATE EXPLANATION
# ---------------------------------------------------------------------------

def generate_explanation(
    twin_outputs: dict,
    shap_features: list[dict],
    rag_chunks: list[str],
) -> str:
    """Generate a plain-language explanation combining scores, SHAP, and RAG.

    Args:
        twin_outputs: Dict with keys like "health", "finance", "insurance",
                      "simulation" — each containing the respective profile.
        shap_features: List of SHAP feature dicts from HealthTwin:
                       [{"feature": str, "impact": float, "model": str}, ...]
        rag_chunks: List of retrieved guideline text chunks from RAGAgent.

    Returns:
        A warm, cited, plain-language explanation string.
    """
    client = _get_gemini_client()

    system_prompt = (
        "You are a compassionate health and financial wellness advisor for the "
        "TwinLife AI digital twin platform. Your task is to write a SHORT, WARM, "
        "PLAIN-LANGUAGE explanation for the user based on their computed scores, "
        "the key factors driving their health prediction (from SHAP analysis), "
        "and relevant guideline excerpts (from medical/financial guidelines).\n\n"
        "RULES:\n"
        "1. Do NOT invent any numeric claims not provided in the input data.\n"
        "2. When referencing guideline information, cite the source naturally "
        "(e.g., 'According to AHA guidelines...' or 'As per financial planning "
        "best practices...').\n"
        "3. Keep the tone warm, supportive, and non-alarming.\n"
        "4. Use simple language — avoid medical jargon where possible.\n"
        "5. End with 1-2 actionable next steps the user can take.\n"
        "6. Include a brief disclaimer that this is illustrative guidance, "
        "not licensed medical or financial advice.\n"
        "7. Keep the response under 300 words.\n"
        "8. When discussing a specific treatment simulation, use the "
        "treatment cost, covered amount, and coverage gap from the simulation "
        "tool results. Do NOT use the overall insurance coverage ratio to "
        "describe the coverage gap for that specific treatment.\n"
    )

    # Build the user prompt with all available data
    sections = []

    if "health" in twin_outputs:
        hp = twin_outputs["health"]
        sections.append(
            "=== HEALTH PROFILE ===\n"
            f"BMI: {hp.get('bmi')} ({hp.get('obesity_class')})\n"
            f"Blood Pressure: {hp.get('hypertension_stage')}\n"
            f"Cardiovascular Risk: "
            f"{hp.get('cardio_risk', {}).get('probability', 'N/A')} "
            f"({hp.get('cardio_risk', {}).get('label', 'N/A')})\n"
            f"Diabetes Risk: "
            f"{hp.get('diabetes_risk', {}).get('probability', 'N/A')} "
            f"({hp.get('diabetes_risk', {}).get('label', 'N/A')})\n"
            f"Overall Health Score: {hp.get('overall_health_score')}/100"
        )

    if "finance" in twin_outputs:
        fp = twin_outputs["finance"]
        sections.append(
            "=== FINANCE PROFILE ===\n"
            f"DTI Ratio: {fp.get('dti_ratio')}\n"
            f"Savings Rate: {fp.get('savings_rate')}\n"
            f"Emergency Fund: {fp.get('emergency_fund_months')} months\n"
            f"Financial Stability Score: "
            f"{fp.get('financial_stability_score')}/100"
        )

    if "insurance" in twin_outputs:
        ip = twin_outputs["insurance"]
        sections.append(
            "=== INSURANCE PROFILE ===\n"
            f"Estimated Future Cost: "
            f"Rs.{ip.get('estimated_future_cost', 0):,.0f}\n"
            f"Coverage Ratio: {ip.get('coverage_adequacy_ratio')}\n"
            f"Premium Ratio: {ip.get('premium_affordability_ratio')}\n"
            f"Rider Gaps: "
            f"{ip.get('rider_analysis', {}).get('gaps', [])}\n"
            f"Insurance Score: {ip.get('insurance_adequacy_score')}/100"
        )

    if "simulation" in twin_outputs and twin_outputs["simulation"]:
        sim = twin_outputs["simulation"]
        simulation_lines = ["=== TREATMENT SIMULATION ==="]

        tool_trace = sim.get("tool_trace", [])

        for tool in tool_trace:
            name = tool.get("tool")
            result = tool.get("result", {})

            if name == "check_affordability":
                simulation_lines.append(
                    f"Treatment Cost: "
                    f"Rs.{result.get('cost', tool.get('input', {}).get('cost', 0)):,.0f}"
                )

                simulation_lines.append(
                    f"Treatment Affordable From Disposable Income: "
                    f"{result.get('affordable')}"
                )

                simulation_lines.append(
                    f"Affordability Score: {result.get('score')}"
                )

            elif name == "check_insurance_coverage":
                simulation_lines.append(
                    f"Specific Treatment Covered Amount: "
                    f"Rs.{result.get('covered_amount', 0):,.0f}"
                )

                simulation_lines.append(
                    f"Specific Treatment Coverage Gap: "
                    f"Rs.{result.get('gap', 0):,.0f}"
                )

            elif name == "get_health_context":
                simulation_lines.append(
                    f"Health Risk Level: {result.get('risk_level')}"
                )

                simulation_lines.append(
                    f"Clinical Urgency: {result.get('urgency')}"
                )

        if "recommendation" in sim:
            simulation_lines.append(
                f"Agent Recommendation:\n{sim['recommendation'][:500]}"
            )

        sections.append("\n".join(simulation_lines))

    # if "simulation" in twin_outputs and twin_outputs["simulation"]:
    #     sim = twin_outputs["simulation"]
    #     if "recommendation" in sim:
    #         sections.append(
    #             "=== TREATMENT SIMULATION ===\n"
    #             f"Agent Recommendation:\n{sim['recommendation'][:500]}"
    #         )

    if "simulation" in twin_outputs and twin_outputs["simulation"]:
        sim = twin_outputs["simulation"]
        simulation_lines = ["=== TREATMENT SIMULATION ==="]

        tool_trace = sim.get("tool_trace", [])

        for tool in tool_trace:
            name = tool.get("tool")
            result = tool.get("result", {})

            if name == "check_affordability":
                simulation_lines.append(
                    f"Treatment Cost: "
                    f"Rs.{result.get('cost', tool.get('input', {}).get('cost', 0)):,.0f}"
                )

                simulation_lines.append(
                    f"Treatment Affordable From Disposable Income: "
                    f"{result.get('affordable')}"
                )

                simulation_lines.append(
                    f"Affordability Score: {result.get('score')}"
                )

            elif name == "check_insurance_coverage":
                simulation_lines.append(
                    f"Specific Treatment Covered Amount: "
                    f"Rs.{result.get('covered_amount', 0):,.0f}"
                )

                simulation_lines.append(
                    f"Specific Treatment Coverage Gap: "
                    f"Rs.{result.get('gap', 0):,.0f}"
                )

            elif name == "get_health_context":
                simulation_lines.append(
                    f"Health Risk Level: {result.get('risk_level')}"
                )

                simulation_lines.append(
                    f"Clinical Urgency: {result.get('urgency')}"
                )

        if "recommendation" in sim:
            simulation_lines.append(
                f"Agent Recommendation:\n{sim['recommendation'][:500]}"
            )

        sections.append("\n".join(simulation_lines))

    if shap_features:
        shap_text = "\n".join(
            f"  - {sf['feature']}: impact {sf['impact']:+.4f} "
            f"({sf['model']} model)"
            for sf in shap_features[:5]
        )
        sections.append(
            f"=== KEY FACTORS (SHAP Analysis) ===\n{shap_text}"
        )

    if rag_chunks:
        rag_text = "\n---\n".join(
            chunk[:500] for chunk in rag_chunks[:3]
        )
        sections.append(
            f"=== GUIDELINE REFERENCES ===\n{rag_text}"
        )

    user_prompt = (
        "Based on the following data, write a personalized health and "
        "financial wellness explanation for the user:\n\n"
        + "\n\n".join(sections)
    )

    return _call_gemini(client, system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# GENERATE RANKED WHAT-IFS
# ---------------------------------------------------------------------------

def generate_ranked_whatifs(scenarios: list[dict]) -> str:
    """Rank and narrate 3-4 candidate what-if scenarios.

    Each scenario dict should contain:
        - "name": str (e.g., "Quit smoking", "Reduce weight by 5kg")
        - "risk_reduction": float (0-1, how much risk improves)
        - "cost": float (monetary cost of the change)
        - "insurance_fit": float (0-1, how well it aligns with insurance)

    Ranking formula:
        score = risk_reduction * 0.5 + (1 / max(cost, 1)) * 0.3 + insurance_fit * 0.2

    Args:
        scenarios: List of scenario dicts.

    Returns:
        Narrated comparison text with trade-offs.
    """
    if not scenarios:
        return "No what-if scenarios to evaluate."

    # Rank scenarios
    for s in scenarios:
        risk_red = s.get("risk_reduction", 0)
        cost = max(s.get("cost", 1), 1)
        ins_fit = s.get("insurance_fit", 0)

        s["score"] = round(
            risk_red * 0.5
            + (1.0 / cost) * 0.3 * 100000
            + ins_fit * 0.2,
            4,
        )

    ranked = sorted(
        scenarios,
        key=lambda s: s["score"],
        reverse=True,
    )

    client = _get_gemini_client()

    system_prompt = (
        "You are a wellness advisor helping a user compare what-if scenarios. "
        "You are given ranked scenarios with their scores. Narrate them in a "
        "warm, clear way, explaining trade-offs between risk reduction, cost, "
        "and insurance fit. Keep it under 250 words. Include a disclaimer."
    )

    scenario_text = "\n".join(
        f"{i + 1}. {s['name']} (Score: {s['score']:.2f}) — "
        f"Risk Reduction: {s.get('risk_reduction', 0):.0%}, "
        f"Cost: Rs.{s.get('cost', 0):,.0f}, "
        f"Insurance Fit: {s.get('insurance_fit', 0):.0%}"
        for i, s in enumerate(ranked[:4])
    )

    user_prompt = (
        f"Narrate these ranked what-if scenarios for the user, "
        f"explaining trade-offs:\n\n{scenario_text}"
    )

    return _call_gemini(client, system_prompt, user_prompt)


# ======================================================================
# STANDALONE VERIFICATION (python narration_agent.py)
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Module 7: Narration Agent — Explanation Test")
    print("=" * 60)

    # Test with sample data (no twin initialization needed)
    sample_twin_outputs = {
        "health": {
            "bmi": 26.3,
            "obesity_class": "Overweight (Pre-obese)",
            "hypertension_stage": "Stage 1",
            "cardio_risk": {
                "probability": 0.42,
                "label": "Elevated",
            },
            "diabetes_risk": {
                "probability": 0.15,
                "label": "Low",
            },
            "overall_health_score": 58,
        },
        "finance": {
            "dti_ratio": 0.35,
            "savings_rate": 0.25,
            "emergency_fund_months": 4.2,
            "financial_stability_score": 62,
        },
    }

    sample_shap = [
        {
            "feature": "ap_hi",
            "impact": 0.0823,
            "model": "cardio",
        },
        {
            "feature": "bmi",
            "impact": 0.0615,
            "model": "cardio",
        },
        {
            "feature": "age_years",
            "impact": 0.0512,
            "model": "cardio",
        },
    ]

    sample_rag = [
        "For Hypertension Stage 1, if the 10-year cardiovascular risk "
        "exceeds 10%, antihypertensive medication should be initiated "
        "alongside lifestyle changes. Target blood pressure should be "
        "below 130/80 mmHg. (Source: AHA/ACC 2017 Guidelines)",
    ]

    print("\n[1] Generating personalized explanation...")

    explanation = generate_explanation(
        sample_twin_outputs,
        sample_shap,
        sample_rag,
    )

    print(f"\n{explanation}")

    # Test what-if ranking
    print(f"\n{'=' * 60}")
    print("  What-If Scenario Ranking")
    print("=" * 60)

    sample_scenarios = [
        {
            "name": "Quit smoking",
            "risk_reduction": 0.35,
            "cost": 0,
            "insurance_fit": 0.8,
        },
        {
            "name": "Lose 5kg through diet and exercise",
            "risk_reduction": 0.20,
            "cost": 5000,
            "insurance_fit": 0.5,
        },
        {
            "name": "Start blood pressure medication",
            "risk_reduction": 0.30,
            "cost": 12000,
            "insurance_fit": 0.9,
        },
    ]

    print("\n[2] Generating ranked what-if narration...")

    whatif_text = generate_ranked_whatifs(sample_scenarios)

    print(f"\n{whatif_text}")