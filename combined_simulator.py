"""
Module 4 -- combined_simulator.py
Autonomous tool-use agent for treatment-plan advisory.

Uses Google Gemini API (free tier) function calling to orchestrate
deterministic tool calls against the twin/formula functions. The LLM
decides the order, branching, and reasoning -- every number it relies on
comes from a tool call, never from the LLM itself.

Tools:
  1. check_affordability(cost)        -> {affordable, score}
  2. check_insurance_coverage(cost)   -> {covered_amount, gap}
  3. get_cheaper_alternative(condition, current_cost) -> {new_cost, tier}
  4. get_health_context(condition)     -> {risk_level, urgency}

Setup:
  1. Get a free API key from https://aistudio.google.com/apikey
  2. Set environment variable: GEMINI_API_KEY=your-key-here

Usage:
    sim = CombinedSimulator(health_profile, finance_twin,
                            insurance_twin, insurance_inputs)
    result = sim.evaluate_treatment(cost=500000, condition="cardio")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from dotenv import load_dotenv

from insurance_twin import ESTIMATED_COST_TABLE

# Load .env file (contains GEMINI_API_KEY)
load_dotenv()


# ---------------------------------------------------------------------------
#  COST TIER ORDERING (for get_cheaper_alternative)
# ---------------------------------------------------------------------------
TIER_ORDER: dict[str, list[str]] = {
    "cardio": ["Low", "Moderate", "Elevated", "High", "Very High"],
    "diabetes": ["Low", "Moderate", "Elevated", "High", "Very High"],
    "hypertension": ["Normal", "Elevated", "Stage 1", "Stage 2"],
}

URGENCY_MAP: dict[str, str] = {
    "Low":       "routine monitoring, no immediate action needed",
    "Moderate":  "schedule a specialist consultation within 1-2 months",
    "Elevated":  "consult a specialist soon, within 2-4 weeks",
    "High":      "urgent -- seek specialist attention within 1-2 weeks",
    "Very High": "critical -- immediate medical attention recommended",
    "Normal":    "routine monitoring, no immediate action needed",
    "Stage 1":   "lifestyle changes needed; consult doctor within a month",
    "Stage 2":   "urgent -- medication likely needed, see doctor this week",
}


# ---------------------------------------------------------------------------
#  TOOL FUNCTION DECLARATIONS (google-genai format)
# ---------------------------------------------------------------------------
TOOL_DECLARATIONS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="check_affordability",
            description=(
                "Check whether the user can afford a given treatment cost "
                "from their current disposable income over 12 months. "
                "Returns an affordability score (>=1.0 means fully coverable) "
                "and a boolean."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "cost": types.Schema(
                        type="NUMBER",
                        description="The treatment cost in INR.",
                    ),
                },
                required=["cost"],
            ),
        ),
        types.FunctionDeclaration(
            name="check_insurance_coverage",
            description=(
                "Check how much of a treatment cost is covered by the "
                "user's insurance policy. Returns covered_amount and gap."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "cost": types.Schema(
                        type="NUMBER",
                        description="The treatment cost in INR.",
                    ),
                },
                required=["cost"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_cheaper_alternative",
            description=(
                "Look up the next cheaper treatment tier for a given "
                "medical condition. Returns the cost and tier name of "
                "the lower-cost alternative."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "condition": types.Schema(
                        type="STRING",
                        description="Medical condition: 'cardio', 'diabetes', or 'hypertension'.",
                    ),
                    "current_cost": types.Schema(
                        type="NUMBER",
                        description="The current treatment cost in INR.",
                    ),
                },
                required=["condition", "current_cost"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_health_context",
            description=(
                "Get the user's current risk level and clinical urgency "
                "for a specific condition, based on their health profile."
            ),
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "condition": types.Schema(
                        type="STRING",
                        description="Medical condition: 'cardio', 'diabetes', or 'hypertension'.",
                    ),
                },
                required=["condition"],
            ),
        ),
    ],
)


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation for explainability."""
    tool_name: str
    tool_input: dict
    tool_result: dict
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "input": self.tool_input,
            "result": self.tool_result,
        }


class CombinedSimulator:
    """Autonomous treatment-plan advisor using Gemini API tool use.

    The agent evaluates a treatment cost by calling deterministic tool
    functions in whatever order it judges necessary, then produces a
    final recommendation with full explainability trace.
    """

    def __init__(
        self,
        health_profile: dict,
        finance_twin,
        insurance_twin,
        insurance_inputs: dict,
        model_name: str = "gemini-3.5-flash-lite",
    ) -> None:
        self.health_profile = health_profile
        self.finance_twin = finance_twin
        self.insurance_twin = insurance_twin
        self.insurance_inputs = insurance_inputs
        self.model_name = model_name

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------
    #  DETERMINISTIC TOOL IMPLEMENTATIONS
    # ------------------------------------------------------------------
    def _check_affordability(self, cost: float) -> dict:
        score = self.finance_twin.affordability_for(cost, months=12)
        return {"affordable": score >= 1.0, "score": round(score, 2)}

    def _check_insurance_coverage(self, cost: float) -> dict:
        sum_insured = self.insurance_inputs["sum_insured"]
        return self.insurance_twin.coverage_for(cost, sum_insured)

    def _get_cheaper_alternative(self, condition: str, current_cost: float) -> dict:
        condition = condition.lower().strip()
        if condition not in ESTIMATED_COST_TABLE:
            return {"error": f"Unknown condition: {condition}"}

        cost_table = ESTIMATED_COST_TABLE[condition]
        tiers = TIER_ORDER.get(condition, [])

        cheaper = [
            (tier, cost_table[tier])
            for tier in tiers
            if cost_table[tier] < current_cost
        ]
        if not cheaper:
            return {"new_cost": current_cost, "tier": "none -- already at lowest tier"}
        best = max(cheaper, key=lambda x: x[1])
        return {"new_cost": best[1], "tier": best[0]}

    def _get_health_context(self, condition: str) -> dict:
        condition = condition.lower().strip()
        if condition == "cardio":
            label = self.health_profile.get("cardio_risk", {}).get("label", "Low")
        elif condition == "diabetes":
            label = self.health_profile.get("diabetes_risk", {}).get("label", "Low")
        elif condition == "hypertension":
            label = self.health_profile.get("hypertension_stage", "Normal")
        else:
            return {"error": f"Unknown condition: {condition}"}

        urgency = URGENCY_MAP.get(label, "consult a specialist")
        return {"risk_level": label, "urgency": urgency}

    def _dispatch_tool(self, name: str, args: dict) -> dict:
        if name == "check_affordability":
            return self._check_affordability(args["cost"])
        elif name == "check_insurance_coverage":
            return self._check_insurance_coverage(args["cost"])
        elif name == "get_cheaper_alternative":
            return self._get_cheaper_alternative(args["condition"], args["current_cost"])
        elif name == "get_health_context":
            return self._get_health_context(args["condition"])
        else:
            return {"error": f"Unknown tool: {name}"}

    def _call_with_retry(
        self,
        contents: list,
        system_instruction: str,
        max_retries: int = 3,
        base_wait: int = 30,
    ):
        """Call Gemini API with automatic retry on rate-limit (429) errors.

        New API keys often take 30-60 seconds before quotas activate.
        This method waits and retries instead of crashing.
        """
        for attempt in range(max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=[TOOL_DECLARATIONS],
                        temperature=0.2,
                    ),
                )
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries:
                        wait = base_wait * (attempt + 1)  # 30s, 60s, 90s
                        print(f"    [Rate limited -- waiting {wait}s before retry "
                              f"{attempt + 1}/{max_retries}...]")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Gemini API rate limit after {max_retries} retries. "
                            "Wait a few minutes and try again."
                        ) from e
                else:
                    raise  # Non-rate-limit errors propagate immediately

    # ------------------------------------------------------------------
    #  MAIN AGENT LOOP
    # ------------------------------------------------------------------
    def evaluate_treatment(
        self,
        cost: float,
        condition: str = "cardio",
        max_iterations: int = 10,
    ) -> dict:
        """Run the autonomous tool-use agent to evaluate a treatment plan.

        Returns:
            {"recommendation": str, "tool_trace": [...], "iterations": int}
        """
        system_instruction = (
            f"You are a treatment-plan advisor agent evaluating a Rs.{cost:,.0f} "
            f"treatment plan for the condition '{condition}'. "
            "Your objective: determine the most affordable, appropriate way "
            "for the user to proceed, minimizing financial strain while "
            "addressing their health need. You have the tools listed. "
            "Reason step by step, call tools as needed, in whatever order "
            "you judge necessary, and stop once you reach a clear "
            "recommendation. Prefer self-funding if genuinely affordable; "
            "otherwise explore insurance coverage and lower-cost alternatives "
            "before concluding the plan is currently out of reach. "
            "IMPORTANT: All monetary figures must come from tool calls -- "
            "do not invent numbers. Include a disclaimer that this is "
            "illustrative guidance, not licensed medical or financial advice."
        )

        user_message = (
            f"Please evaluate a treatment plan costing Rs.{cost:,.0f} "
            f"for my {condition} condition and recommend the best "
            "funding approach."
        )

        # Build conversation history for the loop
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            ),
        ]

        tool_trace: list[ToolCallRecord] = []
        recommendation = ""
        iteration = 0

        for iteration in range(max_iterations):
            # Retry with backoff for rate-limit errors (common with new keys)
            response = self._call_with_retry(
                contents=contents,
                system_instruction=system_instruction,
            )

            # Collect function calls and text from the response
            function_calls = []
            text_parts = []

            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
                elif part.text:
                    text_parts.append(part.text)

            # Add the model's response to conversation history
            contents.append(response.candidates[0].content)

            # If no function calls, we have the final answer
            if not function_calls:
                recommendation += "\n".join(text_parts)
                break

            # Capture intermediate reasoning
            if text_parts:
                recommendation += "\n".join(text_parts) + "\n"

            # Execute function calls and build response parts
            fn_response_parts = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args)
                tool_result = self._dispatch_tool(tool_name, tool_args)

                record = ToolCallRecord(
                    tool_name=tool_name,
                    tool_input=tool_args,
                    tool_result=tool_result,
                )
                tool_trace.append(record)

                fn_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_result},
                    ),
                )

            # Send function results back
            contents.append(
                types.Content(
                    role="user",
                    parts=fn_response_parts,
                ),
            )
        else:
            # Max iterations -- grab any remaining text
            for part in response.candidates[0].content.parts:
                if part.text:
                    recommendation += part.text

        return {
            "recommendation": recommendation.strip(),
            "tool_trace": [r.to_dict() for r in tool_trace],
            "iterations": iteration + 1,
        }


# ======================================================================
#  STANDALONE QUICK TEST
# ======================================================================
if __name__ == "__main__":
    print("Use test_combined_simulator.py for full test runs.")
    print("This module provides the CombinedSimulator class.")
    print(f"Tools registered: {len(TOOL_DECLARATIONS.function_declarations)}")
