"""
Module 6 -- orchestrator.py
LangGraph-based orchestrator for TwinLife AI.

Defines a state-graph with nodes for each twin (health, finance, insurance),
a RAG node, a narration node, and a combined-simulator node. Routes queries
using two-step logic: keyword matching first, LLM fallback for ambiguous cases.

Usage:
    orch = Orchestrator(user_health_inputs, user_finance_inputs,
                        user_insurance_inputs)
    result = orch.process_query("Can I afford this surgery?")
"""

from __future__ import annotations

import os
import re
import json
import warnings
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from langgraph.graph import StateGraph, END

from health_twin import HealthTwin
from finance_twin import FinanceTwin
from insurance_twin import InsuranceTwin
from rag_agent import RAGAgent
from narration_agent import generate_explanation, generate_ranked_whatifs
from combined_simulator import CombinedSimulator

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
#  KEYWORD ROUTING TABLES (Step 1: deterministic, no API call)
# ---------------------------------------------------------------------------
HEALTH_KEYWORDS = [
    "diabetes", "weight", "blood pressure", "bmi", "obesity", "cholesterol",
    "cardio", "heart", "glucose", "hba1c", "hypertension", "smoking",
    "fitness", "health", "risk", "bp",
]
FINANCE_KEYWORDS = [
    "save", "saving", "afford", "loan", "emi", "debt", "income", "expense",
    "budget", "investment", "financial", "money", "salary", "dti",
    "emergency fund", "disposable",
]
INSURANCE_KEYWORDS = [
    "coverage", "premium", "claim", "rider", "insured", "policy",
    "insurance", "sum insured", "deductible", "cashless",
]
CROSS_DOMAIN_KEYWORDS = [
    "treatment", "medication", "surgery", "procedure", "hospital",
    "operation",
]
COST_KEYWORDS = [
    "cost", "price", "expensive", "afford", "pay", "rs", "rupee", "lakh",
    "crore", "inr", "₹",
]


# ---------------------------------------------------------------------------
#  ORCHESTRATOR STATE
# ---------------------------------------------------------------------------
class OrchestratorState(dict):
    """TypedDict-like state for the LangGraph orchestrator.

    Keys:
        query (str):              The user's natural-language query.
        routing (dict):           Routing decision from route_query().
        health_profile (dict):    Output from HealthTwin.assess().
        finance_profile (dict):   Output from FinanceTwin.assess().
        insurance_profile (dict): Output from InsuranceTwin.assess().
        rag_chunks (list[str]):   Retrieved guideline chunks.
        simulation_result (dict): Output from CombinedSimulator.
        narration (str):          Final narrated explanation.
        error (str | None):       Error message if any step fails.
    """
    pass


# ---------------------------------------------------------------------------
#  ORCHESTRATOR CLASS
# ---------------------------------------------------------------------------
class Orchestrator:
    """LangGraph-based orchestrator coordinating all TwinLife AI modules.

    Routes queries to the appropriate twin(s), retrieves RAG context,
    and narrates the final explanation using the Gemini API.

    Args:
        health_inputs:    Dict of user health vitals (for HealthTwin).
        finance_inputs:   Dict of user financial data (for FinanceTwin).
        insurance_inputs: Dict of user insurance data (for InsuranceTwin).
    """

    def __init__(
        self,
        health_inputs: dict,
        finance_inputs: dict,
        insurance_inputs: dict,
    ) -> None:
        self.health_inputs = health_inputs
        self.finance_inputs = finance_inputs
        self.insurance_inputs = insurance_inputs

        # Initialize twins
        self.health_twin = HealthTwin()
        self.finance_twin = FinanceTwin()
        self.insurance_twin = InsuranceTwin()

        # Initialize RAG agent
        self.rag_agent = RAGAgent()
        self.rag_agent.build_index()  # no-op if already indexed

        # Pre-compute profiles
        self.health_profile = self.health_twin.assess(health_inputs)
        self.finance_profile = self.finance_twin.assess(finance_inputs)
        self.insurance_profile = self.insurance_twin.assess(
            insurance_inputs, self.health_profile, self.finance_profile,
        )

        # Build LangGraph
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    #  ROUTING LOGIC (Two-Step)
    # ------------------------------------------------------------------
    def route_query(self, query: str) -> dict:
        """Route a query to the appropriate twin(s).

        Step 1: Keyword-based deterministic matching.
        Step 2: LLM fallback for ambiguous queries.

        Args:
            query: Natural language query from the user.

        Returns:
            {"twins": [...], "mode": "single-domain"|"cross-domain",
             "simulate": bool}
        """
        q_lower = query.lower()

        # --- Step 1: Keyword matching ---
        health_match = any(kw in q_lower for kw in HEALTH_KEYWORDS)
        finance_match = any(kw in q_lower for kw in FINANCE_KEYWORDS)
        insurance_match = any(kw in q_lower for kw in INSURANCE_KEYWORDS)
        cross_match = any(kw in q_lower for kw in CROSS_DOMAIN_KEYWORDS)
        cost_match = any(kw in q_lower for kw in COST_KEYWORDS)

        # Cross-domain: treatment/medication + cost word → all three + simulate
        if cross_match and cost_match:
            return {
                "twins": ["health", "finance", "insurance"],
                "mode": "cross-domain",
                "simulate": True,
            }

        # Single-domain matches
        matched_twins = []
        if health_match:
            matched_twins.append("health")
        if finance_match:
            matched_twins.append("finance")
        if insurance_match:
            matched_twins.append("insurance")

        if matched_twins:
            return {
                "twins": matched_twins,
                "mode": "single-domain" if len(matched_twins) == 1 else "cross-domain",
                "simulate": cross_match or (len(matched_twins) > 1 and cost_match),
            }

        # --- Step 2: LLM fallback for ambiguous queries ---
        return self._llm_route(query)

    def _llm_route(self, query: str) -> dict:
        """Use Gemini API to determine routing for ambiguous queries."""
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            # Fallback: route to all twins if no API key
            return {
                "twins": ["health", "finance", "insurance"],
                "mode": "cross-domain",
                "simulate": False,
            }

        client = genai.Client(api_key=api_key)

        system_prompt = (
            "You are a query router for a digital twin system with three domains: "
            "health, finance, insurance. Given a user query, respond ONLY with "
            "valid JSON (no markdown, no explanation) in this exact format:\n"
            '{"twins": ["health"], "mode": "single-domain", "simulate": false}\n'
            "Rules:\n"
            "- twins: list containing one or more of: health, finance, insurance\n"
            "- mode: 'single-domain' if only one twin, 'cross-domain' if multiple\n"
            "- simulate: true if the query involves comparing costs or what-if scenarios\n"
            "Respond with ONLY the JSON object, nothing else."
        )

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=[genai_types.Content(
                    role="user",
                    parts=[genai_types.Part.from_text(text=f"Route this query: {query}")],
                )],
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                ),
            )
            text = response.candidates[0].content.parts[0].text.strip()
            # Strip markdown code fences if present
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            result = json.loads(text)
            # Validate keys
            if "twins" in result and "mode" in result:
                return result
        except Exception:
            pass

        # Default fallback
        return {
            "twins": ["health", "finance", "insurance"],
            "mode": "cross-domain",
            "simulate": False,
        }

    # ------------------------------------------------------------------
    #  LANGGRAPH NODE FUNCTIONS
    # ------------------------------------------------------------------
    def _health_node(self, state: dict) -> dict:
        """Execute HealthTwin assessment."""
        state["health_profile"] = self.health_profile
        return state

    def _finance_node(self, state: dict) -> dict:
        """Execute FinanceTwin assessment."""
        state["finance_profile"] = self.finance_profile
        return state

    def _insurance_node(self, state: dict) -> dict:
        """Execute InsuranceTwin assessment."""
        state["insurance_profile"] = self.insurance_profile
        return state

    def _rag_node(self, state: dict) -> dict:
        """Retrieve relevant guideline chunks via RAG."""
        query = state.get("query", "")
        chunks = self.rag_agent.retrieve(query, top_k=3)
        state["rag_chunks"] = chunks
        return state

    def _simulation_node(self, state: dict) -> dict:
        """Run the Combined Simulator agent (if routing says simulate)."""
        routing = state.get("routing", {})
        if not routing.get("simulate", False):
            state["simulation_result"] = None
            return state

        try:
            # Extract cost from query (simple regex for Rs./₹ amounts)
            query = state.get("query", "")
            cost = self._extract_cost(query)
            condition = self._extract_condition(query)

            sim = CombinedSimulator(
                health_profile=self.health_profile,
                finance_twin=self.finance_twin,
                insurance_twin=self.insurance_twin,
                insurance_inputs=self.insurance_inputs,
            )
            result = sim.evaluate_treatment(cost=cost, condition=condition)
            state["simulation_result"] = result
        except Exception as e:
            state["simulation_result"] = {"error": str(e)}

        return state

    def _narration_node(self, state: dict) -> dict:
        """Generate final narrated explanation."""
        twin_outputs = {}
        if state.get("health_profile"):
            twin_outputs["health"] = state["health_profile"]
        if state.get("finance_profile"):
            twin_outputs["finance"] = state["finance_profile"]
        if state.get("insurance_profile"):
            twin_outputs["insurance"] = state["insurance_profile"]
        if state.get("simulation_result"):
            twin_outputs["simulation"] = state["simulation_result"]

        shap_features = []
        hp = state.get("health_profile", {})
        if hp:
            shap_features = hp.get("shap_top_features", [])

        rag_chunks = state.get("rag_chunks", [])

        try:
            narration = generate_explanation(
                twin_outputs=twin_outputs,
                shap_features=shap_features,
                rag_chunks=rag_chunks,
            )
            state["narration"] = narration
        except Exception as e:
            state["narration"] = f"[Narration error: {e}]"

        return state

    # ------------------------------------------------------------------
    #  GRAPH BUILDER
    # ------------------------------------------------------------------
    def _build_graph(self) -> Any:
        """Build the LangGraph state graph."""
        builder = StateGraph(dict)

        # Add nodes
        builder.add_node("route", self._route_node)
        builder.add_node("health", self._health_node)
        builder.add_node("finance", self._finance_node)
        builder.add_node("insurance", self._insurance_node)
        builder.add_node("rag", self._rag_node)
        builder.add_node("simulate", self._simulation_node)
        builder.add_node("narrate", self._narration_node)

        # Entry point
        builder.set_entry_point("route")

        # Conditional edges from route
        def route_decision(state: dict) -> str:
            """Determine next node after routing."""
            twins = state.get("routing", {}).get("twins", [])
            if "health" in twins:
                return "health"
            elif "finance" in twins:
                return "finance"
            elif "insurance" in twins:
                return "insurance"
            return "rag"

        builder.add_conditional_edges(
            "route",
            route_decision,
            {
                "health": "health",
                "finance": "finance",
                "insurance": "insurance",
                "rag": "rag",
            },
        )

        # After health → finance (if needed) or rag
        def after_health(state: dict) -> str:
            twins = state.get("routing", {}).get("twins", [])
            if "finance" in twins:
                return "finance"
            return "rag"

        builder.add_conditional_edges(
            "health", after_health, {"finance": "finance", "rag": "rag"},
        )

        # After finance → insurance (if needed) or rag
        def after_finance(state: dict) -> str:
            twins = state.get("routing", {}).get("twins", [])
            if "insurance" in twins:
                return "insurance"
            return "rag"

        builder.add_conditional_edges(
            "finance", after_finance, {"insurance": "insurance", "rag": "rag"},
        )

        # After insurance → rag
        builder.add_edge("insurance", "rag")

        # After rag → simulate (if needed) or narrate
        def after_rag(state: dict) -> str:
            if state.get("routing", {}).get("simulate", False):
                return "simulate"
            return "narrate"

        builder.add_conditional_edges(
            "rag", after_rag, {"simulate": "simulate", "narrate": "narrate"},
        )

        # After simulate → narrate
        builder.add_edge("simulate", "narrate")

        # After narrate → END
        builder.add_edge("narrate", END)

        return builder.compile()

    def _route_node(self, state: dict) -> dict:
        """Initial routing node."""
        query = state.get("query", "")
        routing = self.route_query(query)
        state["routing"] = routing
        return state

    # ------------------------------------------------------------------
    #  PUBLIC API
    # ------------------------------------------------------------------
    def process_query(self, query: str) -> dict:
        """Process a user query through the full orchestrator pipeline.

        Args:
            query: Natural language query from the user.

        Returns:
            Final state dict containing routing, profiles, RAG chunks,
            simulation result (if any), and narration.
        """
        initial_state = {
            "query": query,
            "routing": {},
            "health_profile": None,
            "finance_profile": None,
            "insurance_profile": None,
            "rag_chunks": [],
            "simulation_result": None,
            "narration": "",
            "error": None,
        }

        result = self.graph.invoke(initial_state)
        return result

    # ------------------------------------------------------------------
    #  HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_cost(query: str) -> float:
        """Extract a monetary amount from a query string."""
        # Match patterns like: Rs.500000, ₹5,00,000, 5 lakh, 500000
        patterns = [
            r"(?:rs\.?|₹|inr)\s*([\d.]+)\s*(?:lakh|lac)",
            r"(?:rs\.?|₹|inr)\s*([\d.]+)\s*(?:crore|cr)",
            r"([\d.]+)\s*(?:lakh|lac)",
            r"([\d.]+)\s*(?:crore|cr)",
            r"(?:rs\.?|₹|inr)\s*([\d,]+(?:\.\d+)?)",
            r"\b(\d{4,})\b",
        ]
        q_lower = query.lower()
        for i, pat in enumerate(patterns):
            match = re.search(pat, q_lower, re.IGNORECASE)
            if match:
                val = match.group(1).replace(",", "")
                num = float(val)
                if i in (0, 2):  # lakh
                    num *= 100_000
                elif i in (1, 3):  # crore
                    num *= 10_000_000
                return num
        return 500_000  # default fallback

    @staticmethod
    def _extract_condition(query: str) -> str:
        """Extract medical condition from a query string."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["diabetes", "glucose", "sugar", "hba1c"]):
            return "diabetes"
        if any(w in q_lower for w in ["blood pressure", "hypertension", "bp"]):
            return "hypertension"
        return "cardio"  # default


# ======================================================================
#  STANDALONE VERIFICATION (python orchestrator.py)
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Module 6: Orchestrator — Routing & Pipeline Test")
    print("=" * 60)

    # --- Test routing logic only (no full pipeline) ---
    # Create a temporary orchestrator-like object for routing tests
    class RoutingTester:
        """Minimal class to test route_query without full init."""
        route_query = Orchestrator.route_query
        _llm_route = Orchestrator._llm_route

    tester = RoutingTester()

    test_queries = [
        "What is my blood pressure status?",
        "Can I afford a ₹5,00,000 heart surgery?",
        "How much is my insurance coverage?",
        "Am I saving enough money?",
        "I need a treatment for diabetes costing Rs.300000",
        "Tell me about my overall well-being",
    ]

    for query in test_queries:
        routing = tester.route_query(query)
        print(f"\n  Query: \"{query}\"")
        print(f"  Routing: {json.dumps(routing)}")

    print("\n" + "=" * 60)
    print("  Full pipeline test requires all twins initialized.")
    print("  Use the full Orchestrator class with valid inputs.")
    print("=" * 60)
