"""
TwinLife AI integration service.

Wires existing twins, RAG, LangGraph orchestrator, combined simulator,
narration, recommendation engines, and monitoring into one backend flow.

Does not reimplement scoring, routing, RAG, or agent tool logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance_twin import FinanceTwin
from health_twin import HealthTwin
from insurance_twin import InsuranceTwin
from monitoring_agent import MonitoringAgent
from orchestrator import Orchestrator
from rag_agent import RAGAgent
from recommendation_engines import (
    investment_recommendation,
    lifestyle_impact_simulator,
    purchase_impact_simulator,
)

HEALTH_REQUIRED = (
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
)

FINANCE_REQUIRED = (
    "income",
    "fixed_expenses",
    "variable_expenses",
    "emis",
    "savings_balance",
)

INSURANCE_REQUIRED = (
    "sum_insured",
    "annual_premium",
    "annual_income",
)

DEFAULT_PROFILE: dict[str, dict] = {
    "health": {
        "age": 45,
        "gender": "male",
        "height_cm": 175,
        "weight_kg": 85,
        "ap_hi": 140,
        "ap_lo": 90,
        "cholesterol": 2,
        "gluc": 2,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "former",
        "HbA1c_level": 6.2,
        "blood_glucose_level": 115,
    },
    "finance": {
        "income": 80000,
        "fixed_expenses": 25000,
        "variable_expenses": 20000,
        "emis": 18000,
        "savings_balance": 150000,
    },
    "insurance": {
        "sum_insured": 500000,
        "annual_premium": 30000,
        "annual_income": 960000,
        "existing_riders": ["Personal Accident Rider"],
    },
}


class ProfileValidationError(ValueError):
    """Raised when user profile inputs are missing or invalid."""


class TwinLifeService:
    """Single programmatic backend for the existing TwinLife modules.

    Flow:
        query → Orchestrator.route_query
              → LangGraph
              → required twins
              → RAG
              → optional simulation
              → narration
              → structured response
    """

    def __init__(
        self,
        profile: dict | None = None,
        *,
        user_id: str = "default_user",
        health_inputs: dict | None = None,
        finance_inputs: dict | None = None,
        insurance_inputs: dict | None = None,
    ) -> None:
        profile = profile or {}

        self.user_id = user_id

        self.health_inputs = dict(
            health_inputs
            or profile.get("health")
            or DEFAULT_PROFILE["health"]
        )

        self.finance_inputs = dict(
            finance_inputs
            or profile.get("finance")
            or DEFAULT_PROFILE["finance"]
        )

        self.insurance_inputs = dict(
            insurance_inputs
            or profile.get("insurance")
            or DEFAULT_PROFILE["insurance"]
        )

        self._validate_profile()

        self.health_twin = HealthTwin()
        self.finance_twin = FinanceTwin()
        self.insurance_twin = InsuranceTwin()

        self.rag_agent = RAGAgent()
        self.rag_agent.build_index()

        self.health_profile = self.health_twin.assess(
            self.health_inputs
        )

        self.finance_profile = self.finance_twin.assess(
            self.finance_inputs
        )

        self.insurance_profile = self.insurance_twin.assess(
            self.insurance_inputs,
            self.health_profile,
            self.finance_profile,
        )

        self.orchestrator = Orchestrator(
            self.health_inputs,
            self.finance_inputs,
            self.insurance_inputs,
        )

        self.health_profile = self.orchestrator.health_profile
        self.finance_profile = self.orchestrator.finance_profile
        self.insurance_profile = self.orchestrator.insurance_profile

    def _validate_profile(self) -> None:
        missing: list[str] = []

        missing.extend(
            _missing_keys(
                "health",
                self.health_inputs,
                HEALTH_REQUIRED,
            )
        )

        missing.extend(
            _missing_keys(
                "finance",
                self.finance_inputs,
                FINANCE_REQUIRED,
            )
        )

        missing.extend(
            _missing_keys(
                "insurance",
                self.insurance_inputs,
                INSURANCE_REQUIRED,
            )
        )

        if missing:
            raise ProfileValidationError(
                "Invalid user profile. Missing required fields: "
                + ", ".join(missing)
            )

        if not isinstance(
            self.health_inputs.get("gender"),
            str,
        ):
            raise ProfileValidationError(
                "health.gender must be a string such as 'male' or 'female'."
            )

        if (
            float(self.health_inputs["height_cm"]) <= 0
            or float(self.health_inputs["weight_kg"]) <= 0
        ):
            raise ProfileValidationError(
                "health.height_cm and health.weight_kg must be positive."
            )

        if float(self.finance_inputs["income"]) <= 0:
            raise ProfileValidationError(
                "finance.income must be positive."
            )

        if float(self.insurance_inputs["sum_insured"]) < 0:
            raise ProfileValidationError(
                "insurance.sum_insured cannot be negative."
            )

        if "existing_riders" not in self.insurance_inputs:
            self.insurance_inputs["existing_riders"] = []

    def process_query(self, query: str) -> dict[str, Any]:
        """Run the full orchestrator pipeline and normalize the response."""

        if not query or not str(query).strip():
            raise ProfileValidationError(
                "Query must be a non-empty string."
            )

        raw = self.orchestrator.process_query(
            str(query).strip()
        )

        return self._normalize_response(raw)

    def affordability_for(
        self,
        cost: float,
        months: float = 12,
    ) -> dict[str, Any]:
        """Ask FinanceTwin whether a treatment cost is affordable."""

        score = self.finance_twin.affordability_for(
            float(cost),
            months=months,
        )

        return {
            "cost": float(cost),
            "months": months,
            "score": score,
            "affordable": score >= 1.0,
            "source": "FinanceTwin.affordability_for",
        }

    def coverage_for(
        self,
        cost: float,
    ) -> dict[str, Any]:
        """Ask InsuranceTwin how much of a treatment cost is covered."""

        result = self.insurance_twin.coverage_for(
            float(cost),
            float(self.insurance_inputs["sum_insured"]),
        )

        result = dict(result)

        result["cost"] = float(cost)
        result["sum_insured"] = float(
            self.insurance_inputs["sum_insured"]
        )
        result["source"] = "InsuranceTwin.coverage_for"

        return result

    def recommendations(
        self,
        purchase_cost: float | None = None,
        habit_changes: dict | None = None,
    ) -> dict[str, Any]:
        """Call existing recommendation engines."""

        age = int(self.health_inputs["age"])

        result: dict[str, Any] = {
            "investment": investment_recommendation(
                self.finance_profile,
                age,
            ),
            "lifestyle": lifestyle_impact_simulator(
                self.health_profile,
                habit_changes or {},
            ),
        }

        if purchase_cost is not None:
            result["purchase"] = purchase_impact_simulator(
                self.finance_profile,
                float(purchase_cost),
            )

        return result

    def run_monitor_check(self) -> list[dict]:
        """Run one MonitoringAgent cycle."""

        monitor = MonitoringAgent(
            user_id=self.user_id,
            health_inputs=self.health_inputs,
            finance_inputs=self.finance_inputs,
            insurance_inputs=self.insurance_inputs,
        )

        return monitor.run_check()

    @staticmethod
    def _normalize_response(
        raw: dict,
    ) -> dict[str, Any]:
        """Convert orchestrator state into stable API response."""

        routing = raw.get("routing") or {}
        twins = routing.get("twins") or []

        health = (
            raw.get("health_profile")
            if "health" in twins
            else None
        )

        finance = (
            raw.get("finance_profile")
            if "finance" in twins
            else None
        )

        insurance = (
            raw.get("insurance_profile")
            if "insurance" in twins
            else None
        )

        return {
            "query": raw.get("query", ""),
            "routing": routing,
            "health": health,
            "finance": finance,
            "insurance": insurance,
            "simulation": raw.get("simulation_result"),
            "rag": raw.get("rag_chunks") or [],
            "explanation": raw.get("narration") or "",
            "error": raw.get("error"),
        }


def _missing_keys(
    domain: str,
    data: dict,
    required: tuple[str, ...],
) -> list[str]:
    if not isinstance(data, dict):
        return [
            f"{domain} (entire section missing or not a dict)"
        ]

    return [
        f"{domain}.{key}"
        for key in required
        if key not in data
    ]