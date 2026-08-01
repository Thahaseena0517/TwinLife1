"""
Module 3C -- insurance_twin.py
InsuranceTwin: rule-based insurance adequacy assessment.

Uses the health_profile (from HealthTwin) and finance_profile (from
FinanceTwin) to estimate future medical costs, evaluate coverage gaps,
and compute an overall insurance adequacy score.

Usage:
    twin = InsuranceTwin()
    profile = twin.assess(inputs, health_profile, finance_profile)
"""

from formulas import (
    coverage_adequacy_ratio,
    premium_affordability_ratio,
    insurance_adequacy_score,
)


# ---------------------------------------------------------------------------
#  ESTIMATED FUTURE COST LOOKUP
# ---------------------------------------------------------------------------
#  Maps (condition_label, severity) -> estimated 5-year medical cost (INR).
#  These are illustrative figures for academic demonstration only.
ESTIMATED_COST_TABLE: dict[str, dict[str, float]] = {
    "cardio": {
        "Low":       50_000,
        "Moderate":  200_000,
        "Elevated":  500_000,
        "High":      800_000,
        "Very High": 1_200_000,
    },
    "diabetes": {
        "Low":       30_000,
        "Moderate":  150_000,
        "Elevated":  400_000,
        "High":      700_000,
        "Very High": 1_000_000,
    },
    "hypertension": {
        "Normal":    10_000,
        "Elevated":  80_000,
        "Stage 1":   200_000,
        "Stage 2":   450_000,
    },
}

# Recommended riders keyed to health risk signals
RECOMMENDED_RIDERS: list[dict] = [
    {
        "rider": "Critical Illness Rider",
        "trigger": lambda hp: hp.get("cardio_risk", {}).get("label") in
                   ("High", "Very High", "Elevated"),
    },
    {
        "rider": "Diabetes Care Rider",
        "trigger": lambda hp: hp.get("diabetes_indicator") in ("Medium", "High"),
    },
    {
        "rider": "Hospital Cash Rider",
        "trigger": lambda hp: hp.get("hypertension_stage") in ("Stage 1", "Stage 2"),
    },
    {
        "rider": "Personal Accident Rider",
        "trigger": lambda _: True,  # universally recommended
    },
]


class InsuranceTwin:
    """Digital twin for insurance adequacy assessment.

    Entirely rule-based: estimates future medical costs from the health
    profile's risk labels, checks coverage against the user's policy,
    evaluates premium affordability, identifies missing riders, and
    computes a composite insurance adequacy score.

    Note: All monetary estimates are illustrative for academic purposes
    and should not be treated as licensed financial advice.
    """

    def assess(
        self,
        inputs: dict,
        health_profile: dict,
        finance_profile: dict,
    ) -> dict:
        """Compute an insurance adequacy profile.

        Args:
            inputs: Dict with keys:
                sum_insured (float)     - total sum insured under policy
                annual_premium (float)  - annual premium paid
                annual_income (float)   - annual gross income
                existing_riders (list[str]) - names of riders already held

            health_profile: Output of HealthTwin.assess()
            finance_profile: Output of FinanceTwin.assess()

        Returns:
            insurance_profile dict:
            {
                "estimated_future_cost": float,
                "cost_breakdown": dict,
                "coverage_adequacy_ratio": float,
                "premium_affordability_ratio": float,
                "rider_analysis": {
                    "recommended": list[str],
                    "existing": list[str],
                    "gaps": list[str],
                    "gap_count": int,
                },
                "insurance_adequacy_score": int,   # 0-100
                "disclaimer": str,
            }
        """
        sum_insured    = inputs["sum_insured"]
        annual_premium = inputs["annual_premium"]
        annual_income  = inputs["annual_income"]
        existing       = [r.lower().strip() for r in inputs.get("existing_riders", [])]

        # --- Estimated future cost (sum across conditions) ---
        cost_breakdown = self._estimate_future_costs(health_profile)
        total_cost = sum(cost_breakdown.values())

        # --- Coverage ratio ---
        cov_ratio = coverage_adequacy_ratio(sum_insured, total_cost)

        # --- Premium affordability ---
        prem_ratio = premium_affordability_ratio(annual_premium, annual_income)

        # --- Rider gap analysis ---
        recommended = []
        for r in RECOMMENDED_RIDERS:
            if r["trigger"](health_profile):
                recommended.append(r["rider"])
        gaps = [r for r in recommended if r.lower().strip() not in existing]

        # --- Composite score ---
        score = insurance_adequacy_score(cov_ratio, prem_ratio, len(gaps))

        return {
            "estimated_future_cost": round(total_cost, 2),
            "cost_breakdown": cost_breakdown,
            "coverage_adequacy_ratio": cov_ratio,
            "premium_affordability_ratio": prem_ratio,
            "rider_analysis": {
                "recommended": recommended,
                "existing": inputs.get("existing_riders", []),
                "gaps": gaps,
                "gap_count": len(gaps),
            },
            "insurance_adequacy_score": score,
            "disclaimer": (
                "These estimates are illustrative guidance for academic "
                "purposes only and do not constitute licensed financial or "
                "insurance advice."
            ),
        }

    # ------------------------------------------------------------------
    #  Cost-lookup for the combined simulator (Module 4)
    # ------------------------------------------------------------------
    def coverage_for(self, cost: float, sum_insured: float) -> dict:
        """How much of a specific cost the insurance covers.

        Args:
            cost:        The treatment/procedure cost.
            sum_insured: Current sum insured.

        Returns:
            {"covered_amount": float, "gap": float}
        """
        covered = min(sum_insured, cost)
        return {"covered_amount": covered, "gap": max(0, cost - covered)}

    # ------------------------------------------------------------------
    #  HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_future_costs(health_profile: dict) -> dict:
        """Look up estimated future costs from health risk labels."""
        breakdown = {}

        # Cardio
        cardio_label = health_profile.get("cardio_risk", {}).get("label", "Low")
        breakdown["cardio"] = ESTIMATED_COST_TABLE["cardio"].get(cardio_label, 50_000)

        # Diabetes
        diabetes_label = health_profile.get("diabetes_risk", {}).get("label", "Low")
        breakdown["diabetes"] = ESTIMATED_COST_TABLE["diabetes"].get(diabetes_label, 30_000)

        # Hypertension
        bp_stage_str = health_profile.get("hypertension_stage", "Normal")
        breakdown["hypertension"] = ESTIMATED_COST_TABLE["hypertension"].get(
            bp_stage_str, 10_000,
        )

        return breakdown


# ======================================================================
#  STANDALONE VERIFICATION (python insurance_twin.py)
# ======================================================================
if __name__ == "__main__":
    import json

    twin = InsuranceTwin()

    # We simulate health & finance profiles inline (normally from the twins)
    test_cases = [
        {
            "label": "Well-insured healthy person",
            "inputs": {
                "sum_insured": 1_500_000,
                "annual_premium": 25_000,
                "annual_income": 1_200_000,
                "existing_riders": [
                    "Critical Illness Rider",
                    "Personal Accident Rider",
                ],
            },
            "health_profile": {
                "cardio_risk": {"probability": 0.12, "label": "Low"},
                "diabetes_risk": {"probability": 0.05, "label": "Low"},
                "diabetes_indicator": "Low",
                "hypertension_stage": "Normal",
            },
            "finance_profile": {
                "dti_ratio": 0.25,
                "savings_rate": 0.45,
                "financial_stability_score": 78,
            },
        },
        {
            "label": "Under-insured high-risk person",
            "inputs": {
                "sum_insured": 300_000,
                "annual_premium": 40_000,
                "annual_income": 600_000,
                "existing_riders": [],
            },
            "health_profile": {
                "cardio_risk": {"probability": 0.65, "label": "High"},
                "diabetes_risk": {"probability": 0.45, "label": "Elevated"},
                "diabetes_indicator": "High",
                "hypertension_stage": "Stage 2",
            },
            "finance_profile": {
                "dti_ratio": 0.55,
                "savings_rate": 0.10,
                "financial_stability_score": 32,
            },
        },
        {
            "label": "Moderate risk, partial coverage",
            "inputs": {
                "sum_insured": 800_000,
                "annual_premium": 35_000,
                "annual_income": 900_000,
                "existing_riders": ["Personal Accident Rider"],
            },
            "health_profile": {
                "cardio_risk": {"probability": 0.35, "label": "Moderate"},
                "diabetes_risk": {"probability": 0.15, "label": "Low"},
                "diabetes_indicator": "Medium",
                "hypertension_stage": "Stage 1",
            },
            "finance_profile": {
                "dti_ratio": 0.38,
                "savings_rate": 0.20,
                "financial_stability_score": 55,
            },
        },
    ]

    for tc in test_cases:
        print("=" * 60)
        print(f"  {tc['label']}")
        print("=" * 60)
        profile = twin.assess(
            tc["inputs"], tc["health_profile"], tc["finance_profile"],
        )
        print(json.dumps(profile, indent=2))
        print()
