"""
Module 3B -- finance_twin.py
FinanceTwin: rule-based financial health assessment using formulas.py only
(no ML model).

Usage:
    twin = FinanceTwin()
    profile = twin.assess(inputs_dict)
    score   = twin.affordability_for(cost)
"""

from formulas import (
    dti_ratio,
    savings_rate,
    emergency_fund_months,
    financial_stability_score,
    affordability_score,
)


class FinanceTwin:
    """Digital twin for financial health assessment.

    All computations are deterministic rule/formula-based (no ML model).
    The twin stores the latest inputs so that affordability_for() can be
    called without re-supplying the full input dict.

    Attributes:
        latest_inputs: The most recent inputs dict passed to assess().
        latest_profile: The most recent computed finance_profile.
    """

    def __init__(self) -> None:
        self.latest_inputs: dict | None = None
        self.latest_profile: dict | None = None

    def assess(self, inputs: dict) -> dict:
        """Compute a full financial profile.

        Args:
            inputs: Dict with keys:
                income (float)            - monthly gross income
                fixed_expenses (float)    - monthly fixed outflows (rent, etc.)
                variable_expenses (float) - monthly variable outflows
                emis (float)              - monthly loan EMI payments
                savings_balance (float)   - total liquid savings

        Returns:
            finance_profile dict matching the spec shape:
            {
                "dti_ratio": float,
                "savings_rate": float,
                "emergency_fund_months": float,
                "financial_stability_score": int,  # 0-100
                "disposable_income": float,
            }
        """
        income   = inputs["income"]
        fixed    = inputs["fixed_expenses"]
        variable = inputs["variable_expenses"]
        emis_val = inputs["emis"]
        savings  = inputs["savings_balance"]

        dti   = dti_ratio(fixed, emis_val, income)
        s_rate = savings_rate(income, fixed, variable)
        ef_months = emergency_fund_months(savings, fixed, variable)
        stability = financial_stability_score(dti, s_rate, ef_months)
        disposable = income - fixed - variable - emis_val

        profile = {
            "dti_ratio": dti,
            "savings_rate": s_rate,
            "emergency_fund_months": ef_months,
            "financial_stability_score": stability,
            "disposable_income": round(disposable, 2),
        }

        # Cache for affordability_for()
        self.latest_inputs = inputs
        self.latest_profile = profile
        return profile

    def affordability_for(self, cost: float, months: float = 12) -> float:
        """Check how affordable a one-time cost is given current finances.

        Uses the disposable income and a configurable saving horizon.

        Args:
            cost:   Total cost of the item/treatment.
            months: Months available to save for the cost (default 12).

        Returns:
            Affordability score (>= 1.0 means fully coverable).

        Raises:
            RuntimeError: If assess() has not been called yet.
        """
        if self.latest_inputs is None or self.latest_profile is None:
            raise RuntimeError(
                "Call assess() before affordability_for() so that "
                "disposable income is available."
            )
        disposable = self.latest_profile["disposable_income"]
        return affordability_score(disposable, months, cost)


# ======================================================================
#  STANDALONE VERIFICATION (python finance_twin.py)
# ======================================================================
if __name__ == "__main__":
    import json

    twin = FinanceTwin()

    samples = [
        {
            "label": "Comfortable salaried professional",
            "inputs": {
                "income": 120000,
                "fixed_expenses": 30000,
                "variable_expenses": 20000,
                "emis": 15000,
                "savings_balance": 500000,
            },
        },
        {
            "label": "Stretched household budget",
            "inputs": {
                "income": 50000,
                "fixed_expenses": 20000,
                "variable_expenses": 15000,
                "emis": 12000,
                "savings_balance": 30000,
            },
        },
        {
            "label": "High-earner with heavy EMIs",
            "inputs": {
                "income": 250000,
                "fixed_expenses": 60000,
                "variable_expenses": 40000,
                "emis": 80000,
                "savings_balance": 200000,
            },
        },
    ]

    for s in samples:
        print("=" * 60)
        print(f"  {s['label']}")
        print("=" * 60)
        profile = twin.assess(s["inputs"])
        print(json.dumps(profile, indent=2))

        # Test affordability for a 500,000 treatment cost
        aff = twin.affordability_for(500000)
        print(f"  Affordability for 500,000 (12 months): {aff}")
        print()





