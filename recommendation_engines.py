"""
Module 9 -- recommendation_engines.py
Finance/Health Recommendation Engines for TwinLife AI.

Three recommendation engines:
  1. investment_recommendation(finance_profile, age) -> str
  2. purchase_impact_simulator(finance_profile, purchase_cost) -> dict
  3. lifestyle_impact_simulator(health_profile, habit_change) -> dict

All engines are rule-based (no ML models) and include appropriate
disclaimers as required by the project spec.

Usage:
    rec = investment_recommendation(finance_profile, age=35)
    impact = purchase_impact_simulator(finance_profile, 500000)
    lifestyle = lifestyle_impact_simulator(health_profile, {"quit_smoking": True})
"""

from __future__ import annotations

from formulas import (
    dti_ratio,
    savings_rate,
    emergency_fund_months,
    financial_stability_score,
    affordability_score,
    bmi as calc_bmi,
    bp_stage,
    framingham_cvd_risk,
)

DISCLAIMER_FINANCE = (
    "⚠ Disclaimer: This is illustrative guidance for academic purposes only "
    "and does not constitute licensed financial advice. Consult a SEBI-registered "
    "investment advisor for personalized recommendations."
)

DISCLAIMER_HEALTH = (
    "⚠ Disclaimer: This is illustrative guidance for academic purposes only "
    "and does not constitute licensed medical advice. Consult a qualified "
    "healthcare professional before making any lifestyle changes."
)


# ═══════════════════════════════════════════════════════════════════════
#  1. INVESTMENT RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════
def investment_recommendation(
    finance_profile: dict,
    age: int,
) -> str:
    """Generate a rule-based investment recommendation.

    Tier logic (from spec):
      - Emergency fund < 3 months → Build emergency fund first
      - Stability >= 70 AND age < 40 → Growth allocation
      - Stability >= 70 AND age >= 40 → Balanced allocation
      - Stability 40-69 → Conservative allocation
      - Stability < 40 → No new investment, focus on debt reduction

    Args:
        finance_profile: Output of FinanceTwin.assess().
        age: User's age in years.

    Returns:
        Multi-line recommendation string with rationale and disclaimer.
    """
    stability = finance_profile.get("financial_stability_score", 0)
    ef_months = finance_profile.get("emergency_fund_months", 0)
    dti = finance_profile.get("dti_ratio", 0)
    sav_rate = finance_profile.get("savings_rate", 0)

    lines = ["📊 INVESTMENT RECOMMENDATION", "─" * 40]

    # --- Tier 1: Emergency fund deficiency ---
    if ef_months < 3:
        lines.extend([
            f"Current emergency fund: {ef_months:.1f} months",
            f"⚠ Priority: Build your emergency fund to at least 3 months",
            "",
            "Your emergency fund covers fewer than 3 months of expenses,",
            "which leaves you vulnerable to unexpected costs (medical bills,",
            "job loss, car repairs).",
            "",
            "Recommended actions:",
            "  1. Set aside a fixed amount each month into a liquid savings",
            "     account or overnight mutual fund.",
            f"  2. Target: Rs.{finance_profile.get('disposable_income', 0) * 3:,.0f}",
            "     (3 months × your current monthly expenses).",
            "  3. Avoid locking these funds in equity or long-term instruments.",
            "  4. Once you reach 3 months, consider extending to 6 months.",
        ])

    # --- Tier 2: Growth allocation (high stability, young) ---
    elif stability >= 70 and age < 40:
        lines.extend([
            f"Financial Stability Score: {stability}/100 (Strong)",
            f"Age: {age} (Growth phase)",
            "",
            "Your strong financial health and young age support a",
            "growth-oriented investment strategy.",
            "",
            "Recommended allocation:",
            "  • 70-80% Equity (index funds, diversified equity mutual funds)",
            "  • 10-20% Debt (PPF, debt mutual funds, corporate bonds)",
            "  • 5-10% Gold/Alternatives (gold ETFs, REITs)",
            "",
            "Suggested instruments:",
            "  • Nifty 50 / Nifty Next 50 index funds (low-cost equity exposure)",
            "  • SIP (Systematic Investment Plan) for rupee-cost averaging",
            "  • PPF for tax-efficient long-term debt allocation",
        ])

    # --- Tier 3: Balanced allocation (high stability, older) ---
    elif stability >= 70 and age >= 40:
        lines.extend([
            f"Financial Stability Score: {stability}/100 (Strong)",
            f"Age: {age} (Preservation phase)",
            "",
            "Your strong finances but closer proximity to retirement",
            "suggest a balanced approach.",
            "",
            "Recommended allocation:",
            "  • 50-60% Equity (large-cap / balanced advantage funds)",
            "  • 30-40% Debt (PPF, debt funds, senior citizen savings)",
            "  • 5-10% Gold/Alternatives (sovereign gold bonds, REITs)",
            "",
            "Key considerations:",
            "  • Gradually shift from equity to debt as you approach retirement",
            "  • Ensure adequate health insurance coverage",
            "  • Consider systematic withdrawal plans (SWPs) for income",
        ])

    # --- Tier 4: Conservative allocation (moderate stability) ---
    elif stability >= 40:
        lines.extend([
            f"Financial Stability Score: {stability}/100 (Moderate)",
            f"DTI Ratio: {dti:.2f} | Savings Rate: {sav_rate:.0%}",
            "",
            "Your financial position has room for improvement.",
            "A conservative approach protects existing savings while",
            "building stability.",
            "",
            "Recommended allocation:",
            "  • 30-40% Equity (large-cap funds, balanced funds only)",
            "  • 50-60% Debt (FDs, debt funds, PPF)",
            "  • 5-10% Gold (sovereign gold bonds)",
            "",
            "Priority actions:",
            "  • Reduce high-interest debt to improve DTI ratio",
            "  • Build emergency fund to at least 6 months",
            "  • Avoid speculative investments until stability score > 70",
        ])

    # --- Tier 5: No new investment (low stability) ---
    else:
        lines.extend([
            f"Financial Stability Score: {stability}/100 (Needs Attention)",
            f"DTI Ratio: {dti:.2f} | Savings Rate: {sav_rate:.0%}",
            "",
            "⚠ Focus on financial recovery before investing.",
            "",
            "Your current financial stress level suggests that new",
            "investments could increase risk without providing benefit.",
            "",
            "Recommended actions:",
            "  1. Prioritize paying off high-interest debt (credit cards first)",
            "  2. Reduce monthly expenses to improve savings rate",
            "  3. Build a minimum emergency fund (1-2 months of expenses)",
            "  4. Consider debt consolidation at lower interest rates",
            "  5. Seek professional financial counselling if needed",
        ])

    lines.extend(["", DISCLAIMER_FINANCE])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  2. PURCHASE IMPACT SIMULATOR
# ═══════════════════════════════════════════════════════════════════════
def purchase_impact_simulator(
    finance_profile: dict,
    purchase_cost: float,
) -> dict:
    """Simulate the financial impact of a major purchase/treatment.

    If the purchase would breach healthy DTI/savings thresholds,
    generates ranked alternatives (defer, cheaper tier, EMI split,
    reallocate).

    Args:
        finance_profile: Output of FinanceTwin.assess().
        purchase_cost:   Cost of the proposed purchase/treatment.

    Returns:
        {
            "affordable": bool,
            "affordability_score": float,
            "impact_analysis": {
                "new_dti_if_emi": float,
                "months_to_save": float,
                "emergency_fund_impact": str,
            },
            "alternatives": [
                {"rank": 1, "strategy": str, "detail": str, "savings": float},
                ...
            ],
            "disclaimer": str,
        }
    """
    disposable = finance_profile.get("disposable_income", 0)
    dti = finance_profile.get("dti_ratio", 0)
    ef_months = finance_profile.get("emergency_fund_months", 0)
    stability = finance_profile.get("financial_stability_score", 0)

    # Affordability check
    aff_score = affordability_score(disposable, 12, purchase_cost) if purchase_cost > 0 else float("inf")
    affordable = aff_score >= 1.0

    # Impact analysis: what if they took a 12-month EMI?
    monthly_emi = purchase_cost / 12
    # Approximate new DTI if they added this EMI
    # (income is disposable + current expenses, back-calculate)
    approx_income = disposable / max(1 - dti, 0.01) if dti < 1 else disposable * 10
    new_dti = dti + (monthly_emi / max(approx_income, 1))

    # Months needed to save the full amount
    months_to_save = purchase_cost / max(disposable, 1) if disposable > 0 else float("inf")

    # Emergency fund impact
    if purchase_cost <= finance_profile.get("disposable_income", 0) * ef_months * 0.3:
        ef_impact = "Minimal — purchase uses less than 30% of emergency fund"
    elif purchase_cost <= finance_profile.get("disposable_income", 0) * ef_months:
        ef_impact = "Moderate — would significantly reduce emergency reserves"
    else:
        ef_impact = "Severe — would deplete emergency fund entirely"

    impact = {
        "new_dti_if_emi": round(new_dti, 3),
        "monthly_emi_12months": round(monthly_emi, 0),
        "months_to_save": round(months_to_save, 1),
        "emergency_fund_impact": ef_impact,
    }

    # Generate ranked alternatives (only if not easily affordable)
    alternatives = []
    if not affordable or new_dti > 0.40:
        # Alt 1: Defer
        alternatives.append({
            "rank": 1,
            "strategy": "Defer purchase",
            "detail": (
                f"Save for {months_to_save:.0f} months at current disposable "
                f"income (Rs.{disposable:,.0f}/month) to self-fund without debt."
            ),
            "estimated_savings": round(purchase_cost * 0.05, 0),  # avoid interest
        })

        # Alt 2: Cheaper tier (30% less)
        cheaper = purchase_cost * 0.70
        alternatives.append({
            "rank": 2,
            "strategy": "Explore cheaper alternative (30% lower tier)",
            "detail": (
                f"A Rs.{cheaper:,.0f} option would be more manageable: "
                f"affordability score {affordability_score(disposable, 12, cheaper):.2f}."
            ),
            "estimated_savings": round(purchase_cost * 0.30, 0),
        })

        # Alt 3: EMI split (24 months)
        emi_24 = purchase_cost / 24
        new_dti_24 = dti + (emi_24 / max(approx_income, 1))
        alternatives.append({
            "rank": 3,
            "strategy": "EMI over 24 months",
            "detail": (
                f"Monthly EMI: Rs.{emi_24:,.0f}. "
                f"New DTI ratio: {new_dti_24:.2f} "
                f"({'within safe range' if new_dti_24 < 0.40 else 'above safe threshold'})."
            ),
            "estimated_savings": 0,
        })

        # Alt 4: Reallocate non-essential expenses
        alt_savings = disposable * 0.30
        alternatives.append({
            "rank": 4,
            "strategy": "Reallocate 30% of discretionary spending",
            "detail": (
                f"Redirect Rs.{alt_savings:,.0f}/month from variable expenses. "
                f"New effective saving: Rs.{disposable + alt_savings:,.0f}/month. "
                f"Time to fund: {purchase_cost / max(disposable + alt_savings, 1):.0f} months."
            ),
            "estimated_savings": round(alt_savings * 12, 0),
        })

    return {
        "affordable": affordable,
        "affordability_score": aff_score,
        "impact_analysis": impact,
        "alternatives": alternatives,
        "disclaimer": DISCLAIMER_FINANCE,
    }


# ═══════════════════════════════════════════════════════════════════════
#  3. LIFESTYLE IMPACT SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

# Risk adjustment factors (simplified, evidence-based estimates)
LIFESTYLE_ADJUSTMENTS = {
    "quit_smoking": {
        "cardio_risk_reduction": 0.30,  # 30% reduction in CVD risk
        "description": "Quitting smoking reduces cardiovascular risk by ~30% within 1 year",
        "source": "AHA Guidelines",
    },
    "increase_activity": {
        "cardio_risk_reduction": 0.15,
        "bmi_change": -1.5,  # approximate BMI reduction with regular exercise
        "description": "150 min/week moderate exercise reduces CVD risk by ~15%",
        "source": "WHO Physical Activity Guidelines",
    },
    "reduce_weight_5kg": {
        "bmi_change": -1.7,  # approximate for average height
        "bp_reduction_systolic": 5,
        "description": "5kg weight loss reduces systolic BP by ~5 mmHg",
        "source": "AHA/ACC Weight Management Guidelines",
    },
    "reduce_weight_10kg": {
        "bmi_change": -3.4,
        "bp_reduction_systolic": 10,
        "cardio_risk_reduction": 0.20,
        "description": "10kg weight loss significantly reduces cardiovascular risk",
        "source": "AHA/ACC Weight Management Guidelines",
    },
    "reduce_alcohol": {
        "cardio_risk_reduction": 0.10,
        "bp_reduction_systolic": 4,
        "description": "Reducing alcohol intake lowers BP by ~4 mmHg",
        "source": "AHA Guidelines",
    },
    "improve_diet": {
        "cardio_risk_reduction": 0.10,
        "glucose_reduction": 10,
        "description": "DASH/Mediterranean diet reduces CVD risk by ~10%",
        "source": "AHA/ICMR Dietary Guidelines",
    },
    "start_smoking": {
        "cardio_risk_increase": 0.50,
        "description": "Starting smoking doubles cardiovascular disease risk",
        "source": "AHA Guidelines",
        "worsening": True,
    },
    "increase_alcohol": {
        "cardio_risk_increase": 0.15,
        "bp_increase_systolic": 5,
        "description": "Excessive alcohol increases BP and CVD risk",
        "source": "WHO Guidelines",
        "worsening": True,
    },
    "stop_exercise": {
        "cardio_risk_increase": 0.20,
        "bmi_change": 2.0,
        "description": "Sedentary lifestyle increases CVD risk by ~20%",
        "source": "WHO Physical Activity Guidelines",
        "worsening": True,
    },
}


def lifestyle_impact_simulator(
    health_profile: dict,
    habit_changes: dict,
) -> dict:
    """Simulate the health impact of lifestyle changes.

    Generates ranked alternatives (substitute, moderate, offset) if a
    habit change would worsen the risk band.

    Args:
        health_profile: Output of HealthTwin.assess().
        habit_changes:  Dict mapping change names to bool/value.
                        Keys should match LIFESTYLE_ADJUSTMENTS keys.
                        e.g., {"quit_smoking": True, "reduce_weight_5kg": True}

    Returns:
        {
            "current_risk_summary": dict,
            "projected_changes": [
                {"change": str, "description": str, "impact": str, "source": str},
                ...
            ],
            "projected_health_score": int,
            "risk_band_change": str,
            "alternatives": [...] (only if worsening),
            "disclaimer": str,
        }
    """
    current_cardio_prob = health_profile.get("cardio_risk", {}).get("probability", 0.5)
    current_bmi = health_profile.get("bmi", 25.0)
    current_bp_stage = health_profile.get("hypertension_stage", "Normal")
    current_score = health_profile.get("overall_health_score", 50)

    current_risk = {
        "cardio_probability": current_cardio_prob,
        "bmi": current_bmi,
        "hypertension_stage": current_bp_stage,
        "overall_health_score": current_score,
    }

    projected_changes = []
    net_cardio_adjustment = 0
    net_bmi_change = 0
    has_worsening = False

    for change_key, active in habit_changes.items():
        if not active:
            continue

        adjustment = LIFESTYLE_ADJUSTMENTS.get(change_key)
        if not adjustment:
            continue

        is_worsening = adjustment.get("worsening", False)
        if is_worsening:
            has_worsening = True

        # Calculate impact
        risk_change = adjustment.get("cardio_risk_reduction", 0)
        risk_increase = adjustment.get("cardio_risk_increase", 0)
        bmi_delta = adjustment.get("bmi_change", 0)

        net_cardio_adjustment += risk_increase - risk_change
        net_bmi_change += bmi_delta

        if risk_change > 0:
            impact_str = f"CVD risk reduced by ~{risk_change:.0%}"
        elif risk_increase > 0:
            impact_str = f"CVD risk increased by ~{risk_increase:.0%}"
        else:
            impact_str = "Indirect health improvement"

        if bmi_delta != 0:
            impact_str += f", BMI change: {bmi_delta:+.1f}"

        projected_changes.append({
            "change": change_key,
            "description": adjustment["description"],
            "impact": impact_str,
            "source": adjustment.get("source", "Clinical guidelines"),
            "worsening": is_worsening,
        })

    # Project new health score (simplified estimation)
    projected_cardio = max(0.01, min(0.99,
        current_cardio_prob + net_cardio_adjustment * current_cardio_prob,
    ))
    projected_bmi = max(15.0, current_bmi + net_bmi_change)

    # Approximate score change
    cardio_score_delta = (current_cardio_prob - projected_cardio) * 100 * 0.25
    bmi_score_delta = (abs(current_bmi - 22) - abs(projected_bmi - 22)) * 5 * 0.25
    projected_score = int(round(max(0, min(100,
        current_score + cardio_score_delta + bmi_score_delta,
    ))))

    # Risk band
    if projected_score > current_score + 5:
        band_change = "Improved ↑"
    elif projected_score < current_score - 5:
        band_change = "Worsened ↓"
    else:
        band_change = "Stable →"

    # Generate alternatives if worsening
    alternatives = []
    if has_worsening:
        alternatives = [
            {
                "rank": 1,
                "strategy": "Substitute",
                "detail": "Replace the harmful habit with a healthier alternative "
                          "(e.g., replace smoking with nicotine patches, replace "
                          "alcohol with non-alcoholic beverages).",
            },
            {
                "rank": 2,
                "strategy": "Moderate",
                "detail": "Reduce the intensity rather than fully adopting the "
                          "harmful change (e.g., reduce alcohol to 1 drink/week "
                          "instead of daily).",
            },
            {
                "rank": 3,
                "strategy": "Offset",
                "detail": "Counterbalance the negative impact with positive changes "
                          "(e.g., increase exercise to offset dietary changes, "
                          "improve diet to offset reduced activity).",
            },
        ]

    return {
        "current_risk_summary": current_risk,
        "projected_changes": projected_changes,
        "projected_health_score": projected_score,
        "risk_band_change": band_change,
        "alternatives": alternatives,
        "disclaimer": DISCLAIMER_HEALTH,
    }


# ======================================================================
#  STANDALONE VERIFICATION (python recommendation_engines.py)
# ======================================================================
if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  Module 9: Recommendation Engines — Test Suite")
    print("=" * 70)

    # ── Test 1: Investment Recommendation ──
    test_profiles = [
        {
            "label": "Young high-earner (age 28, stability 82)",
            "profile": {
                "dti_ratio": 0.20,
                "savings_rate": 0.40,
                "emergency_fund_months": 8.5,
                "financial_stability_score": 82,
                "disposable_income": 55000,
            },
            "age": 28,
        },
        {
            "label": "Mid-career moderate (age 45, stability 55)",
            "profile": {
                "dti_ratio": 0.38,
                "savings_rate": 0.18,
                "emergency_fund_months": 3.5,
                "financial_stability_score": 55,
                "disposable_income": 17000,
            },
            "age": 45,
        },
        {
            "label": "Financially stressed (age 35, stability 28)",
            "profile": {
                "dti_ratio": 0.55,
                "savings_rate": 0.06,
                "emergency_fund_months": 0.9,
                "financial_stability_score": 28,
                "disposable_income": 3000,
            },
            "age": 35,
        },
    ]

    for tp in test_profiles:
        print(f"\n{'─' * 70}")
        print(f"  {tp['label']}")
        print(f"{'─' * 70}")
        rec = investment_recommendation(tp["profile"], tp["age"])
        print(rec)

    # ── Test 2: Purchase Impact Simulator ──
    print(f"\n{'=' * 70}")
    print("  Purchase Impact Simulator")
    print(f"{'=' * 70}")

    fp = {
        "dti_ratio": 0.35,
        "savings_rate": 0.22,
        "emergency_fund_months": 4.0,
        "financial_stability_score": 60,
        "disposable_income": 22000,
    }

    for cost in [100_000, 500_000, 1_500_000]:
        print(f"\n{'─' * 70}")
        print(f"  Purchase cost: Rs.{cost:,.0f}")
        print(f"{'─' * 70}")
        result = purchase_impact_simulator(fp, cost)
        print(f"  Affordable: {result['affordable']}")
        print(f"  Score: {result['affordability_score']}")
        print(f"  Impact: {json.dumps(result['impact_analysis'], indent=4)}")
        if result["alternatives"]:
            print("  Alternatives:")
            for alt in result["alternatives"]:
                print(f"    [{alt['rank']}] {alt['strategy']}: {alt['detail']}")

    # ── Test 3: Lifestyle Impact Simulator ──
    print(f"\n{'=' * 70}")
    print("  Lifestyle Impact Simulator")
    print(f"{'=' * 70}")

    sample_health = {
        "bmi": 28.5,
        "obesity_class": "Overweight (Pre-obese)",
        "hypertension_stage": "Stage 1",
        "cardio_risk": {"probability": 0.45, "label": "Elevated"},
        "diabetes_risk": {"probability": 0.20, "label": "Moderate"},
        "overall_health_score": 52,
    }

    # Positive changes
    print(f"\n{'─' * 70}")
    print("  Positive changes: quit smoking + reduce weight 5kg + improve diet")
    print(f"{'─' * 70}")
    result = lifestyle_impact_simulator(sample_health, {
        "quit_smoking": True,
        "reduce_weight_5kg": True,
        "improve_diet": True,
    })
    print(f"  Current score: {result['current_risk_summary']['overall_health_score']}")
    print(f"  Projected score: {result['projected_health_score']}")
    print(f"  Risk band: {result['risk_band_change']}")
    for pc in result["projected_changes"]:
        print(f"    • {pc['change']}: {pc['impact']} ({pc['source']})")

    # Negative changes
    print(f"\n{'─' * 70}")
    print("  Negative changes: start smoking + stop exercise")
    print(f"{'─' * 70}")
    result = lifestyle_impact_simulator(sample_health, {
        "start_smoking": True,
        "stop_exercise": True,
    })
    print(f"  Current score: {result['current_risk_summary']['overall_health_score']}")
    print(f"  Projected score: {result['projected_health_score']}")
    print(f"  Risk band: {result['risk_band_change']}")
    for pc in result["projected_changes"]:
        prefix = "⚠" if pc.get("worsening") else "✓"
        print(f"    {prefix} {pc['change']}: {pc['impact']}")
    if result["alternatives"]:
        print("  Alternatives suggested:")
        for alt in result["alternatives"]:
            print(f"    [{alt['rank']}] {alt['strategy']}: {alt['detail']}")
