"""
Module 1 — formulas.py
Pure, deterministic formula/rule functions for the TwinLife AI project.

Domains covered:
  • Health  : BMI, obesity class, blood-pressure staging, Framingham CVD risk,
              diabetes indicator
  • Finance : DTI ratio, savings rate, emergency-fund months,
              financial-stability score, affordability score
  • Insurance: coverage-adequacy ratio, premium-affordability ratio,
               insurance-adequacy score

Every function is a standalone pure function (no side effects, no I/O)
so it can be unit-tested individually.
"""

from __future__ import annotations

import math

# ─────────────────────────────────────────────────────────────
#  HEALTH FORMULAS
# ─────────────────────────────────────────────────────────────

def bmi(height_cm: float, weight_kg: float) -> float:
    """Calculate Body Mass Index.

    Formula: weight_kg / (height_m ** 2)

    Args:
        height_cm: Height in centimetres (must be > 0).
        weight_kg: Weight in kilograms (must be > 0).

    Returns:
        BMI value rounded to one decimal place.

    Raises:
        ValueError: If height_cm or weight_kg is non-positive.
    """
    if height_cm <= 0:
        raise ValueError(f"height_cm must be positive, got {height_cm}")
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be positive, got {weight_kg}")

    height_m = height_cm / 100.0
    return round(weight_kg / (height_m ** 2), 1)


def who_obesity_class(bmi_value: float) -> str:
    """Map a BMI value to the WHO obesity classification.

    Classes (WHO standard):
        < 18.5           → Underweight
        18.5 – 24.9      → Normal weight
        25.0 – 29.9      → Overweight (Pre-obese)
        30.0 – 34.9      → Obese Class I
        35.0 – 39.9      → Obese Class II
        ≥ 40.0           → Obese Class III

    Args:
        bmi_value: A numeric BMI value.

    Returns:
        A human-readable WHO obesity class string.
    """
    if bmi_value < 18.5:
        return "Underweight"
    elif bmi_value < 25.0:
        return "Normal weight"
    elif bmi_value < 30.0:
        return "Overweight (Pre-obese)"
    elif bmi_value < 35.0:
        return "Obese Class I"
    elif bmi_value < 40.0:
        return "Obese Class II"
    else:
        return "Obese Class III"


def bp_stage(systolic: float, diastolic: float) -> str:
    """Classify blood pressure per ACC/AHA 2017 guidelines.

    Stages:
        Normal    : systolic < 120  AND diastolic < 80
        Elevated  : systolic 120-129 AND diastolic < 80
        Stage 1   : systolic 130-139 OR  diastolic 80-89
        Stage 2   : systolic ≥ 140  OR  diastolic ≥ 90

    The *highest applicable stage* is returned (worst-case).

    Args:
        systolic:  Systolic BP in mmHg.
        diastolic: Diastolic BP in mmHg.

    Returns:
        One of: "Normal", "Elevated", "Stage 1", "Stage 2".
    """
    # Evaluate from worst to best; return first match.
    if systolic >= 140 or diastolic >= 90:
        return "Stage 2"
    if systolic >= 130 or diastolic >= 80:
        return "Stage 1"
    if 120 <= systolic <= 129 and diastolic < 80:
        return "Elevated"
    return "Normal"


def framingham_cvd_risk(
    age: int,
    gender: str,
    cholesterol: float,
    systolic_bp: float,
    smoker: bool,
) -> float:
    """Estimate 10-year cardiovascular-disease risk (simplified Framingham).

    Uses the *simplified point-based Framingham Risk Score* published by
    Wilson et al. (1998), adapted to a compact implementation suitable for
    academic demonstration.

    Approach:
        1. Compute a linear risk score from log-transformed predictors with
           gender-specific coefficients.
        2. Convert to a 10-year probability via the Framingham survival
           function: risk = 1 − S0^exp(score − mean_score).

    Coefficients are taken from the simplified/published version commonly
    cited in textbooks (D'Agostino 2008, General CVD risk model).

    Args:
        age:          Age in years.
        gender:       "male" or "female" (case-insensitive).
        cholesterol:  Total cholesterol in mg/dL.
        systolic_bp:  Systolic blood pressure in mmHg.
        smoker:       Whether the person currently smokes.

    Returns:
        10-year CVD risk as a percentage (0-100), rounded to one decimal.

    Raises:
        ValueError: If gender is not "male" or "female", or numeric inputs
                    are non-positive.
    """
    gender = gender.strip().lower()
    if gender not in ("male", "female"):
        raise ValueError(f"gender must be 'male' or 'female', got '{gender}'")
    if age <= 0 or cholesterol <= 0 or systolic_bp <= 0:
        raise ValueError("age, cholesterol, and systolic_bp must be positive")

    ln = math.log  # natural log shortcut

    if gender == "male":
        # --- Male coefficients (D'Agostino 2008, General CVD) ---
        coeffs = {
            "ln_age":        3.06117,
            "ln_chol":       1.12370,
            "ln_sbp":        0.93263,
            "smoker":        0.65451,
        }
        mean_score = 23.9802
        baseline_survival = 0.88936
    else:
        # --- Female coefficients ---
        coeffs = {
            "ln_age":        2.32888,
            "ln_chol":       1.20904,
            "ln_sbp":        2.76157,
            "smoker":        0.52873,
        }
        mean_score = 26.1931
        baseline_survival = 0.95012

    score = (
        coeffs["ln_age"]  * ln(age)
        + coeffs["ln_chol"]  * ln(cholesterol)
        + coeffs["ln_sbp"]   * ln(systolic_bp)
        + coeffs["smoker"]   * (1.0 if smoker else 0.0)
    )

    risk = 1.0 - baseline_survival ** math.exp(score - mean_score)
    risk_pct = max(0.0, min(100.0, risk * 100.0))
    return round(risk_pct, 1)


def diabetes_indicator(glucose: float) -> str:
    """Classify diabetes risk via fasting-glucose thresholds (ADA).

    Thresholds (mg/dL):
        < 100  → Low
        100-125 → Medium (pre-diabetes range)
        ≥ 126  → High (diabetes range)

    Args:
        glucose: Fasting blood-glucose level in mg/dL.

    Returns:
        One of: "Low", "Medium", "High".
    """
    if glucose < 100:
        return "Low"
    elif glucose < 126:
        return "Medium"
    else:
        return "High"


# ─────────────────────────────────────────────────────────────
#  FINANCE FORMULAS
# ─────────────────────────────────────────────────────────────

def dti_ratio(fixed_expenses: float, emis: float, income: float) -> float:
    """Debt-to-Income ratio.

    Formula: (fixed_expenses + emis) / income

    A lower DTI indicates better financial health.  Values above 0.40 are
    typically considered risky.

    Args:
        fixed_expenses: Monthly fixed outflows (rent, utilities, etc.).
        emis:           Monthly loan EMI payments.
        income:         Monthly gross income (must be > 0).

    Returns:
        DTI ratio as a float (0.0–1.0+), rounded to two decimals.

    Raises:
        ValueError: If income is non-positive.
    """
    if income <= 0:
        raise ValueError(f"income must be positive, got {income}")
    return round((fixed_expenses + emis) / income, 2)


def savings_rate(
    income: float,
    fixed_expenses: float,
    variable_expenses: float,
) -> float:
    """Monthly savings rate.

    Formula: (income − fixed_expenses − variable_expenses) / income

    Args:
        income:            Monthly gross income (must be > 0).
        fixed_expenses:    Monthly fixed outflows.
        variable_expenses: Monthly variable outflows (food, entertainment…).

    Returns:
        Savings rate as a float (can be negative if expenses exceed income),
        rounded to two decimals.

    Raises:
        ValueError: If income is non-positive.
    """
    if income <= 0:
        raise ValueError(f"income must be positive, got {income}")
    return round((income - fixed_expenses - variable_expenses) / income, 2)


def emergency_fund_months(
    savings_balance: float,
    fixed_expenses: float,
    variable_expenses: float,
) -> float:
    """How many months of expenses the current savings balance can cover.

    Formula: savings_balance / (fixed_expenses + variable_expenses)

    Args:
        savings_balance:   Total liquid savings.
        fixed_expenses:    Monthly fixed outflows.
        variable_expenses: Monthly variable outflows.

    Returns:
        Number of months (float), rounded to one decimal.
        Returns inf if total monthly expenses are zero.
    """
    monthly_expenses = fixed_expenses + variable_expenses
    if monthly_expenses == 0:
        return float("inf")
    return round(savings_balance / monthly_expenses, 1)


def financial_stability_score(
    dti: float,
    sav_rate: float,
    emf_months: float,
) -> int:
    """Composite financial-stability score on a 0–100 scale.

    Weighted components (spec):
        30 %  inverse DTI   → (1 − clamp(dti, 0, 1)) × 100
        30 %  savings rate  → clamp(sav_rate, 0, 1) × 100
        20 %  emergency-fund adequacy → min(emf_months / 6, 1) × 100
              (6 months = perfect score for this component)
        20 %  debt-to-income component (duplicate weight per spec,
              interpreted as reinforcing the DTI signal)
              → (1 − clamp(dti, 0, 1)) × 100

    Note: The spec lists both "30% inverse DTI" and "20% debt-to-income".
    Since DTI and debt-to-income are the same metric, we implement both
    weights faithfully: 30% + 20% = 50% total weight on inverse DTI,
    giving it the dominant influence — which is financially sensible.

    Args:
        dti:        Debt-to-income ratio (output of dti_ratio()).
        sav_rate:   Savings rate (output of savings_rate()).
        emf_months: Emergency-fund months (output of emergency_fund_months()).

    Returns:
        Integer score in [0, 100].
    """
    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    inverse_dti_score = (1.0 - clamp01(dti)) * 100.0
    savings_score     = clamp01(sav_rate) * 100.0
    # 6 months of expenses = fully adequate emergency fund
    emf_score         = min(emf_months / 6.0, 1.0) * 100.0
    dti_component     = inverse_dti_score  # same formula, separate weight

    weighted = (
        0.30 * inverse_dti_score
        + 0.30 * savings_score
        + 0.20 * emf_score
        + 0.20 * dti_component
    )
    return int(round(weighted))


def affordability_score(
    disposable_income: float,
    months_available: float,
    cost: float,
) -> float:
    """How affordable a one-time cost is given disposable income and time.

    Formula: (disposable_income × months_available) / cost

    A score ≥ 1.0 means the cost is fully coverable within the timeframe.
    Scores < 1.0 indicate a shortfall.

    Args:
        disposable_income: Monthly disposable income after expenses.
        months_available:  Months available to save for the cost.
        cost:              Total cost of the item/treatment.

    Returns:
        Affordability score as a float, rounded to two decimals.
        Returns inf if cost is zero.

    Raises:
        ValueError: If cost is negative.
    """
    if cost < 0:
        raise ValueError(f"cost must be non-negative, got {cost}")
    if cost == 0:
        return float("inf")
    return round((disposable_income * months_available) / cost, 2)


# ─────────────────────────────────────────────────────────────
#  INSURANCE FORMULAS
# ─────────────────────────────────────────────────────────────

def coverage_adequacy_ratio(
    sum_insured: float,
    estimated_future_cost: float,
) -> float:
    """How well insurance covers an estimated future medical cost.

    Formula: sum_insured / estimated_future_cost

    A ratio ≥ 1.0 means full coverage; < 1.0 indicates a gap.

    Args:
        sum_insured:          Total sum insured under the policy.
        estimated_future_cost: Projected medical costs.

    Returns:
        Coverage ratio as a float, rounded to two decimals.
        Returns inf if estimated_future_cost is zero.
    """
    if estimated_future_cost == 0:
        return float("inf")
    return round(sum_insured / estimated_future_cost, 2)


def premium_affordability_ratio(
    annual_premium: float,
    annual_income: float,
) -> float:
    """What fraction of annual income goes toward insurance premiums.

    Formula: annual_premium / annual_income

    A lower ratio is better.  Common guideline: keep below 0.05–0.10.

    Args:
        annual_premium: Annual premium amount.
        annual_income:  Annual gross income (must be > 0).

    Returns:
        Premium-to-income ratio, rounded to three decimals.

    Raises:
        ValueError: If annual_income is non-positive.
    """
    if annual_income <= 0:
        raise ValueError(f"annual_income must be positive, got {annual_income}")
    return round(annual_premium / annual_income, 3)


def insurance_adequacy_score(
    coverage_ratio: float,
    premium_ratio: float,
    rider_gaps: int,
) -> int:
    """Composite insurance-adequacy score on a 0–100 scale.

    Components:
        40 %  coverage adequacy  → min(coverage_ratio, 1.5) / 1.5 × 100
              (ratio of 1.5 or above = perfect; partial credit below)
        35 %  premium affordability → (1 − clamp(premium_ratio, 0, 0.20)) / 0.20 × 100
              (0% of income = perfect; 20%+ = zero)
        25 %  rider-gap penalty  → max(0, 100 − rider_gaps × 25)
              (each missing rider deducts 25 points from this component)

    Args:
        coverage_ratio: Output of coverage_adequacy_ratio().
        premium_ratio:  Output of premium_affordability_ratio().
        rider_gaps:     Number of missing/recommended insurance riders.

    Returns:
        Integer score in [0, 100].
    """
    # Coverage component: cap at 1.5 for full marks
    cov_score = min(coverage_ratio, 1.5) / 1.5 * 100.0

    # Premium component: 0% premium = 100, ≥20% = 0
    clamped_premium = max(0.0, min(premium_ratio, 0.20))
    prem_score = (1.0 - clamped_premium / 0.20) * 100.0

    # Rider-gap component: each gap costs 25 points
    rider_score = max(0.0, 100.0 - rider_gaps * 25.0)

    weighted = 0.40 * cov_score + 0.35 * prem_score + 0.25 * rider_score
    return int(round(max(0.0, min(100.0, weighted))))
