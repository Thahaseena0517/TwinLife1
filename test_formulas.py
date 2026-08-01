"""
Tests for Module 1 — formulas.py

Covers every public function with:
  • At least one normal-case test
  • Boundary / edge-case tests where applicable
  • Error-handling tests for invalid inputs

Run with:  python -m pytest test_formulas.py -v
"""

import math
import pytest

from formulas import (
    bmi,
    who_obesity_class,
    bp_stage,
    framingham_cvd_risk,
    diabetes_indicator,
    dti_ratio,
    savings_rate,
    emergency_fund_months,
    financial_stability_score,
    affordability_score,
    coverage_adequacy_ratio,
    premium_affordability_ratio,
    insurance_adequacy_score,
)


# ─────────────────────────────────────────────────────────────
#  HEALTH FORMULA TESTS
# ─────────────────────────────────────────────────────────────

class TestBMI:
    """Tests for bmi()."""

    def test_normal_case(self):
        # 170 cm, 70 kg → 70 / 1.7^2 ≈ 24.2
        assert bmi(170, 70) == 24.2

    def test_tall_heavy(self):
        # 190 cm, 100 kg → 100 / 3.61 ≈ 27.7
        assert bmi(190, 100) == 27.7

    def test_short_light(self):
        # 150 cm, 45 kg → 45 / 2.25 = 20.0
        assert bmi(150, 45) == 20.0

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="height_cm must be positive"):
            bmi(0, 70)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight_kg must be positive"):
            bmi(170, -5)


class TestWHOObesityClass:
    """Tests for who_obesity_class()."""

    def test_underweight(self):
        assert who_obesity_class(17.0) == "Underweight"

    def test_normal_weight(self):
        assert who_obesity_class(22.5) == "Normal weight"

    def test_overweight(self):
        assert who_obesity_class(27.3) == "Overweight (Pre-obese)"

    def test_obese_class_i(self):
        assert who_obesity_class(32.0) == "Obese Class I"

    def test_obese_class_ii(self):
        assert who_obesity_class(37.5) == "Obese Class II"

    def test_obese_class_iii(self):
        assert who_obesity_class(42.0) == "Obese Class III"

    def test_boundary_18_5(self):
        assert who_obesity_class(18.5) == "Normal weight"

    def test_boundary_25(self):
        assert who_obesity_class(25.0) == "Overweight (Pre-obese)"

    def test_boundary_30(self):
        assert who_obesity_class(30.0) == "Obese Class I"

    def test_boundary_40(self):
        assert who_obesity_class(40.0) == "Obese Class III"


class TestBPStage:
    """Tests for bp_stage() — ACC/AHA 2017 staging."""

    def test_normal(self):
        assert bp_stage(115, 75) == "Normal"

    def test_elevated(self):
        assert bp_stage(125, 78) == "Elevated"

    def test_stage_1_by_systolic(self):
        assert bp_stage(135, 75) == "Stage 1"

    def test_stage_1_by_diastolic(self):
        assert bp_stage(118, 85) == "Stage 1"

    def test_stage_2_by_systolic(self):
        assert bp_stage(145, 78) == "Stage 2"

    def test_stage_2_by_diastolic(self):
        assert bp_stage(115, 95) == "Stage 2"

    def test_stage_2_both_high(self):
        assert bp_stage(160, 100) == "Stage 2"

    def test_boundary_120_80(self):
        # 120/80 → systolic is in Elevated range but diastolic is 80 → Stage 1
        # diastolic 80 triggers Stage 1
        assert bp_stage(120, 80) == "Stage 1"

    def test_boundary_120_79(self):
        assert bp_stage(120, 79) == "Elevated"


class TestFraminghamCVDRisk:
    """Tests for framingham_cvd_risk()."""

    def test_low_risk_young_female(self):
        risk = framingham_cvd_risk(30, "female", 180, 110, False)
        assert 0 <= risk <= 100
        # Simplified model overestimates for younger ages (known limitation)
        assert risk < 20.0

    def test_high_risk_older_male_smoker(self):
        risk = framingham_cvd_risk(65, "male", 280, 160, True)
        assert risk > 10.0  # elevated risk for older male smoker with bad lipids

    def test_moderate_risk(self):
        risk = framingham_cvd_risk(50, "male", 220, 135, False)
        assert 1.0 < risk < 30.0  # moderate range with simplified coefficients

    def test_gender_case_insensitive(self):
        r1 = framingham_cvd_risk(45, "Male", 200, 120, False)
        r2 = framingham_cvd_risk(45, "male", 200, 120, False)
        assert r1 == r2

    def test_invalid_gender_raises(self):
        with pytest.raises(ValueError, match="gender must be"):
            framingham_cvd_risk(50, "other", 200, 120, False)

    def test_non_positive_age_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            framingham_cvd_risk(0, "male", 200, 120, False)

    def test_smoker_increases_risk(self):
        non_smoker = framingham_cvd_risk(50, "male", 220, 130, False)
        smoker = framingham_cvd_risk(50, "male", 220, 130, True)
        assert smoker > non_smoker


class TestDiabetesIndicator:
    """Tests for diabetes_indicator()."""

    def test_low(self):
        assert diabetes_indicator(85) == "Low"

    def test_medium(self):
        assert diabetes_indicator(110) == "Medium"

    def test_high(self):
        assert diabetes_indicator(140) == "High"

    def test_boundary_100(self):
        assert diabetes_indicator(100) == "Medium"

    def test_boundary_126(self):
        assert diabetes_indicator(126) == "High"

    def test_boundary_99(self):
        assert diabetes_indicator(99) == "Low"


# ─────────────────────────────────────────────────────────────
#  FINANCE FORMULA TESTS
# ─────────────────────────────────────────────────────────────

class TestDTIRatio:
    """Tests for dti_ratio()."""

    def test_basic(self):
        # (20000 + 10000) / 100000 = 0.30
        assert dti_ratio(20000, 10000, 100000) == 0.30

    def test_zero_emi(self):
        assert dti_ratio(15000, 0, 50000) == 0.30

    def test_high_dti(self):
        assert dti_ratio(30000, 20000, 60000) == 0.83

    def test_zero_income_raises(self):
        with pytest.raises(ValueError, match="income must be positive"):
            dti_ratio(10000, 5000, 0)


class TestSavingsRate:
    """Tests for savings_rate()."""

    def test_positive_savings(self):
        # (100000 - 30000 - 20000) / 100000 = 0.50
        assert savings_rate(100000, 30000, 20000) == 0.50

    def test_negative_savings(self):
        # Expenses exceed income
        result = savings_rate(50000, 30000, 30000)
        assert result < 0

    def test_all_saved(self):
        assert savings_rate(100000, 0, 0) == 1.0

    def test_zero_income_raises(self):
        with pytest.raises(ValueError):
            savings_rate(0, 10000, 5000)


class TestEmergencyFundMonths:
    """Tests for emergency_fund_months()."""

    def test_six_months(self):
        # 300000 / (30000 + 20000) = 6.0
        assert emergency_fund_months(300000, 30000, 20000) == 6.0

    def test_partial(self):
        # 100000 / 50000 = 2.0
        assert emergency_fund_months(100000, 30000, 20000) == 2.0

    def test_zero_savings(self):
        assert emergency_fund_months(0, 20000, 10000) == 0.0

    def test_zero_expenses(self):
        result = emergency_fund_months(50000, 0, 0)
        assert result == float("inf")


class TestFinancialStabilityScore:
    """Tests for financial_stability_score()."""

    def test_perfect_score(self):
        # DTI=0, savings_rate=1.0, emergency_fund=6+ months → 100
        score = financial_stability_score(0.0, 1.0, 6.0)
        assert score == 100

    def test_worst_score(self):
        # DTI=1.0, savings_rate=0, emergency_fund=0 → 0
        score = financial_stability_score(1.0, 0.0, 0.0)
        assert score == 0

    def test_moderate(self):
        # DTI=0.30, savings_rate=0.40, emergency_fund=3 months
        score = financial_stability_score(0.30, 0.40, 3.0)
        assert 40 <= score <= 80

    def test_returns_int(self):
        score = financial_stability_score(0.25, 0.35, 4.0)
        assert isinstance(score, int)

    def test_clamped_above_one(self):
        # DTI > 1.0 should clamp, not go negative
        score = financial_stability_score(1.5, 0.0, 0.0)
        assert score == 0


class TestAffordabilityScore:
    """Tests for affordability_score()."""

    def test_fully_affordable(self):
        # 50000/mo × 12 months = 600000 > 500000
        score = affordability_score(50000, 12, 500000)
        assert score == 1.20

    def test_not_affordable(self):
        score = affordability_score(10000, 6, 500000)
        assert score < 1.0

    def test_zero_cost(self):
        assert affordability_score(50000, 12, 0) == float("inf")

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost must be non-negative"):
            affordability_score(50000, 12, -100)


# ─────────────────────────────────────────────────────────────
#  INSURANCE FORMULA TESTS
# ─────────────────────────────────────────────────────────────

class TestCoverageAdequacyRatio:
    """Tests for coverage_adequacy_ratio()."""

    def test_full_coverage(self):
        assert coverage_adequacy_ratio(500000, 500000) == 1.0

    def test_over_insured(self):
        assert coverage_adequacy_ratio(1000000, 500000) == 2.0

    def test_under_insured(self):
        assert coverage_adequacy_ratio(250000, 500000) == 0.5

    def test_zero_future_cost(self):
        assert coverage_adequacy_ratio(500000, 0) == float("inf")


class TestPremiumAffordabilityRatio:
    """Tests for premium_affordability_ratio()."""

    def test_low_ratio(self):
        # 30000 / 1200000 = 0.025
        assert premium_affordability_ratio(30000, 1200000) == 0.025

    def test_high_ratio(self):
        # 120000 / 600000 = 0.20
        assert premium_affordability_ratio(120000, 600000) == 0.2

    def test_zero_income_raises(self):
        with pytest.raises(ValueError, match="annual_income must be positive"):
            premium_affordability_ratio(30000, 0)


class TestInsuranceAdequacyScore:
    """Tests for insurance_adequacy_score()."""

    def test_perfect(self):
        # coverage_ratio=1.5, premium_ratio=0.0, rider_gaps=0 → 100
        score = insurance_adequacy_score(1.5, 0.0, 0)
        assert score == 100

    def test_worst(self):
        # coverage=0, premium=0.20 (max), rider_gaps=4+ → 0
        score = insurance_adequacy_score(0.0, 0.20, 4)
        assert score == 0

    def test_partial(self):
        # coverage_ratio=1.0, premium_ratio=0.05, rider_gaps=1
        score = insurance_adequacy_score(1.0, 0.05, 1)
        assert 40 <= score <= 90

    def test_returns_int(self):
        score = insurance_adequacy_score(1.2, 0.03, 2)
        assert isinstance(score, int)

    def test_high_rider_gaps_floor(self):
        # Many rider gaps shouldn't make score negative
        score = insurance_adequacy_score(1.5, 0.0, 10)
        assert score >= 0
