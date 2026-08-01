"""
Test script for Module 4 -- combined_simulator.py (Gemini API version)

Runs the autonomous tool-use agent across 5 varied treatment-plan
scenarios and prints the tool-call trace + final recommendation for each.

Setup:
  1. Get a free API key from https://aistudio.google.com/apikey
  2. Set the environment variable before running:
     PowerShell:  $env:GEMINI_API_KEY = "your-key-here"
     CMD:         set GEMINI_API_KEY=your-key-here

Usage:
    python test_combined_simulator.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

# Check API key early
if not os.environ.get("GEMINI_API_KEY"):
    print("=" * 60)
    print("  ERROR: GEMINI_API_KEY not set!")
    print("=" * 60)
    print()
    print("  1. Get a FREE key from: https://aistudio.google.com/apikey")
    print()
    print("  2. Then set it:")
    print('     PowerShell:  $env:GEMINI_API_KEY = "your-key-here"')
    print("     CMD:         set GEMINI_API_KEY=your-key-here")
    print()
    print("  3. Run again:   python test_combined_simulator.py")
    sys.exit(1)

from health_twin import HealthTwin
from finance_twin import FinanceTwin
from insurance_twin import InsuranceTwin
from combined_simulator import CombinedSimulator


# ---------------------------------------------------------------------------
#  TEST SCENARIOS
# ---------------------------------------------------------------------------

SCENARIOS = [
    # --- 1. Affordable, low-risk ---
    {
        "name": "Low-cost checkup for healthy person",
        "health_inputs": {
            "age": 30, "gender": "female",
            "height_cm": 165, "weight_kg": 60,
            "ap_hi": 115, "ap_lo": 75,
            "cholesterol": 1, "gluc": 1,
            "smoke": 0, "alco": 0, "active": 1,
            "hypertension": 0, "heart_disease": 0,
            "smoking_history": "never",
            "HbA1c_level": 5.0, "blood_glucose_level": 85,
        },
        "finance_inputs": {
            "income": 120000, "fixed_expenses": 30000,
            "variable_expenses": 20000, "emis": 15000,
            "savings_balance": 500000,
        },
        "insurance_inputs": {
            "sum_insured": 1000000, "annual_premium": 20000,
            "annual_income": 1440000, "existing_riders": [],
        },
        "cost": 50000,
        "condition": "cardio",
    },
    # --- 2. Moderate cost, well-insured ---
    {
        "name": "Moderate cardiac procedure, good insurance",
        "health_inputs": {
            "age": 50, "gender": "male",
            "height_cm": 175, "weight_kg": 85,
            "ap_hi": 140, "ap_lo": 90,
            "cholesterol": 2, "gluc": 1,
            "smoke": 0, "alco": 0, "active": 1,
            "hypertension": 1, "heart_disease": 0,
            "smoking_history": "former",
            "HbA1c_level": 5.8, "blood_glucose_level": 105,
        },
        "finance_inputs": {
            "income": 200000, "fixed_expenses": 50000,
            "variable_expenses": 30000, "emis": 25000,
            "savings_balance": 800000,
        },
        "insurance_inputs": {
            "sum_insured": 1500000, "annual_premium": 35000,
            "annual_income": 2400000,
            "existing_riders": ["Critical Illness Rider"],
        },
        "cost": 500000,
        "condition": "cardio",
    },
    # --- 3. Expensive surgery, under-insured, tight budget ---
    {
        "name": "Expensive cardiac surgery, poor coverage",
        "health_inputs": {
            "age": 60, "gender": "male",
            "height_cm": 170, "weight_kg": 90,
            "ap_hi": 160, "ap_lo": 100,
            "cholesterol": 3, "gluc": 2,
            "smoke": 1, "alco": 0, "active": 0,
            "hypertension": 1, "heart_disease": 1,
            "smoking_history": "current",
            "HbA1c_level": 7.5, "blood_glucose_level": 170,
        },
        "finance_inputs": {
            "income": 40000, "fixed_expenses": 18000,
            "variable_expenses": 12000, "emis": 8000,
            "savings_balance": 50000,
        },
        "insurance_inputs": {
            "sum_insured": 200000, "annual_premium": 15000,
            "annual_income": 480000, "existing_riders": [],
        },
        "cost": 1200000,
        "condition": "cardio",
    },
    # --- 4. Diabetes management, moderate budget ---
    {
        "name": "Diabetes treatment plan, moderate finances",
        "health_inputs": {
            "age": 45, "gender": "female",
            "height_cm": 160, "weight_kg": 75,
            "ap_hi": 130, "ap_lo": 85,
            "cholesterol": 2, "gluc": 2,
            "smoke": 0, "alco": 0, "active": 1,
            "hypertension": 0, "heart_disease": 0,
            "smoking_history": "never",
            "HbA1c_level": 7.0, "blood_glucose_level": 145,
        },
        "finance_inputs": {
            "income": 80000, "fixed_expenses": 25000,
            "variable_expenses": 15000, "emis": 10000,
            "savings_balance": 200000,
        },
        "insurance_inputs": {
            "sum_insured": 500000, "annual_premium": 18000,
            "annual_income": 960000,
            "existing_riders": ["Personal Accident Rider"],
        },
        "cost": 400000,
        "condition": "diabetes",
    },
    # --- 5. Hypertension treatment, fully affordable high earner ---
    {
        "name": "Hypertension plan, high earner",
        "health_inputs": {
            "age": 40, "gender": "male",
            "height_cm": 178, "weight_kg": 80,
            "ap_hi": 145, "ap_lo": 92,
            "cholesterol": 1, "gluc": 1,
            "smoke": 0, "alco": 1, "active": 1,
            "hypertension": 1, "heart_disease": 0,
            "smoking_history": "never",
            "HbA1c_level": 5.2, "blood_glucose_level": 90,
        },
        "finance_inputs": {
            "income": 300000, "fixed_expenses": 60000,
            "variable_expenses": 40000, "emis": 30000,
            "savings_balance": 1500000,
        },
        "insurance_inputs": {
            "sum_insured": 2000000, "annual_premium": 30000,
            "annual_income": 3600000,
            "existing_riders": [
                "Critical Illness Rider", "Hospital Cash Rider",
                "Personal Accident Rider",
            ],
        },
        "cost": 200000,
        "condition": "hypertension",
    },
]


def run_scenario(scenario: dict, index: int) -> dict:
    """Run a single test scenario through the combined simulator."""

    print(f"\n{'='*70}")
    print(f"  SCENARIO {index}: {scenario['name']}")
    print(f"  Cost: Rs.{scenario['cost']:,.0f} | Condition: {scenario['condition']}")
    print(f"{'='*70}")

    # --- Set up twins ---
    health_twin = HealthTwin()
    health_profile = health_twin.assess(scenario["health_inputs"])

    finance_twin = FinanceTwin()
    finance_profile = finance_twin.assess(scenario["finance_inputs"])

    insurance_twin = InsuranceTwin()

    # --- Build simulator ---
    sim = CombinedSimulator(
        health_profile=health_profile,
        finance_twin=finance_twin,
        insurance_twin=insurance_twin,
        insurance_inputs=scenario["insurance_inputs"],
    )

    # --- Run agent ---
    start = time.time()
    result = sim.evaluate_treatment(
        cost=scenario["cost"],
        condition=scenario["condition"],
    )
    elapsed = time.time() - start

    # --- Print tool trace ---
    print(f"\n  Tool-call trace ({len(result['tool_trace'])} calls, "
          f"{result['iterations']} iteration(s), {elapsed:.1f}s):")
    print(f"  {'-'*60}")
    for i, call in enumerate(result["tool_trace"], 1):
        print(f"  [{i}] {call['tool']}({json.dumps(call['input'])})")
        print(f"      -> {json.dumps(call['result'])}")

    # --- Print recommendation ---
    print(f"\n  RECOMMENDATION:")
    print(f"  {'-'*60}")
    for line in result["recommendation"].split("\n"):
        print(f"  {line}")

    return result


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  COMBINED SIMULATOR -- AUTONOMOUS AGENT TEST (Gemini API)")
    print("  Testing 5 varied treatment-plan scenarios")
    print("=" * 70)

    results = []
    for i, scenario in enumerate(SCENARIOS, 1):
        try:
            result = run_scenario(scenario, i)
            results.append({
                "scenario": scenario["name"],
                "cost": scenario["cost"],
                "condition": scenario["condition"],
                "num_tool_calls": len(result["tool_trace"]),
                "iterations": result["iterations"],
                "tools_used": [c["tool"] for c in result["tool_trace"]],
            })
        except Exception as e:
            print(f"\n  [SKIPPED] {scenario['name']}: {str(e)[:60]}")
            results.append({
                "scenario": scenario["name"] + " [SKIPPED]",
                "cost": scenario["cost"],
                "condition": scenario["condition"],
                "num_tool_calls": 0,
                "iterations": 0,
                "tools_used": [],
            })
        # Delay between scenarios to respect free-tier rate limits
        if i < len(SCENARIOS):
            print("\n    [Waiting 15s before next scenario...]")
            time.sleep(15)

    # --- Summary table ---
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'#':<3} {'Scenario':<45} {'Cost':>12} {'Tools':>6}")
    print(f"  {'-'*3} {'-'*45} {'-'*12} {'-'*6}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<3} {r['scenario']:<45} Rs.{r['cost']:>9,} {r['num_tool_calls']:>5}")
    print(f"\n  Tool sequences:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {' -> '.join(r['tools_used'])}")
