"""
Module 3A -- health_twin.py
HealthTwin: loads BOTH the cardio and diabetes ML models, runs predictions
with SHAP explanations, and combines with rule-based formulas from Module 1.

Usage:
    twin = HealthTwin()
    profile = twin.assess(inputs_dict)
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("shap not installed -- SHAP explanations will be unavailable")

from formulas import (
    bmi as calc_bmi,
    who_obesity_class,
    bp_stage,
    framingham_cvd_risk,
    diabetes_indicator,
)

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
#  PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARDIO_MODEL_PATH = os.path.join(BASE_DIR, "models", "cardio_model.pkl")
DIABETES_MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.pkl")
DIABETES_FEATURES_PATH = os.path.join(BASE_DIR, "models", "diabetes_feature_cols.pkl")

# Cardio model feature order (must match train_cardio_model.py)
CARDIO_FEATURES = [
    "age_years", "gender", "bmi",
    "ap_hi", "ap_lo",
    "cholesterol", "gluc",
    "smoke", "alco", "active",
]


class HealthTwin:
    """Digital twin for health assessment.

    Combines two trained ML models (cardiovascular disease, diabetes) with
    rule-based formulas (BMI, BP staging, Framingham risk, diabetes indicator)
    to produce a unified health profile with SHAP-based explainability.

    Attributes:
        cardio_model:   Trained RandomForestClassifier for CVD prediction.
        diabetes_model: Trained RandomForestClassifier for diabetes prediction.
        diabetes_feature_cols: Feature column order for the diabetes model.
    """

    def __init__(self) -> None:
        """Load both pre-trained models from disk."""
        self.cardio_model = joblib.load(CARDIO_MODEL_PATH)
        self.diabetes_model = joblib.load(DIABETES_MODEL_PATH)
        self.diabetes_feature_cols = joblib.load(DIABETES_FEATURES_PATH)

    # ------------------------------------------------------------------
    #  PUBLIC API
    # ------------------------------------------------------------------
    def assess(self, inputs: dict) -> dict:
        """Run a full health assessment.

        Args:
            inputs: Dict with keys:
                Health vitals:
                  age (int)        - age in years
                  gender (str)     - "male" or "female" (for formulas);
                                     also mapped to numeric for models
                  height_cm (float)
                  weight_kg (float)
                  ap_hi (float)    - systolic BP in mmHg
                  ap_lo (float)    - diastolic BP in mmHg
                  cholesterol (int)- 1=normal, 2=above normal, 3=well above
                  gluc (int)       - 1=normal, 2=above normal, 3=well above
                  smoke (int)      - 0/1
                  alco (int)       - 0/1
                  active (int)     - 0/1
                Diabetes-specific (optional -- used by diabetes model):
                  hypertension (int)       - 0/1
                  heart_disease (int)      - 0/1
                  smoking_history (str)    - one of: never, No Info, current,
                                             former, ever, not current
                  HbA1c_level (float)
                  blood_glucose_level (float)

        Returns:
            health_profile dict matching the spec shape:
            {
                "bmi": float,
                "obesity_class": str,
                "hypertension_stage": str,
                "cardio_risk": {"probability": float, "label": str},
                "diabetes_risk": {"probability": float, "label": str},
                "diabetes_indicator": str,
                "overall_health_score": int,   # 0-100
                "shap_top_features": [
                    {"feature": str, "impact": float, "model": str}, ...
                ],
            }
        """
        # --- Rule-based formulas ---
        bmi_val = calc_bmi(inputs["height_cm"], inputs["weight_kg"])
        obesity = who_obesity_class(bmi_val)
        bp = bp_stage(inputs["ap_hi"], inputs["ap_lo"])
        framingham = framingham_cvd_risk(
            age=inputs["age"],
            gender=inputs["gender"],
            cholesterol=self._cholesterol_to_mgdl(inputs["cholesterol"]),
            systolic_bp=inputs["ap_hi"],
            smoker=bool(inputs["smoke"]),
        )

        # Diabetes indicator from glucose (if provided)
        glucose = inputs.get("blood_glucose_level", 90)
        diab_indicator = diabetes_indicator(glucose)

        # --- Cardio model prediction ---
        cardio_prob, cardio_shap = self._predict_cardio(inputs, bmi_val)
        cardio_label = self._risk_label(cardio_prob)

        # --- Diabetes model prediction ---
        diabetes_prob, diabetes_shap = self._predict_diabetes(inputs, bmi_val)
        diabetes_label = self._risk_label(diabetes_prob)

        # --- Merge SHAP features (top 5 from each model, deduplicated) ---
        shap_top = self._merge_shap(cardio_shap, diabetes_shap)

        # --- Overall health score (0-100, higher = healthier) ---
        overall = self._compute_overall_score(
            bmi_val, bp, cardio_prob, diabetes_prob, framingham,
        )

        return {
            "bmi": bmi_val,
            "obesity_class": obesity,
            "hypertension_stage": bp,
            "cardio_risk": {
                "probability": round(cardio_prob, 3),
                "label": cardio_label,
            },
            "diabetes_risk": {
                "probability": round(diabetes_prob, 3),
                "label": diabetes_label,
            },
            "diabetes_indicator": diab_indicator,
            "framingham_10yr_risk": round(framingham, 1),
            "overall_health_score": overall,
            "shap_top_features": shap_top,
        }

    # ------------------------------------------------------------------
    #  CARDIO MODEL HELPERS
    # ------------------------------------------------------------------
    def _predict_cardio(
        self, inputs: dict, bmi_val: float,
    ) -> tuple[float, list[dict]]:
        """Run cardio model prediction + SHAP explanation."""
        gender_numeric = 2 if inputs["gender"].lower() == "female" else 1
        row = pd.DataFrame([{
            "age_years":   inputs["age"],
            "gender":      gender_numeric,
            "bmi":         bmi_val,
            "ap_hi":       inputs["ap_hi"],
            "ap_lo":       inputs["ap_lo"],
            "cholesterol": inputs["cholesterol"],
            "gluc":        inputs["gluc"],
            "smoke":       inputs["smoke"],
            "alco":        inputs["alco"],
            "active":      inputs["active"],
        }])[CARDIO_FEATURES]

        prob = float(self.cardio_model.predict_proba(row)[0, 1])

        # SHAP
        shap_features = []
        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(self.cardio_model)
            sv = explainer.shap_values(row)
            # sv may be a list [class0, class1] or ndarray
            if isinstance(sv, list):
                vals = sv[1][0]  # class-1 shap values
            else:
                vals = sv[0] if sv.ndim == 1 else sv[0, :, 1] if sv.ndim == 3 else sv[0]
            for feat, val in sorted(
                zip(CARDIO_FEATURES, vals), key=lambda x: abs(x[1]), reverse=True,
            )[:5]:
                shap_features.append({
                    "feature": feat, "impact": round(float(val), 4), "model": "cardio",
                })
        return prob, shap_features

    # ------------------------------------------------------------------
    #  DIABETES MODEL HELPERS
    # ------------------------------------------------------------------
    def _predict_diabetes(
        self, inputs: dict, bmi_val: float,
    ) -> tuple[float, list[dict]]:
        """Run diabetes model prediction + SHAP explanation."""
        # Build the raw row, then one-hot encode to match training
        gender_str = inputs["gender"].strip().capitalize()  # Female / Male
        if gender_str not in ("Female", "Male", "Other"):
            gender_str = "Male" if gender_str == "male" else gender_str
            gender_str = gender_str.capitalize()

        smoking = inputs.get("smoking_history", "No Info")

        raw = {
            "age":                inputs["age"],
            "hypertension":       inputs.get("hypertension", 0),
            "heart_disease":      inputs.get("heart_disease", 0),
            "bmi":                bmi_val,
            "HbA1c_level":        inputs.get("HbA1c_level", 5.5),
            "blood_glucose_level": inputs.get("blood_glucose_level", 90),
        }
        # One-hot: gender
        for cat in ("Female", "Male", "Other"):
            raw[f"gender_{cat}"] = 1 if gender_str == cat else 0
        # One-hot: smoking_history
        for cat in ("No Info", "current", "ever", "former", "never", "not current"):
            raw[f"smoking_{cat}"] = 1 if smoking == cat else 0

        row = pd.DataFrame([raw])[self.diabetes_feature_cols]

        prob = float(self.diabetes_model.predict_proba(row)[0, 1])

        # SHAP
        shap_features = []
        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(self.diabetes_model)
            sv = explainer.shap_values(row)
            if isinstance(sv, list):
                vals = sv[1][0]
            else:
                vals = sv[0] if sv.ndim == 1 else sv[0, :, 1] if sv.ndim == 3 else sv[0]
            for feat, val in sorted(
                zip(self.diabetes_feature_cols, vals),
                key=lambda x: abs(x[1]), reverse=True,
            )[:5]:
                shap_features.append({
                    "feature": feat, "impact": round(float(val), 4), "model": "diabetes",
                })
        return prob, shap_features

    # ------------------------------------------------------------------
    #  SCORING HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _cholesterol_to_mgdl(level: int) -> float:
        """Map categorical cholesterol (1/2/3) to approximate mg/dL for
        the Framingham formula.  1=normal(~190), 2=above(~240), 3=high(~300)."""
        return {1: 190.0, 2: 240.0, 3: 300.0}.get(level, 200.0)

    @staticmethod
    def _risk_label(probability: float) -> str:
        """Convert a 0-1 probability to a human-readable risk label."""
        if probability < 0.20:
            return "Low"
        elif probability < 0.40:
            return "Moderate"
        elif probability < 0.60:
            return "Elevated"
        elif probability < 0.80:
            return "High"
        else:
            return "Very High"

    @staticmethod
    def _compute_overall_score(
        bmi_val: float,
        bp_stage_str: str,
        cardio_prob: float,
        diabetes_prob: float,
        framingham: float,
    ) -> int:
        """Weighted composite health score (0-100, higher = healthier).

        Components (weights):
          25%  BMI component  -- distance from ideal BMI 22
          25%  BP component   -- Normal=100, Elevated=70, S1=40, S2=10
          25%  Cardio risk    -- (1 - cardio_prob) * 100
          15%  Diabetes risk  -- (1 - diabetes_prob) * 100
          10%  Framingham     -- (1 - clamp(framingham/50, 0, 1)) * 100
        """
        # BMI: ideal=22, penalty grows with distance
        bmi_dist = abs(bmi_val - 22.0)
        bmi_score = max(0.0, 100.0 - bmi_dist * 5.0)  # 20 units away = 0

        bp_map = {"Normal": 100, "Elevated": 70, "Stage 1": 40, "Stage 2": 10}
        bp_score = bp_map.get(bp_stage_str, 50)

        cardio_score = (1.0 - cardio_prob) * 100.0
        diabetes_score = (1.0 - diabetes_prob) * 100.0
        fram_score = (1.0 - min(framingham / 50.0, 1.0)) * 100.0

        weighted = (
            0.25 * bmi_score
            + 0.25 * bp_score
            + 0.25 * cardio_score
            + 0.15 * diabetes_score
            + 0.10 * fram_score
        )
        return int(round(max(0, min(100, weighted))))

    @staticmethod
    def _merge_shap(
        cardio_shap: list[dict], diabetes_shap: list[dict],
    ) -> list[dict]:
        """Merge and sort top SHAP features from both models."""
        combined = cardio_shap + diabetes_shap
        combined.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return combined[:10]  # top 10 across both models


# ======================================================================
#  STANDALONE VERIFICATION (python health_twin.py)
# ======================================================================
if __name__ == "__main__":
    import json

    twin = HealthTwin()

    samples = [
        {
            "label": "Healthy 30-year-old female",
            "inputs": {
                "age": 30, "gender": "female",
                "height_cm": 165, "weight_kg": 60,
                "ap_hi": 115, "ap_lo": 75,
                "cholesterol": 1, "gluc": 1,
                "smoke": 0, "alco": 0, "active": 1,
                "hypertension": 0, "heart_disease": 0,
                "smoking_history": "never",
                "HbA1c_level": 5.0, "blood_glucose_level": 85,
            },
        },
        {
            "label": "At-risk 55-year-old male smoker",
            "inputs": {
                "age": 55, "gender": "male",
                "height_cm": 175, "weight_kg": 95,
                "ap_hi": 150, "ap_lo": 95,
                "cholesterol": 3, "gluc": 2,
                "smoke": 1, "alco": 1, "active": 0,
                "hypertension": 1, "heart_disease": 0,
                "smoking_history": "current",
                "HbA1c_level": 7.2, "blood_glucose_level": 160,
            },
        },
        {
            "label": "Moderate-risk 45-year-old male",
            "inputs": {
                "age": 45, "gender": "male",
                "height_cm": 180, "weight_kg": 85,
                "ap_hi": 135, "ap_lo": 85,
                "cholesterol": 2, "gluc": 1,
                "smoke": 0, "alco": 0, "active": 1,
                "hypertension": 0, "heart_disease": 0,
                "smoking_history": "former",
                "HbA1c_level": 6.0, "blood_glucose_level": 110,
            },
        },
    ]

    for s in samples:
        print("=" * 60)
        print(f"  {s['label']}")
        print("=" * 60)
        profile = twin.assess(s["inputs"])
        print(json.dumps(profile, indent=2))
        print()
