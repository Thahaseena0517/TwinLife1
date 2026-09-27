"""
Module 8 -- monitoring_agent.py
Proactive Monitoring Agent for TwinLife AI.

Uses APScheduler to run periodic (weekly) health/finance/insurance
profile checks. Compares current profiles against saved snapshots and
generates alerts when scores cross defined thresholds.

Usage:
    monitor = MonitoringAgent(user_id="user_001",
                              health_inputs={...},
                              finance_inputs={...},
                              insurance_inputs={...})
    monitor.start()   # starts the scheduler
    monitor.stop()    # stops the scheduler
"""

from __future__ import annotations

import os
import json
import warnings
from datetime import datetime
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from health_twin import HealthTwin
from finance_twin import FinanceTwin
from insurance_twin import InsuranceTwin

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
#  PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "snapshots")
NOTIFICATIONS_DIR = os.path.join(BASE_DIR, "data", "notifications")


# ---------------------------------------------------------------------------
#  THRESHOLD CONFIGURATION
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "health_score_drop": 10,     # alert if overall_health_score drops by >10
    "dti_rise_above": 0.40,      # alert if DTI ratio rises above 0.40
    "savings_rate_below": 0.10,  # alert if savings rate falls below 10%
    "insurance_score_drop": 15,  # alert if insurance score drops by >15
    "new_rider_gap": True,       # alert if new insurance rider gaps appear
}


class MonitoringAgent:
    """Proactive monitoring agent with scheduled profile checks.

    Periodically recomputes all three twin profiles, compares against
    the last saved snapshot, and generates alerts when defined thresholds
    are crossed.

    Args:
        user_id:          Unique user identifier.
        health_inputs:    Dict of user health vitals.
        finance_inputs:   Dict of user financial data.
        insurance_inputs: Dict of user insurance data.
        check_interval_seconds: Interval between checks (default: 604800 = 1 week).
    """

    def __init__(
        self,
        user_id: str,
        health_inputs: dict,
        finance_inputs: dict,
        insurance_inputs: dict,
        check_interval_seconds: int = 604800,  # 1 week
    ) -> None:
        self.user_id = user_id
        self.health_inputs = health_inputs
        self.finance_inputs = finance_inputs
        self.insurance_inputs = insurance_inputs
        self.check_interval = check_interval_seconds

        # Initialize twins
        self.health_twin = HealthTwin()
        self.finance_twin = FinanceTwin()
        self.insurance_twin = InsuranceTwin()

        # Ensure storage directories exist
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        os.makedirs(NOTIFICATIONS_DIR, exist_ok=True)

        # Scheduler
        self.scheduler = BackgroundScheduler()
        self._is_running = False

    # ------------------------------------------------------------------
    #  SCHEDULER CONTROL
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the monitoring scheduler."""
        if self._is_running:
            print(f"[Monitor] Already running for user {self.user_id}")
            return

        self.scheduler.add_job(
            func=self.run_check,
            trigger=IntervalTrigger(seconds=self.check_interval),
            id=f"monitor_{self.user_id}",
            name=f"TwinLife Monitor — {self.user_id}",
            replace_existing=True,
        )
        self.scheduler.start()
        self._is_running = True
        print(f"[Monitor] Started for user {self.user_id} "
              f"(interval: {self.check_interval}s)")

    def stop(self) -> None:
        """Stop the monitoring scheduler."""
        if self._is_running:
            self.scheduler.shutdown(wait=False)
            self._is_running = False
            print(f"[Monitor] Stopped for user {self.user_id}")

    # ------------------------------------------------------------------
    #  CORE CHECK LOGIC
    # ------------------------------------------------------------------
    def run_check(self) -> list[dict]:
        """Run a single monitoring check cycle.

        Recomputes all three profiles, compares against the last snapshot,
        generates alerts for threshold crossings, and saves the new snapshot.

        Returns:
            List of alert dicts generated during this check.
        """
        timestamp = datetime.now().isoformat()
        print(f"\n[Monitor] Running check for {self.user_id} at {timestamp}")

        # --- Recompute current profiles ---
        current_health = self.health_twin.assess(self.health_inputs)
        current_finance = self.finance_twin.assess(self.finance_inputs)
        current_insurance = self.insurance_twin.assess(
            self.insurance_inputs, current_health, current_finance,
        )

        current_snapshot = {
            "timestamp": timestamp,
            "health": current_health,
            "finance": current_finance,
            "insurance": current_insurance,
        }

        # --- Load last snapshot ---
        last_snapshot = self._load_latest_snapshot()

        # --- Compare and generate alerts ---
        alerts = self._compare_snapshots(last_snapshot, current_snapshot)

        # --- Save new snapshot ---
        self._save_snapshot(current_snapshot)

        # --- Save notifications ---
        for alert in alerts:
            self._save_notification(alert)

        if alerts:
            print(f"[Monitor] Generated {len(alerts)} alert(s):")
            for a in alerts:
                print(f"  ⚠ [{a['severity']}] {a['message']}")
        else:
            print("[Monitor] No threshold crossings detected.")

        return alerts

    # ------------------------------------------------------------------
    #  COMPARISON LOGIC
    # ------------------------------------------------------------------
    def _compare_snapshots(
        self, old: dict | None, new: dict,
    ) -> list[dict]:
        """Compare old and new snapshots, return alerts for threshold crossings."""
        alerts = []
        timestamp = new["timestamp"]

        if old is None:
            # First snapshot — no comparison possible, just baseline
            alerts.append({
                "user_id": self.user_id,
                "timestamp": timestamp,
                "type": "baseline",
                "severity": "info",
                "message": "Initial baseline snapshot recorded. "
                           "Future checks will compare against this.",
                "data": {},
            })
            return alerts

        # --- Health score drop ---
        old_health_score = old.get("health", {}).get("overall_health_score", 100)
        new_health_score = new["health"].get("overall_health_score", 100)
        drop = old_health_score - new_health_score
        if drop > THRESHOLDS["health_score_drop"]:
            alerts.append({
                "user_id": self.user_id,
                "timestamp": timestamp,
                "type": "health_score_drop",
                "severity": "warning",
                "message": (
                    f"Health score dropped by {drop} points "
                    f"(from {old_health_score} to {new_health_score}). "
                    "Review recent health changes and consult a healthcare provider."
                ),
                "data": {
                    "old_score": old_health_score,
                    "new_score": new_health_score,
                    "drop": drop,
                },
            })

        # --- DTI ratio rise ---
        new_dti = new["finance"].get("dti_ratio", 0)
        old_dti = old.get("finance", {}).get("dti_ratio", 0)
        if new_dti > THRESHOLDS["dti_rise_above"] and new_dti > old_dti:
            alerts.append({
                "user_id": self.user_id,
                "timestamp": timestamp,
                "type": "dti_high",
                "severity": "warning",
                "message": (
                    f"Debt-to-Income ratio has risen to {new_dti:.2f} "
                    f"(above the {THRESHOLDS['dti_rise_above']:.2f} threshold). "
                    "Consider reducing debt obligations or increasing income."
                ),
                "data": {
                    "old_dti": old_dti,
                    "new_dti": new_dti,
                    "threshold": THRESHOLDS["dti_rise_above"],
                },
            })

        # --- Savings rate drop ---
        new_savings = new["finance"].get("savings_rate", 1)
        if new_savings < THRESHOLDS["savings_rate_below"]:
            alerts.append({
                "user_id": self.user_id,
                "timestamp": timestamp,
                "type": "low_savings_rate",
                "severity": "warning",
                "message": (
                    f"Savings rate has fallen to {new_savings:.0%} "
                    f"(below the {THRESHOLDS['savings_rate_below']:.0%} minimum). "
                    "Review monthly expenses and identify areas to cut back."
                ),
                "data": {
                    "savings_rate": new_savings,
                    "threshold": THRESHOLDS["savings_rate_below"],
                },
            })

        # --- Insurance score drop ---
        old_ins_score = old.get("insurance", {}).get("insurance_adequacy_score", 100)
        new_ins_score = new["insurance"].get("insurance_adequacy_score", 100)
        ins_drop = old_ins_score - new_ins_score
        if ins_drop > THRESHOLDS["insurance_score_drop"]:
            alerts.append({
                "user_id": self.user_id,
                "timestamp": timestamp,
                "type": "insurance_score_drop",
                "severity": "warning",
                "message": (
                    f"Insurance adequacy score dropped by {ins_drop} points "
                    f"(from {old_ins_score} to {new_ins_score}). "
                    "Review your insurance coverage and consider enhancing it."
                ),
                "data": {
                    "old_score": old_ins_score,
                    "new_score": new_ins_score,
                    "drop": ins_drop,
                },
            })

        # --- New rider gaps ---
        if THRESHOLDS["new_rider_gap"]:
            old_gaps = set(old.get("insurance", {}).get(
                "rider_analysis", {},
            ).get("gaps", []))
            new_gaps = set(new["insurance"].get(
                "rider_analysis", {},
            ).get("gaps", []))
            fresh_gaps = new_gaps - old_gaps
            if fresh_gaps:
                alerts.append({
                    "user_id": self.user_id,
                    "timestamp": timestamp,
                    "type": "new_rider_gap",
                    "severity": "info",
                    "message": (
                        f"New insurance rider gap(s) detected: "
                        f"{', '.join(fresh_gaps)}. "
                        "Consider adding these riders to your policy."
                    ),
                    "data": {
                        "new_gaps": list(fresh_gaps),
                    },
                })

        return alerts

    # ------------------------------------------------------------------
    #  PERSISTENCE (file-based for academic simplicity)
    # ------------------------------------------------------------------
    def _snapshot_path(self) -> str:
        return os.path.join(SNAPSHOTS_DIR, f"{self.user_id}_latest.json")

    def _load_latest_snapshot(self) -> dict | None:
        """Load the most recent snapshot from disk."""
        path = self._snapshot_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_snapshot(self, snapshot: dict) -> None:
        """Save a snapshot to disk (overwrites latest)."""
        # Make snapshot JSON-serializable (handle non-serializable types)
        clean = json.loads(json.dumps(snapshot, default=str))
        path = self._snapshot_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)

    def _save_notification(self, alert: dict) -> None:
        """Append a notification to the user's notification log."""
        log_path = os.path.join(
            NOTIFICATIONS_DIR, f"{self.user_id}_notifications.jsonl",
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, default=str) + "\n")

    # ------------------------------------------------------------------
    #  QUERY NOTIFICATIONS
    # ------------------------------------------------------------------
    def get_notifications(self, limit: int = 20) -> list[dict]:
        """Retrieve the latest notifications for this user.

        Args:
            limit: Maximum number of notifications to return.

        Returns:
            List of notification dicts, most recent first.
        """
        log_path = os.path.join(
            NOTIFICATIONS_DIR, f"{self.user_id}_notifications.jsonl",
        )
        if not os.path.exists(log_path):
            return []

        notifications = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    notifications.append(json.loads(line))

        # Return most recent first
        return notifications[-limit:][::-1]

    # ------------------------------------------------------------------
    #  UPDATE INPUTS (for dynamic monitoring)
    # ------------------------------------------------------------------
    def update_inputs(
        self,
        health_inputs: dict | None = None,
        finance_inputs: dict | None = None,
        insurance_inputs: dict | None = None,
    ) -> None:
        """Update the user's inputs for future monitoring checks.

        Args:
            health_inputs:    New health vitals (or None to keep current).
            finance_inputs:   New financial data (or None to keep current).
            insurance_inputs: New insurance data (or None to keep current).
        """
        if health_inputs:
            self.health_inputs = health_inputs
        if finance_inputs:
            self.finance_inputs = finance_inputs
        if insurance_inputs:
            self.insurance_inputs = insurance_inputs


# ======================================================================
#  STANDALONE VERIFICATION (python monitoring_agent.py)
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Module 8: Monitoring Agent — Threshold Check Test")
    print("=" * 60)

    # Sample user inputs
    health_inputs = {
        "age": 45, "gender": "male",
        "height_cm": 175, "weight_kg": 85,
        "ap_hi": 140, "ap_lo": 90,
        "cholesterol": 2, "gluc": 2,
        "smoke": 0, "alco": 0, "active": 1,
        "hypertension": 0, "heart_disease": 0,
        "smoking_history": "former",
        "HbA1c_level": 6.2, "blood_glucose_level": 115,
    }

    finance_inputs = {
        "income": 80000,
        "fixed_expenses": 25000,
        "variable_expenses": 20000,
        "emis": 18000,
        "savings_balance": 150000,
    }

    insurance_inputs = {
        "sum_insured": 500000,
        "annual_premium": 30000,
        "annual_income": 960000,
        "existing_riders": ["Personal Accident Rider"],
    }

    monitor = MonitoringAgent(
        user_id="test_user_001",
        health_inputs=health_inputs,
        finance_inputs=finance_inputs,
        insurance_inputs=insurance_inputs,
    )

    # Run first check (baseline)
    print("\n[1] First check (baseline)...")
    alerts1 = monitor.run_check()
    print(f"    Alerts: {len(alerts1)}")

    # Run second check (should detect no changes with same inputs)
    print("\n[2] Second check (same inputs)...")
    alerts2 = monitor.run_check()
    print(f"    Alerts: {len(alerts2)}")

    # Simulate worsened finances
    print("\n[3] Third check (worsened finances)...")
    monitor.update_inputs(finance_inputs={
        "income": 80000,
        "fixed_expenses": 30000,
        "variable_expenses": 25000,
        "emis": 25000,  # increased EMIs
        "savings_balance": 80000,  # reduced savings
    })
    alerts3 = monitor.run_check()
    print(f"    Alerts: {len(alerts3)}")

    # Show all notifications
    print(f"\n{'─' * 60}")
    print("  All Notifications:")
    print(f"{'─' * 60}")
    for n in monitor.get_notifications():
        print(f"  [{n['severity'].upper()}] {n['message']}")
