"""Central paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
LOG_DIR = ROOT / "reports" / "logs"

# The dataset ends on this date (last viewing / churn / ticket date).
AS_OF_DATE = "2026-06-30"

TABLES = [
    "customers", "subscription_plans", "subscriptions", "content",
    "viewing_activity", "payments", "support_tickets", "customer_feedback",
    "churn_labels",
]
