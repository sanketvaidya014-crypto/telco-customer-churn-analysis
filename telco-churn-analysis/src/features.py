"""Shared paths, data access and feature engineering."""
import pathlib
import sqlite3

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "telco.db"
SQL_DIR, OUT = ROOT / "sql", ROOT / "outputs"
FIG, TAB, MOD = OUT / "figures", OUT / "tables", OUT / "models"
for _d in (FIG, TAB, MOD):
    _d.mkdir(parents=True, exist_ok=True)

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PROTECTIVE = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
TARGET = "churn_flag"
DROP = ["customerID", "Churn", TARGET]


def load_customers() -> pd.DataFrame:
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query("SELECT * FROM customers", con)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural features. Uses only information available BEFORE churn (no target leakage)."""
    df = df.copy()
    df["num_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["num_protective"] = (df[PROTECTIVE] == "Yes").sum(axis=1)
    df["autopay"] = df.PaymentMethod.str.contains("automatic", case=False).astype(int)
    df["avg_monthly_spend"] = np.where(df.tenure > 0, df.TotalCharges / df.tenure.replace(0, np.nan), df.MonthlyCharges)
    df["charge_vs_history"] = df.MonthlyCharges - df.avg_monthly_spend   # proxy for recent price increases
    return df
