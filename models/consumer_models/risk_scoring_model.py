"""
epoch/models/consumer_models/risk_scoring_model.py
---------------------------------------------------
Consumer-side model: Order / Region Risk Scoring

STATUS: STUB — heuristic scoring active.
        Replace _score_risk() with your trained risk_model.pkl.

Output: risk score 0–100 per order, with tier: Low / Medium / High / Critical
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backend.utils.feature_engineering import build_consumer_features
from backend.utils.preprocessing import normalise_0_1

MODEL_PATH = os.path.join(ROOT, "saved_models", "risk_model.pkl")


def _load_model():
    """Uncomment when model is ready."""
    # import joblib
    # return joblib.load(MODEL_PATH)
    return None


def _score_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    ── REPLACE WITH YOUR TRAINED MODEL ──

    Input : feature-engineered DataFrame
    Output: DataFrame with added columns:
              risk_score (float 0–100)
              risk_tier  (str)  – "Low" | "Medium" | "High" | "Critical"

    When model is ready:
        model  = _load_model()
        X      = df[FEATURE_COLS]
        scores = model.predict_proba(X)[:, 1] * 100
        df["risk_score"] = scores
    """
    # Stub: composite risk from late delivery risk + discount + delay gap
    late   = normalise_0_1(pd.to_numeric(df.get("Late_delivery_risk",     0), errors="coerce").fillna(0))
    disc   = normalise_0_1(pd.to_numeric(df.get("Order_Item_Discount_Rate",0), errors="coerce").fillna(0))
    gap    = normalise_0_1(pd.to_numeric(df.get("shipping_delay_gap",     0), errors="coerce").fillna(0).clip(0))

    df["risk_score"] = ((late * 0.50 + disc * 0.20 + gap * 0.30) * 100).round(1)

    def _tier(score):
        if score >= 75:  return "Critical"
        if score >= 50:  return "High"
        if score >= 25:  return "Medium"
        return "Low"

    df["risk_tier"] = df["risk_score"].apply(_tier)
    return df


def run_risk_scoring(df: pd.DataFrame) -> dict:
    """
    Full pipeline for risk scoring.

    Returns:
        dict with keys:
          summary         – counts by tier, avg risk score
          region_risk     – per-region average risk (for colour-coded table)
          orders          – per-order risk (sample of 500)
    """
    df = build_consumer_features(df)
    df = _score_risk(df)

    summary = {
        "avg_risk_score": round(float(df["risk_score"].mean()), 1),
        "critical_count": int((df["risk_tier"] == "Critical").sum()),
        "high_count":     int((df["risk_tier"] == "High").sum()),
        "medium_count":   int((df["risk_tier"] == "Medium").sum()),
        "low_count":      int((df["risk_tier"] == "Low").sum()),
        "total_orders":   len(df),
    }

    # Region-level aggregation for table
    region_risk = None
    if "Order_Region" in df.columns:
        rdf = df.groupby(["Order_Region", "Market"], as_index=False).agg(
            avg_risk_score = ("risk_score", "mean"),
            order_count    = ("risk_score", "count"),
        ).round(2)
        rdf["risk_tier"] = rdf["avg_risk_score"].apply(
            lambda s: "Critical" if s>=75 else "High" if s>=50 else "Medium" if s>=25 else "Low"
        )
        rdf = rdf.sort_values("avg_risk_score", ascending=False)
        region_risk = rdf.rename(columns={
            "Order_Region": "region", "Market": "market"
        }).to_dict(orient="records")

    # Per-order sample
    cols = ["Order_Id", "Order_Region", "Market", "Shipping_Mode",
            "risk_score", "risk_tier", "Late_delivery_risk"]
    available = [c for c in cols if c in df.columns]
    orders = (
        df[available]
        .sort_values("risk_score", ascending=False)
        .head(500)
        .fillna("—")
        .to_dict(orient="records")
    )

    return {"summary": summary, "region_risk": region_risk, "orders": orders}
