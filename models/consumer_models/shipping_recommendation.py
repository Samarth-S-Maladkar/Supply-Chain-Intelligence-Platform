"""
epoch/models/consumer_models/shipping_recommendation.py
--------------------------------------------------------
Consumer-side model: Optimal Shipping Mode Recommendation

STATUS: STUB — rule-based recommender active.
        Replace _recommend_mode() with your trained model.

Output: recommended shipping mode per order with reasoning.
Modes: Standard Class | First Class | Second Class | Same Day
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backend.utils.feature_engineering import build_consumer_features

SHIPPING_MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]


def _load_model():
    """Uncomment when model is ready."""
    # import joblib
    # return joblib.load(os.path.join(ROOT, "saved_models", "shipping_model.pkl"))
    return None


def _recommend_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    ── REPLACE WITH YOUR TRAINED MODEL ──

    Input : feature-engineered DataFrame
    Output: DataFrame with added columns:
              recommended_mode (str)  – best shipping mode
              recommendation_reason  – short explanation string

    When model is ready:
        model = _load_model()
        X     = df[FEATURE_COLS]
        preds = model.predict(X)
        df["recommended_mode"] = preds
    """
    # Stub: recommend based on scheduled days and late delivery risk
    def _rule(row):
        scheduled = row.get("Days_for_shipment_scheduled", 5)
        risk      = row.get("Late_delivery_risk", 0)
        profit    = row.get("Order_Profit_Per_Order", 0)

        if scheduled <= 1 or risk == 1:
            return "Same Day", "High delay risk — expedited shipping advised"
        elif scheduled <= 2 and profit > 50:
            return "First Class", "High-value order, priority shipping recommended"
        elif scheduled <= 4:
            return "Second Class", "Moderate timeline, balanced cost/speed"
        else:
            return "Standard Class", "Adequate lead time for economy shipping"

    results = df.apply(_rule, axis=1, result_type="expand")
    df["recommended_mode"]      = results[0]
    df["recommendation_reason"] = results[1]
    return df


def run_shipping_recommendation(df: pd.DataFrame) -> dict:
    """
    Full pipeline for shipping recommendation.

    Returns:
        dict with keys:
          summary       – mode distribution counts
          orders        – per-order recommendations (sample of 500)
    """
    df = build_consumer_features(df)
    df = _recommend_mode(df)

    mode_counts = df["recommended_mode"].value_counts().to_dict()

    summary = {
        "total_orders":    len(df),
        "mode_breakdown":  mode_counts,
        "same_day_pct":    round(mode_counts.get("Same Day",       0) / len(df) * 100, 1) if len(df) else 0,
        "first_class_pct": round(mode_counts.get("First Class",    0) / len(df) * 100, 1) if len(df) else 0,
        "second_class_pct":round(mode_counts.get("Second Class",   0) / len(df) * 100, 1) if len(df) else 0,
        "standard_pct":    round(mode_counts.get("Standard Class", 0) / len(df) * 100, 1) if len(df) else 0,
    }

    cols = ["Order_Id", "Order_Region", "Market", "Shipping_Mode",
            "recommended_mode", "recommendation_reason",
            "Days_for_shipment_scheduled", "Late_delivery_risk"]
    available = [c for c in cols if c in df.columns]
    orders = (
        df[available]
        .head(500)
        .fillna("—")
        .to_dict(orient="records")
    )

    return {"summary": summary, "orders": orders}
