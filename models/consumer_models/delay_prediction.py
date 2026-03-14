"""
epoch/models/consumer_models/delay_prediction.py
-------------------------------------------------
Consumer-side model: Delivery Delay Prediction

STATUS: STUB — rule-based placeholder active.
        Replace _predict_delay() with your trained delay_model.pkl.

Output: per-order predicted delay risk (0/1) and confidence score.
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backend.utils.feature_engineering import build_consumer_features

MODEL_PATH = os.path.join(ROOT, "saved_models", "delay_model.pkl")


def _load_model():
    """Uncomment when model is ready."""
    # import joblib
    # return joblib.load(MODEL_PATH)
    return None


def _predict_delay(df: pd.DataFrame) -> pd.DataFrame:
    """
    ── REPLACE WITH YOUR TRAINED MODEL ──

    Input : feature-engineered DataFrame
    Output: DataFrame with added columns:
              predicted_delay  (int)   – 0 = on time, 1 = delayed
              delay_confidence (float) – model confidence 0.0–1.0

    When model is ready:
        model   = _load_model()
        X       = df[FEATURE_COLS]
        preds   = model.predict(X)
        proba   = model.predict_proba(X)[:, 1]
        df["predicted_delay"]    = preds
        df["delay_confidence"]   = proba
    """
    # Stub heuristic: shipping_delay_gap > 0 → predicted late
    if "shipping_delay_gap" in df.columns:
        df["predicted_delay"]   = (df["shipping_delay_gap"] > 0).astype(int)
        df["delay_confidence"]  = df["shipping_delay_gap"].clip(0, 10) / 10.0
    else:
        df["predicted_delay"]   = df.get("Late_delivery_risk", pd.Series([0]*len(df))).astype(int)
        df["delay_confidence"]  = df["predicted_delay"].astype(float)
    return df


def run_delay_prediction(df: pd.DataFrame) -> dict:
    """
    Full pipeline for delay prediction.

    Returns:
        dict with keys:
          summary – overall delay rate, on-time rate, total orders
          orders  – list of per-order predictions (sample of 500 for perf)
    """
    df = build_consumer_features(df)
    df = _predict_delay(df)

    total   = len(df)
    delayed = int(df["predicted_delay"].sum())
    on_time = total - delayed

    summary = {
        "total_orders":    total,
        "predicted_late":  delayed,
        "predicted_on_time": on_time,
        "delay_rate_pct":  round(delayed / total * 100, 1) if total else 0,
        "markets":         sorted(df["Market"].dropna().unique().tolist()) if "Market" in df.columns else [],
    }

    # Sample rows for frontend table (top delayed first)
    cols = ["Order_Id", "Order_Region", "Market", "Shipping_Mode",
            "predicted_delay", "delay_confidence", "shipping_delay_gap"]
    available = [c for c in cols if c in df.columns]
    sample = (
        df[available]
        .sort_values("predicted_delay", ascending=False)
        .head(500)
        .fillna("—")
        .to_dict(orient="records")
    )

    return {"summary": summary, "orders": sample}
