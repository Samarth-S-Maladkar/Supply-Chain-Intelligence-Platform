"""
epoch/models/consumer_models/consumer_insights.py
--------------------------------------------------
Consumer-side unified insight model.

Input  : product name + customer pincode
Output : delivery insights — delay risk, estimated days,
         recommended shipping mode, region context

STATUS : STUB — heuristic logic active.
         Replace the three _predict_* functions with your trained
         delay_model.pkl, risk_model.pkl once ready.
         The input/output contract is fixed.
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


# ── Model loaders (uncomment when .pkl files are ready) ──────────────────────

def _load_delay_model():
    # import joblib
    # return joblib.load(os.path.join(ROOT, "saved_models", "delay_model.pkl"))
    return None

def _load_risk_model():
    # import joblib
    # return joblib.load(os.path.join(ROOT, "saved_models", "risk_model.pkl"))
    return None


# ── Stub predictors (replace these blocks with your model calls) ──────────────

def _predict_delay_risk(subset: pd.DataFrame) -> dict:
    """
    ── REPLACE WITH YOUR delay_model.pkl ──

    Input : filtered DataFrame rows matching product + pincode
    Output: dict with delay_risk (float 0–1), is_late_predicted (bool)
    """
    if subset.empty:
        return {"delay_risk": 0.5, "is_late_predicted": False, "confidence": 50}

    late_rate   = float(subset["Late_delivery_risk"].mean()) if "Late_delivery_risk" in subset.columns else 0.5
    is_late     = late_rate >= 0.5
    confidence  = int(abs(late_rate - 0.5) * 200)   # 0–100

    return {
        "delay_risk":        round(late_rate, 3),
        "is_late_predicted": is_late,
        "confidence":        confidence,
    }


def _predict_estimated_days(subset: pd.DataFrame) -> dict:
    """
    ── REPLACE WITH YOUR model ──

    Returns estimated shipping days (real vs scheduled).
    """
    if subset.empty:
        return {"estimated_days": None, "scheduled_days": None, "avg_gap": None}

    real_days  = subset["Days_for_shipping_real"].dropna()      if "Days_for_shipping_real"          in subset.columns else pd.Series()
    sched_days = subset["Days_for_shipment_scheduled"].dropna() if "Days_for_shipment_scheduled"      in subset.columns else pd.Series()

    return {
        "estimated_days":  round(float(real_days.mean()),  1) if not real_days.empty  else None,
        "scheduled_days":  round(float(sched_days.mean()), 1) if not sched_days.empty else None,
        "avg_gap":         round(float((real_days.mean() - sched_days.mean())), 1)
                           if (not real_days.empty and not sched_days.empty) else None,
    }


def _predict_shipping_mode(subset: pd.DataFrame, delay_risk: float) -> dict:
    """
    ── REPLACE WITH YOUR shipping_recommendation model ──

    Recommends best shipping mode given product + location context.
    """
    if subset.empty or delay_risk is None:
        return {"recommended_mode": "Standard Class", "reason": "Insufficient data for recommendation"}

    # Most common mode used for this product/region
    if "Shipping_Mode" in subset.columns:
        common_mode = subset["Shipping_Mode"].mode()
        common      = common_mode.iloc[0] if not common_mode.empty else "Standard Class"
    else:
        common = "Standard Class"

    # Override if delay risk is high
    if delay_risk >= 0.7:
        return {"recommended_mode": "First Class",    "reason": "High delay risk detected — priority shipping advised"}
    elif delay_risk >= 0.5:
        return {"recommended_mode": "Second Class",   "reason": "Moderate delay risk — expedited shipping recommended"}
    else:
        return {"recommended_mode": common,           "reason": "Low delay risk — standard routing is sufficient"}


def _get_region_context(subset: pd.DataFrame) -> dict:
    """Derive region / market context from matching rows."""
    if subset.empty:
        return {"region": "Unknown", "market": "Unknown", "country": "Unknown"}

    region  = subset["Order_Region"].mode().iloc[0]  if "Order_Region"  in subset.columns and not subset["Order_Region"].dropna().empty  else "Unknown"
    market  = subset["Market"].mode().iloc[0]        if "Market"        in subset.columns and not subset["Market"].dropna().empty        else "Unknown"
    country = subset["Order_Country"].mode().iloc[0] if "Order_Country" in subset.columns and not subset["Order_Country"].dropna().empty else "Unknown"

    return {"region": str(region), "market": str(market), "country": str(country)}


# ── Main entry point ──────────────────────────────────────────────────────────

def predict_for_product_pincode(df: pd.DataFrame, product: str, pincode: str) -> dict:
    """
    Unified consumer insight pipeline.

    Args:
        df      : Full cleaned dataset (from preprocessing)
        product : Product name string (from dropdown)
        pincode : Customer zipcode string (from dropdown)

    Returns:
        dict with keys:
          found         – bool, whether data matched
          product       – echoed product name
          pincode       – echoed pincode
          context       – region / market / country
          delay         – delay risk prediction
          delivery      – estimated days
          shipping      – recommended shipping mode
          order_count   – how many historical orders matched
          avg_sales     – average sales value for this product
    """
    # ── Filter by product ──
    product_col = next((c for c in ["Product_Name", "Category_Name"] if c in df.columns), None)
    if not product_col:
        raise ValueError("No product name column found in dataset.")

    subset = df[df[product_col].astype(str).str.lower() == product.lower()]

    # ── Further filter by pincode if column exists ──
    pincode_col = "Customer_Zipcode" if "Customer_Zipcode" in df.columns else None
    subset_pin  = pd.DataFrame()

    if pincode_col and not subset.empty:
        try:
            subset_pin = subset[
                subset[pincode_col].astype(str).str.split(".").str[0] == str(pincode)
            ]
        except Exception:
            subset_pin = pd.DataFrame()

    # Use pincode-filtered if we have rows, otherwise fall back to product-only
    working_subset = subset_pin if not subset_pin.empty else subset
    found          = not subset.empty

    # ── Run predictions ──
    delay_result    = _predict_delay_risk(working_subset)
    delivery_result = _predict_estimated_days(working_subset)
    shipping_result = _predict_shipping_mode(working_subset, delay_result["delay_risk"])
    context         = _get_region_context(working_subset)

    avg_sales = None
    if "Sales" in working_subset.columns and not working_subset.empty:
        avg_sales = round(float(working_subset["Sales"].mean()), 2)

    return {
        "found":       found,
        "product":     product,
        "pincode":     pincode,
        "context":     context,
        "delay":       delay_result,
        "delivery":    delivery_result,
        "shipping":    shipping_result,
        "order_count": int(len(working_subset)),
        "avg_sales":   avg_sales,
        "data_source": "pincode-matched" if not subset_pin.empty else "product-matched",
    }
