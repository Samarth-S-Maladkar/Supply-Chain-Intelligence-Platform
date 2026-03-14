"""
epoch/models/producer_models/demand_clustering.py
--------------------------------------------------
Producer-side model: Demand Zone Insight

Input  : product name + destination city
Output : demand zone, market context, sales metrics, competitor context

STATUS : STUB — heuristic scoring active.
         Replace _classify_demand_zone() with your trained cluster_model.pkl.
         The input/output contract is fixed.
"""

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from backend.utils.feature_engineering import build_producer_features


def _load_model():
    # import joblib
    # return joblib.load(os.path.join(ROOT, "saved_models", "cluster_model.pkl"))
    return None


def _classify_demand_zone(score: float, low_thresh: float, high_thresh: float) -> str:
    """
    ── REPLACE WITH YOUR cluster_model.pkl ──

    Input : normalised demand score (0–1) for this product+city combo
    Output: "High Demand" | "Emerging Market" | "Low Demand"

    When model is ready:
        model     = _load_model()
        features  = [[score, avg_profit, avg_late_risk, ...]]
        label_int = model.predict(features)[0]
        zone_map  = {2: "High Demand", 1: "Emerging Market", 0: "Low Demand"}
        return zone_map[label_int]
    """
    if score >= high_thresh:
        return "High Demand"
    elif score >= low_thresh:
        return "Emerging Market"
    return "Low Demand"


def _get_top_regions_for_product(df: pd.DataFrame, product_col: str, product: str) -> list:
    """Return top 5 regions by demand score for the given product (global context)."""
    prod_df = df[df[product_col].astype(str).str.lower() == product.lower()]
    if prod_df.empty or "Order_Region" not in prod_df.columns:
        return []
    prod_df = build_producer_features(prod_df)
    region_agg = prod_df.groupby("Order_Region", as_index=False).agg(
        demand_score=("demand_score", "mean"),
        total_sales =("Sales",         "sum"),
        order_count =("Sales",         "count"),
    ).sort_values("demand_score", ascending=False).head(5)

    # Compute zone thresholds from full product data
    all_scores = prod_df.groupby("Order_Region")["demand_score"].mean()
    low_t  = float(all_scores.quantile(0.33)) if len(all_scores) >= 3 else 0.33
    high_t = float(all_scores.quantile(0.67)) if len(all_scores) >= 3 else 0.67

    rows = []
    for _, row in region_agg.iterrows():
        rows.append({
            "region":       row["Order_Region"],
            "demand_score": round(float(row["demand_score"]), 3),
            "demand_zone":  _classify_demand_zone(row["demand_score"], low_t, high_t),
            "total_sales":  round(float(row["total_sales"]), 2),
            "order_count":  int(row["order_count"]),
        })
    return rows


# ── Main entry point ──────────────────────────────────────────────────────────

def predict_for_product_city(df: pd.DataFrame, product: str, city: str) -> dict:
    """
    Producer demand insight pipeline.

    Args:
        df      : Full cleaned dataset
        product : Product name (from dropdown)
        city    : Destination city (from dropdown)

    Returns:
        dict with keys:
          found           – bool
          product         – echoed
          city            – echoed
          demand_zone     – "High Demand" | "Emerging Market" | "Low Demand"
          demand_score    – float 0–1
          market          – market name (Africa / Europe / LATAM / Pacific Asia / USCA)
          region          – order region
          total_sales     – sum of sales for this product+city
          avg_profit      – average profit per order
          avg_late_risk   – average late delivery risk (0–1)
          order_count     – number of historical orders matched
          top_regions     – top 5 regions globally for this product (context)
    """
    # ── Identify product column ──
    product_col = next((c for c in ["Product_Name", "Category_Name"] if c in df.columns), None)
    if not product_col:
        raise ValueError("No product name column found in dataset.")

    city_col = next((c for c in ["Order_City", "Customer_City"] if c in df.columns), None)
    if not city_col:
        raise ValueError("No city column found in dataset.")

    # ── Filter ──
    subset = df[
        (df[product_col].astype(str).str.lower() == product.lower()) &
        (df[city_col].astype(str).str.lower()    == city.lower())
    ]

    found = not subset.empty

    if not found:
        # Fall back: product only for context, flag no city match
        subset = df[df[product_col].astype(str).str.lower() == product.lower()]

    if subset.empty:
        return {
            "found": False, "product": product, "city": city,
            "demand_zone": "Unknown", "demand_score": None,
            "market": "Unknown", "region": "Unknown",
            "total_sales": 0, "avg_profit": 0, "avg_late_risk": 0,
            "order_count": 0, "top_regions": [],
            "message": "No data found for this product."
        }

    # ── Feature engineering ──
    subset = build_producer_features(subset)

    # ── Aggregate ──
    demand_score  = float(subset["demand_score"].mean())
    total_sales   = float(subset["Sales"].sum())                        if "Sales"                  in subset.columns else 0
    avg_profit    = float(subset["Order_Profit_Per_Order"].mean())      if "Order_Profit_Per_Order" in subset.columns else 0
    avg_late_risk = float(subset["Late_delivery_risk"].mean())          if "Late_delivery_risk"     in subset.columns else 0
    order_count   = int(len(subset))

    market = str(subset["Market"].mode().iloc[0])        if "Market"       in subset.columns and not subset["Market"].dropna().empty        else "Unknown"
    region = str(subset["Order_Region"].mode().iloc[0])  if "Order_Region" in subset.columns and not subset["Order_Region"].dropna().empty  else "Unknown"

    # ── Compute thresholds from entire product's data ──
    all_product_df = df[df[product_col].astype(str).str.lower() == product.lower()]
    if len(all_product_df) >= 3:
        all_product_df  = build_producer_features(all_product_df)
        region_scores   = all_product_df.groupby(city_col)["demand_score"].mean()
        low_t  = float(region_scores.quantile(0.33))
        high_t = float(region_scores.quantile(0.67))
    else:
        low_t, high_t = 0.33, 0.67

    demand_zone = _classify_demand_zone(demand_score, low_t, high_t)

    # ── Top regions globally for this product ──
    top_regions = _get_top_regions_for_product(df, product_col, product)

    return {
        "found":         found,
        "product":       product,
        "city":          city,
        "demand_zone":   demand_zone,
        "demand_score":  round(demand_score, 3),
        "market":        market,
        "region":        region,
        "total_sales":   round(total_sales, 2),
        "avg_profit":    round(avg_profit, 2),
        "avg_late_risk": round(avg_late_risk, 3),
        "order_count":   order_count,
        "top_regions":   top_regions,
        "data_source":   "city-matched" if found else "product-only",
        "message":       None if found else f"No exact city match — showing product-level data for '{product}'.",
    }
