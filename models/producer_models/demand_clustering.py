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


def _build_region_category_heatmap(df: pd.DataFrame, product_col: str, product: str) -> dict:
    """Build heatmap data: rows=regions, cols=categories, values=order quantity sum."""
    if "Order_Region" not in df.columns:
        return {"rows": [], "columns": [], "values": []}

    category_col = "Category_Name" if "Category_Name" in df.columns else None
    qty_col = "Order_Item_Quantity" if "Order_Item_Quantity" in df.columns else None
    if not category_col or not qty_col:
        return {"rows": [], "columns": [], "values": []}

    prod_df = df[df[product_col].astype(str).str.lower() == product.lower()].copy()
    if prod_df.empty:
        return {"rows": [], "columns": [], "values": []}

    pivot = (
        prod_df
        .pivot_table(
            values=qty_col,
            index="Order_Region",
            columns=category_col,
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )

    if pivot.empty:
        return {"rows": [], "columns": [], "values": []}

    column_totals = pivot.sum(axis=0).sort_values(ascending=False)
    top_columns = column_totals.head(10).index.tolist()
    pivot = pivot[top_columns]

    row_totals = pivot.sum(axis=1).sort_values(ascending=False)
    top_rows = row_totals.head(10).index.tolist()
    pivot = pivot.loc[top_rows]

    values = [[float(v) for v in row] for row in pivot.values.tolist()]
    return {
        "rows": [str(r) for r in pivot.index.tolist()],
        "columns": [str(c) for c in pivot.columns.tolist()],
        "values": values,
    }


def _get_high_demand_low_profit_regions(df: pd.DataFrame, product_col: str, product: str) -> list:
    """Return top-10 regions with high demand quantity and lower average profit."""
    if "Order_Region" not in df.columns:
        return []

    qty_col = "Order_Item_Quantity" if "Order_Item_Quantity" in df.columns else None
    profit_col = "Order_Profit_Per_Order" if "Order_Profit_Per_Order" in df.columns else None
    sales_col = "Sales" if "Sales" in df.columns else None
    if not qty_col or not profit_col:
        return []

    prod_df = df[df[product_col].astype(str).str.lower() == product.lower()].copy()
    if prod_df.empty:
        return []

    grouped = (
        prod_df
        .groupby("Order_Region", as_index=False)
        .agg(
            total_quantity=(qty_col, "sum"),
            avg_profit_per_order=(profit_col, "mean"),
            total_sales=(sales_col, "sum") if sales_col else (qty_col, "count"),
            order_count=(qty_col, "count"),
        )
    )

    if grouped.empty:
        return []

    grouped = grouped.sort_values(
        ["total_quantity", "avg_profit_per_order"],
        ascending=[False, True],
    ).head(10)

    rows = []
    for rank, (_, row) in enumerate(grouped.iterrows(), start=1):
        rows.append({
            "rank": rank,
            "region": str(row["Order_Region"]),
            "total_quantity": int(row["total_quantity"]),
            "avg_profit_per_order": round(float(row["avg_profit_per_order"]), 2),
            "total_sales": round(float(row["total_sales"]), 2),
            "order_count": int(row["order_count"]),
        })
    return rows


def _build_region_category_heatmap_overview(df: pd.DataFrame) -> dict:
    """Build overview heatmap data: rows=regions, cols=categories, values=order quantity sum."""
    if "Order_Region" not in df.columns:
        return {"rows": [], "columns": [], "values": []}

    category_col = "Category_Name" if "Category_Name" in df.columns else None
    qty_col = "Order_Item_Quantity" if "Order_Item_Quantity" in df.columns else None
    if not category_col or not qty_col:
        return {"rows": [], "columns": [], "values": []}

    pivot = (
        df
        .pivot_table(
            values=qty_col,
            index="Order_Region",
            columns=category_col,
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
    )

    if pivot.empty:
        return {"rows": [], "columns": [], "values": []}

    column_totals = pivot.sum(axis=0).sort_values(ascending=False)
    pivot = pivot[column_totals.head(10).index.tolist()]
    row_totals = pivot.sum(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[row_totals.head(10).index.tolist()]

    values = [[float(v) for v in row] for row in pivot.values.tolist()]
    return {
        "rows": [str(r) for r in pivot.index.tolist()],
        "columns": [str(c) for c in pivot.columns.tolist()],
        "values": values,
    }


def _get_high_demand_low_profit_regions_overview(df: pd.DataFrame) -> list:
    """Return top-10 regions with high demand quantity and lower average profit from full dataset."""
    if "Order_Region" not in df.columns:
        return []

    qty_col = "Order_Item_Quantity" if "Order_Item_Quantity" in df.columns else None
    profit_col = "Order_Profit_Per_Order" if "Order_Profit_Per_Order" in df.columns else None
    sales_col = "Sales" if "Sales" in df.columns else None
    if not qty_col or not profit_col:
        return []

    grouped = (
        df
        .groupby("Order_Region", as_index=False)
        .agg(
            total_quantity=(qty_col, "sum"),
            avg_profit_per_order=(profit_col, "mean"),
            total_sales=(sales_col, "sum") if sales_col else (qty_col, "count"),
            order_count=(qty_col, "count"),
        )
    )

    if grouped.empty:
        return []

    grouped = grouped.sort_values(
        ["total_quantity", "avg_profit_per_order"],
        ascending=[False, True],
    ).head(10)

    rows = []
    for rank, (_, row) in enumerate(grouped.iterrows(), start=1):
        rows.append({
            "rank": rank,
            "region": str(row["Order_Region"]),
            "total_quantity": int(row["total_quantity"]),
            "avg_profit_per_order": round(float(row["avg_profit_per_order"]), 2),
            "total_sales": round(float(row["total_sales"]), 2),
            "order_count": int(row["order_count"]),
        })
    return rows


def _get_top_regions_overview(df: pd.DataFrame) -> list:
    """Return top regions by overall demand score across the full dataset."""
    if "Order_Region" not in df.columns:
        return []

    feat_df = build_producer_features(df.copy())
    if "demand_score" not in feat_df.columns:
        return []

    region_agg = feat_df.groupby("Order_Region", as_index=False).agg(
        demand_score=("demand_score", "mean"),
        total_sales=("Sales", "sum") if "Sales" in feat_df.columns else ("demand_score", "count"),
        order_count=("demand_score", "count"),
    ).sort_values("demand_score", ascending=False).head(10)

    all_scores = region_agg["demand_score"]
    low_t = float(all_scores.quantile(0.33)) if len(all_scores) >= 3 else 0.33
    high_t = float(all_scores.quantile(0.67)) if len(all_scores) >= 3 else 0.67

    rows = []
    for _, row in region_agg.iterrows():
        rows.append({
            "region": row["Order_Region"],
            "demand_score": round(float(row["demand_score"]), 3),
            "demand_zone": _classify_demand_zone(row["demand_score"], low_t, high_t),
            "total_sales": round(float(row["total_sales"]), 2),
            "order_count": int(row["order_count"]),
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
    heatmap = _build_region_category_heatmap(df, product_col, product)
    high_demand_low_profit_top10 = _get_high_demand_low_profit_regions(df, product_col, product)

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
        "heatmap":       heatmap,
        "high_demand_low_profit_top10": high_demand_low_profit_top10,
        "data_source":   "city-matched" if found else "product-only",
        "message":       None if found else f"No exact city match — showing product-level data for '{product}'.",
    }


def predict_overview(df: pd.DataFrame) -> dict:
    """Producer overview analytics for full dataset (no product/city inputs)."""
    return {
        "heatmap": _build_region_category_heatmap_overview(df),
        "high_demand_low_profit_top10": _get_high_demand_low_profit_regions_overview(df),
        "top_regions": _get_top_regions_overview(df),
    }
