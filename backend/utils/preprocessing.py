"""
epoch/backend/utils/preprocessing.py
--------------------------------------
Shared CSV loading and column normalisation used by all model routes.
"""

import pandas as pd
import numpy as np

# Canonical column name map (lowercase key → display name)
COLUMN_MAP = {
    "type":                       "Type",
    "days for shipping (real)":   "Days_for_shipping_real",
    "days for shipment (scheduled)": "Days_for_shipment_scheduled",
    "benefit per order":          "Benefit_per_order",
    "sales per customer":         "Sales_per_customer",
    "delivery status":            "Delivery_Status",
    "late_delivery_risk":         "Late_delivery_risk",
    "category id":                "Category_Id",
    "category name":              "Category_Name",
    "customer city":              "Customer_City",
    "customer country":           "Customer_Country",
    "customer id":                "Customer_Id",
    "customer segment":           "Customer_Segment",
    "customer state":             "Customer_State",
    "customer zipcode":           "Customer_Zipcode",
    "department id":              "Department_Id",
    "department name":            "Department_Name",
    "latitude":                   "Latitude",
    "longitude":                  "Longitude",
    "market":                     "Market",
    "order city":                 "Order_City",
    "order country":              "Order_Country",
    "order customer id":          "Order_Customer_Id",
    "order date (dateorders)":    "Order_Date",
    "order id":                   "Order_Id",
    "order item quantity":        "Order_Item_Quantity",
    "order item discount":        "Order_Item_Discount",
    "order item discount rate":   "Order_Item_Discount_Rate",
    "order item id":              "Order_Item_Id",
    "order item product price":   "Order_Item_Product_Price",
    "order item profit ratio":    "Order_Item_Profit_Ratio",
    "order item total":           "Order_Item_Total",
    "order profit per order":     "Order_Profit_Per_Order",
    "order region":               "Order_Region",
    "order state":                "Order_State",
    "order status":               "Order_Status",
    "product card id":            "Product_Card_Id",
    "product category id":        "Product_Category_Id",
    "product name":               "Product_Name",
    "product price":              "Product_Price",
    "product status":             "Product_Status",
    "shipping date (dateorders)": "Shipping_Date",
    "shipping mode":              "Shipping_Mode",
    "sales":                      "Sales",
}

NUMERIC_COLS = [
    "Days_for_shipping_real", "Days_for_shipment_scheduled",
    "Benefit_per_order", "Sales_per_customer", "Late_delivery_risk",
    "Category_Id", "Customer_Id", "Department_Id",
    "Latitude", "Longitude", "Order_Customer_Id", "Order_Id",
    "Order_Item_Quantity", "Order_Item_Discount", "Order_Item_Discount_Rate",
    "Order_Item_Id", "Order_Item_Product_Price", "Order_Item_Profit_Ratio",
    "Order_Item_Total", "Order_Profit_Per_Order", "Product_Card_Id",
    "Product_Category_Id", "Product_Price", "Product_Status", "Sales",
]

DATE_COLS = ["Order_Date", "Shipping_Date"]


def load_and_clean_csv(filepath: str) -> pd.DataFrame:
    """
    Load CSV, normalise column names, cast types, drop PII columns.
    Returns a clean DataFrame ready for feature engineering or prediction.
    """
    df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")

    # Normalise column names
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in COLUMN_MAP:
            rename[col] = COLUMN_MAP[key]
    df = df.rename(columns=rename)
    df.columns = df.columns.str.strip()

    # Drop PII
    pii = ["Customer_Email", "Customer_Fname", "Customer_Lname",
           "Customer_Password", "Customer_Street", "Product_Description",
           "Product_Image"]
    df = df.drop(columns=[c for c in pii if c in df.columns], errors="ignore")

    # Cast numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse dates
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)

    # Drop fully empty rows
    df = df.dropna(how="all")

    return df


def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def normalise_0_1(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx > mn:
        return (series - mn) / (mx - mn)
    return pd.Series([0.5] * len(series), index=series.index)
