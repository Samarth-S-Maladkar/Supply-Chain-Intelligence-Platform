"""
epoch/backend/routes/prediction_routes.py
------------------------------------------
Endpoints:
  GET  /api/predict/options/products   → unique product names from dataset
  GET  /api/predict/options/cities     → unique destination cities
  GET  /api/predict/options/pincodes   → unique customer zipcodes
  POST /api/predict/demand             → Producer: product + city → demand insights
  POST /api/predict/consumer           → Consumer: product + pincode → delivery insights
"""

import os
import sys
from flask import Blueprint, request, jsonify

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

prediction_bp = Blueprint("prediction", __name__)

# ── Dataset cache (loaded once on first request) ──────────────────────────────
_dataset_cache = None

def _get_dataset():
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    import pandas as pd
    from backend.utils.preprocessing import load_and_clean_csv

    for folder in ["data/processed", "data/raw"]:
        folder_path = os.path.join(ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".csv"):
                try:
                    df = load_and_clean_csv(os.path.join(folder_path, fname))
                    _dataset_cache = df
                    return df
                except Exception:
                    continue

    raise FileNotFoundError(
        "No CSV found in data/processed/ or data/raw/. "
        "Place your dataset CSV there and restart."
    )


# ── Dropdown options ──────────────────────────────────────────────────────────

@prediction_bp.route("/options/products", methods=["GET"])
def get_products():
    try:
        df = _get_dataset()
        col = next((c for c in ["Product_Name","Category_Name"] if c in df.columns), None)
        products = sorted(df[col].dropna().unique().tolist()) if col else []
        return jsonify({"products": products})
    except FileNotFoundError as e:
        return jsonify({"products": [], "warning": str(e)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_bp.route("/options/cities", methods=["GET"])
def get_cities():
    try:
        df = _get_dataset()
        col = next((c for c in ["Order_City","Customer_City"] if c in df.columns), None)
        cities = sorted(df[col].dropna().unique().tolist()) if col else []
        return jsonify({"cities": cities})
    except FileNotFoundError as e:
        return jsonify({"cities": [], "warning": str(e)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@prediction_bp.route("/options/pincodes", methods=["GET"])
def get_pincodes():
    try:
        df = _get_dataset()
        col = "Customer_Zipcode" if "Customer_Zipcode" in df.columns else None
        if not col:
            return jsonify({"pincodes": []})
        raw = df[col].dropna().unique()
        pincodes = sorted(set(
            str(int(float(p))) for p in raw
            if str(p).strip() not in ("", "nan")
        ))
        return jsonify({"pincodes": pincodes})
    except FileNotFoundError as e:
        return jsonify({"pincodes": [], "warning": str(e)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Producer: demand insight ──────────────────────────────────────────────────

@prediction_bp.route("/demand", methods=["POST"])
def predict_demand():
    body    = request.get_json(silent=True) or {}
    product = body.get("product","").strip()
    city    = body.get("city","").strip()
    if not product or not city:
        return jsonify({"error": "Both 'product' and 'city' are required."}), 400
    try:
        df = _get_dataset()
        from models.producer_models.demand_clustering import predict_for_product_city
        return jsonify(predict_for_product_city(df, product, city))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Consumer: delivery insight ────────────────────────────────────────────────

@prediction_bp.route("/consumer", methods=["POST"])
def predict_consumer():
    body    = request.get_json(silent=True) or {}
    product = body.get("product","").strip()
    pincode = body.get("pincode","").strip()
    if not product or not pincode:
        return jsonify({"error": "Both 'product' and 'pincode' are required."}), 400
    try:
        df = _get_dataset()
        from models.consumer_models.consumer_insights import predict_for_product_pincode
        return jsonify(predict_for_product_pincode(df, product, pincode))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
