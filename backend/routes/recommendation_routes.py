"""
epoch/backend/routes/recommendation_routes.py
----------------------------------------------
Endpoints:
  POST /api/recommend/shipping      → Consumer: best shipping mode recommendation
  GET  /api/recommend/weather       → Weather context for supply decisions
  GET  /api/recommend/news          → News signals affecting supply chain
"""

import os
import sys
from flask import Blueprint, request, jsonify

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from utils.preprocessing import load_and_clean_csv

recommendation_bp = Blueprint("recommendation", __name__)


# ── Consumer: Shipping Recommendation ────────────────────────────────────────
@recommendation_bp.route("/shipping", methods=["POST"])
def recommend_shipping():
    """
    Accepts a CSV upload, recommends optimal shipping mode per order.
    Model: shipping_recommendation.py
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    from flask import current_app
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    filepath = os.path.join(upload_dir, f.filename)
    f.save(filepath)

    try:
        from models.consumer_models.shipping_recommendation import run_shipping_recommendation
        df = load_and_clean_csv(filepath)
        result = run_shipping_recommendation(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Weather Context ────────────────────────────────────────────────────────────
@recommendation_bp.route("/weather", methods=["GET"])
def get_weather():
    """
    Returns weather signals for major supply chain hubs.
    Query param: ?region=USCA  (optional filter)
    """
    region = request.args.get("region", None)
    try:
        from services.weather_service import fetch_weather
        data = fetch_weather(region=region)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── News Signals ───────────────────────────────────────────────────────────────
@recommendation_bp.route("/news", methods=["GET"])
def get_news():
    """
    Returns supply-chain-relevant news signals.
    Query param: ?market=Europe  (optional filter)
    """
    market = request.args.get("market", None)
    try:
        from services.news_service import fetch_news
        data = fetch_news(market=market)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
