"""
epoch/backend/app.py
--------------------
Main Flask application entry point.
Registers all route blueprints and serves the frontend.
"""

import os
import yaml
from flask import Flask, send_from_directory
from flask_cors import CORS

from routes.prediction_routes import prediction_bp
from routes.recommendation_routes import recommendation_bp

# ── Load config ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ── App setup ─────────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit
app.config["UPLOAD_FOLDER"]      = os.path.join(BASE_DIR, "data", "raw")
app.config["APP_CONFIG"]         = config

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Register blueprints ───────────────────────────────────────────────────────
app.register_blueprint(prediction_bp,      url_prefix="/api/predict")
app.register_blueprint(recommendation_bp,  url_prefix="/api/recommend")

# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/producer")
def producer():
    return send_from_directory(FRONTEND_DIR, "producer.html")

@app.route("/consumer")
def consumer():
    return send_from_directory(FRONTEND_DIR, "consumer.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return {"status": "ok", "version": config.get("app", {}).get("version", "1.0.0")}

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = config.get("app", {}).get("port", 5000)
    debug = config.get("app", {}).get("debug", True)
    print(f"\n  epoch backend running at http://localhost:{port}\n")
    app.run(debug=debug, port=port)
