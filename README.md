# epoch — Supply Chain Intelligence Platform

ML-powered supply chain analytics with demand zone clustering, delivery delay prediction, risk scoring, and shipping recommendations.

---

## Project Structure

```
epoch/
├── frontend/           → HTML/CSS/JS dashboard (served by Flask)
├── backend/            → Flask API server + routes + services + utils
├── models/
│   ├── consumer_models/   → delay_prediction, risk_scoring, shipping_recommendation
│   └── producer_models/   → demand_clustering
├── data/               → raw / processed / external CSV files
├── notebooks/          → EDA and experimentation
├── saved_models/       → trained .pkl files (drop here when ready)
├── config/config.yaml  → API keys + model parameters
└── requirements.txt
```

---

## Setup

### 1. Install dependencies
```bash
cd epoch
pip install -r requirements.txt
```

### 2. Configure API keys (optional)
Edit `config/config.yaml`:
```yaml
apis:
  openweather_key: "YOUR_KEY_HERE"
  newsapi_key:     "YOUR_KEY_HERE"
```
Without keys, the app falls back to realistic mock data automatically.

### 3. Run the backend
```bash
cd backend
python app.py
```
Open http://localhost:5000

---

## Plugging In Your Trained Models

Each model file has a clearly marked `── REPLACE THIS FUNCTION ──` block.

### Producer: Demand Clustering
File: `models/producer_models/demand_clustering.py`
```python
# Replace _cluster_regions() with:
model = joblib.load("saved_models/cluster_model.pkl")
features = region_df[["demand_score", "avg_profit", ...]]
labels = model.predict(features)
zone_map = {0: "Low Demand", 1: "Emerging Market", 2: "High Demand"}
return pd.Series([zone_map[l] for l in labels])
```

### Consumer: Delay Prediction
File: `models/consumer_models/delay_prediction.py`
```python
# Replace _predict_delay() with:
model = joblib.load("saved_models/delay_model.pkl")
X = df[FEATURE_COLS]
df["predicted_delay"]   = model.predict(X)
df["delay_confidence"]  = model.predict_proba(X)[:, 1]
```

### Consumer: Risk Scoring
File: `models/consumer_models/risk_scoring_model.py`
```python
# Replace _score_risk() with:
model = joblib.load("saved_models/risk_model.pkl")
X = df[FEATURE_COLS]
df["risk_score"] = model.predict_proba(X)[:, 1] * 100
```

### Consumer: Shipping Recommendation
File: `models/consumer_models/shipping_recommendation.py`
```python
# Replace _recommend_mode() with:
model = joblib.load("saved_models/shipping_model.pkl")
X = df[FEATURE_COLS]
df["recommended_mode"] = model.predict(X)
```

---

## API Endpoints

| Method | Endpoint                    | Description                        |
|--------|-----------------------------|------------------------------------|
| POST   | `/api/predict/demand`       | Producer: demand zone clustering   |
| POST   | `/api/predict/delay`        | Consumer: delay prediction         |
| POST   | `/api/predict/risk`         | Consumer: risk scoring             |
| POST   | `/api/recommend/shipping`   | Consumer: shipping recommendation  |
| GET    | `/api/recommend/weather`    | Weather signals (by region)        |
| GET    | `/api/recommend/news`       | News signals (by market)           |
| GET    | `/api/health`               | Health check                       |

All prediction endpoints accept `multipart/form-data` with a `file` field containing the CSV.

---

## CSV Format

The app expects the standard Supply Chain dataset columns. Key columns used:

| Column                   | Used by              |
|--------------------------|----------------------|
| `Order Region`           | All models           |
| `Market`                 | All models           |
| `Sales`                  | Demand clustering    |
| `Order Item Quantity`    | Demand clustering    |
| `Order Profit Per Order` | Demand + risk        |
| `Late_delivery_risk`     | Delay + risk         |
| `Days for shipping (real)` | Delay prediction   |
| `Days for shipment (scheduled)` | Delay + shipping |
| `Shipping Mode`          | Shipping rec.        |

Column names are normalised automatically — spacing and capitalisation variations are handled.

---

## Development Notes

- The preprocessing pipeline (`backend/utils/preprocessing.py`) strips PII columns automatically (email, name, password, street).
- Feature engineering is shared via `backend/utils/feature_engineering.py` — add new features here so all models benefit.
- Models run as stubs (heuristic logic) until `.pkl` files are placed in `saved_models/`. No code changes needed — just drop the files and uncomment the loader lines.
