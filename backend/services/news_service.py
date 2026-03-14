"""
epoch/backend/services/news_service.py
----------------------------------------
Fetches supply-chain-relevant news headlines.
Uses NewsAPI (configure key in config/config.yaml).

STATUS: STUB — returns mock headlines until API key is configured.
"""

import os
import sys
import requests
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOCK_NEWS = [
    {"title": "Port congestion at Singapore causes 3-day delays",      "market": "Pacific Asia", "impact": "High",   "url": "#"},
    {"title": "EU trade policy changes affect cross-border shipments",  "market": "Europe",       "impact": "Medium", "url": "#"},
    {"title": "LATAM logistics firms expand cold-chain capacity",       "market": "LATAM",        "impact": "Low",    "url": "#"},
    {"title": "USCA freight rates stabilise after Q3 surge",            "market": "USCA",         "impact": "Low",    "url": "#"},
    {"title": "African road infrastructure investment drives efficiency","market": "Africa",       "impact": "Medium", "url": "#"},
    {"title": "Global fuel costs rise — air freight premiums expected", "market": "Global",       "impact": "High",   "url": "#"},
    {"title": "Semiconductor shortage eases, electronics supply up",    "market": "Pacific Asia", "impact": "Medium", "url": "#"},
    {"title": "Red Sea disruptions redirect shipping lanes",            "market": "Europe",       "impact": "High",   "url": "#"},
]


def _get_api_key() -> str | None:
    try:
        import yaml
        config_path = os.path.join(ROOT, "config", "config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("apis", {}).get("newsapi_key")
    except Exception:
        return None


def _fetch_live(api_key: str, market: str = None) -> list:
    query = "supply chain logistics shipping"
    if market and market != "Global":
        query += f" {market}"
    from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    r = requests.get(url, params={
        "q": query, "from": from_date, "sortBy": "relevancy",
        "language": "en", "pageSize": 10, "apiKey": api_key
    }, timeout=5)
    r.raise_for_status()
    articles = r.json().get("articles", [])
    return [{"title": a["title"], "market": market or "Global", "impact": "Medium", "url": a["url"]} for a in articles]


def fetch_news(market: str = None) -> dict:
    """
    Returns supply chain news signals, optionally filtered by market.
    Falls back to mock data if no API key is set.
    """
    api_key = _get_api_key()

    if api_key:
        try:
            articles = _fetch_live(api_key, market)
            return {"articles": articles, "source": "live"}
        except Exception:
            pass

    # Mock fallback
    articles = MOCK_NEWS if not market else [n for n in MOCK_NEWS if n["market"] in (market, "Global")]
    return {"articles": articles, "source": "mock"}
