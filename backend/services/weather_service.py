"""
epoch/backend/services/weather_service.py
------------------------------------------
Fetches weather data for major supply chain hub cities.
Uses OpenWeatherMap API (configure key in config/config.yaml).

STATUS: STUB — returns mock data until API key is configured.
"""

import os
import sys
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Supply chain hub cities mapped to markets
HUB_CITIES = {
    "USCA":         [("Los Angeles", "US"), ("New York", "US"), ("Chicago", "US")],
    "Europe":       [("Rotterdam", "NL"),  ("Hamburg", "DE"),  ("London", "GB")],
    "LATAM":        [("São Paulo", "BR"),  ("Mexico City", "MX")],
    "Pacific Asia": [("Shanghai", "CN"),   ("Singapore", "SG"), ("Tokyo", "JP")],
    "Africa":       [("Lagos", "NG"),      ("Nairobi", "KE"),   ("Cairo", "EG")],
}

MOCK_WEATHER = {
    "Los Angeles":  {"temp_c": 22, "condition": "Clear",  "disruption_risk": "Low"},
    "New York":     {"temp_c": 10, "condition": "Cloudy", "disruption_risk": "Low"},
    "Chicago":      {"temp_c": 5,  "condition": "Windy",  "disruption_risk": "Medium"},
    "Rotterdam":    {"temp_c": 8,  "condition": "Rain",   "disruption_risk": "Medium"},
    "Hamburg":      {"temp_c": 6,  "condition": "Rain",   "disruption_risk": "Medium"},
    "London":       {"temp_c": 9,  "condition": "Fog",    "disruption_risk": "Low"},
    "São Paulo":    {"temp_c": 28, "condition": "Humid",  "disruption_risk": "Low"},
    "Mexico City":  {"temp_c": 18, "condition": "Clear",  "disruption_risk": "Low"},
    "Shanghai":     {"temp_c": 15, "condition": "Haze",   "disruption_risk": "Medium"},
    "Singapore":    {"temp_c": 30, "condition": "Storms", "disruption_risk": "High"},
    "Tokyo":        {"temp_c": 12, "condition": "Clear",  "disruption_risk": "Low"},
    "Lagos":        {"temp_c": 32, "condition": "Clear",  "disruption_risk": "Low"},
    "Nairobi":      {"temp_c": 20, "condition": "Clear",  "disruption_risk": "Low"},
    "Cairo":        {"temp_c": 25, "condition": "Clear",  "disruption_risk": "Low"},
}


def _get_api_key() -> str | None:
    try:
        import yaml
        config_path = os.path.join(ROOT, "config", "config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("apis", {}).get("openweather_key")
    except Exception:
        return None


def _fetch_live(city: str, country_code: str, api_key: str) -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    r = requests.get(url, params={"q": f"{city},{country_code}", "appid": api_key, "units": "metric"}, timeout=5)
    r.raise_for_status()
    data = r.json()
    temp = data["main"]["temp"]
    condition = data["weather"][0]["main"]
    risk = "High" if condition in ["Thunderstorm","Snow","Tornado"] else "Medium" if condition in ["Rain","Drizzle","Fog","Haze"] else "Low"
    return {"temp_c": round(temp,1), "condition": condition, "disruption_risk": risk}


def fetch_weather(region: str = None) -> dict:
    """
    Returns weather data per market hub city.
    Falls back to mock data if no API key is set.
    """
    api_key  = _get_api_key()
    markets  = HUB_CITIES if not region else {region: HUB_CITIES.get(region, [])}
    result   = {}

    for market, cities in markets.items():
        result[market] = []
        for city, cc in cities:
            if api_key:
                try:
                    data = _fetch_live(city, cc, api_key)
                except Exception:
                    data = MOCK_WEATHER.get(city, {"temp_c": None, "condition": "Unknown", "disruption_risk": "Unknown"})
            else:
                data = MOCK_WEATHER.get(city, {"temp_c": None, "condition": "Unknown", "disruption_risk": "Unknown"})
            result[market].append({"city": city, **data})

    return {"weather": result, "source": "live" if api_key else "mock"}
