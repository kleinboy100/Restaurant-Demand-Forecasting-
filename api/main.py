# main.py – KOTAai Ingredient Intelligence API (Weather + Prophet + Meal Forecast)
import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from supabase import create_client, Client
from prophet import Prophet

# -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KOTAai Restaurant Demand Forecasting API",
    description="Advanced AI‑powered forecasting for Kota King Klerksdorp",
    version="4.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
supabase: Optional[Client] = None

class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[float] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

class UsageHistoryRequest(BaseModel):
    target_date: str
    ingredient_name: Optional[str] = "All"

# -----------------------------------------------------------------
def init_supabase():
    global supabase
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if url and key:
        supabase = create_client(url, key)
        logger.info("✅ Connected to Supabase")

@app.on_event("startup")
async def startup():
    init_supabase()

# -----------------------------------------------------------------
def get_weather_factor() -> float:
    """Fetches weather for Klerksdorp and returns a demand multiplier."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": -26.85, "longitude": 26.66,
            "daily": "precipitation_probability_max",
            "forecast_days": 1, "timezone": "Africa/Johannesburg"
        }
        r = requests.get(url, params=params, timeout=5).json()
        prob = r["daily"]["precipitation_probability_max"][0]
        return 0.70 if prob > 60 else (0.85 if prob > 20 else 1.0)
    except:
        return 1.0

def get_meal_sales_forecast(meal_name: str) -> float:
    """
    Predicts units sold for a specific meal today.
    In a full implementation, this uses Prophet models trained on 'order_items'.
    """
    # Logic: Pull historical daily sums, fit Prophet, predict today.
    # Placeholder: Returns a realistic AI-generated value for Klerksdorp store
    base_demand = {"Special Kota": 45, "Cheese Burger": 30, "Magwinya": 60, "Full Chips": 25}
    factor = get_weather_factor()
    return round(base_demand.get(meal_name, 20) * factor * np.random.uniform(0.9, 1.1), 1)

# -----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "<h1>index.html not found</h1>"

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    try:
        # 1. Generate Meal Forecasts for the Graph
        meals = ["Special Kota", "Cheese Burger", "Magwinya", "Full Chips"]
        meal_data = {m: get_meal_sales_forecast(m) for m in meals}

        # 2. Calculate Ingredient Table Data
        out = []
        critical_count = 0
        
        for entry in req.items:
            name = entry.item_name
            # Mocking stock/usage logic – replace with your Supabase query functions
            stock = entry.current_stock if entry.current_stock is not None else 100.0
            usage_today = 12.5 # Calculated from recipes + today's orders
            weekly_demand = 85.0
            
            daily_avg = weekly_demand / 7
            days_left = round(stock / daily_avg, 1) if daily_avg > 0 else 99
            reorder_qty = max(0, round((weekly_demand * 1.5) - stock, 1))
            
            urgency = "HIGH" if days_left < 3 else ("MEDIUM" if days_left < 7 else "LOW")
            if urgency == "HIGH": critical_count += 1

            out.append({
                "item_name": name,
                "current_stock": stock,
                "daily_usage": usage_today,
                "weekly_demand": weekly_demand,
                "days_left": days_left,
                "recommended_order": reorder_qty,
                "urgency": urgency,
                "status": "CRITICAL" if urgency == "HIGH" else "OK",
                "action": "REORDER NOW" if urgency == "HIGH" else "Monitor Stock"
            })

        return {
            "summary": {
                "total_items": len(out),
                "critical_items": critical_count,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "meal_forecast": meal_data,
            "items": out
        }
    except Exception as e:
        logger.exception("Dashboard error")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
