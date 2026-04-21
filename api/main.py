# ----------------------------------------------------------
# main.py – KOTAai Ingredient Intelligence API (Enhanced)
# --------------------------------------------------------------
import os
import logging
import pickle
from datetime import datetime, timedelta, date, timezone
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
from prophet import Prophet
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from supabase import create_client, Client
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KOTAai Restaurant Demand Forecasting API",
    version="4.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Optional[Client] = None
model_cache: Dict[str, Prophet] = {}

class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[int] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

class UsageHistoryRequest(BaseModel):
    target_date: str
    ingredient_name: Optional[str] = "All"

FALLBACK_DEMANDS = {
    "chips": 500.0, "cheese": 50.0, "russian": 12.0, "lettuce": 8.0,
    "bread": 25.0, "tomato": 10.0, "atchaar": 5.0, "vienna": 15.0
}

def init_supabase() -> bool:
    global supabase
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        if not supabase_url or not supabase_key: return False
        supabase = create_client(supabase_url, supabase_key)
        return True
    except Exception: return False

@app.on_event("startup")
async def startup():
    init_supabase()

def get_klerksdorp_weather(days_ahead: int = 7) -> Dict[str, float]:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": -26.85, "longitude": 26.66, "daily": "precipitation_probability_max", "forecast_days": days_ahead, "timezone": "Africa/Johannesburg"}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        return {iso: (0.7 if prob > 60 else 0.85 if prob > 20 else 1.0) for i, (iso, prob) in enumerate(zip(data["daily"]["time"], data["daily"]["precipitation_probability_max"]))}
    except: return {}

def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    if not supabase: return pd.DataFrame()
    try:
        items = supabase.table("order_items").select("order_id, quantity").ilike("item_name", f"%{item_name}%").execute()
        if not items.data: return pd.DataFrame()
        order_ids = [it["order_id"] for it in items.data]
        orders = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        if not orders.data: return pd.DataFrame()
        df = pd.merge(pd.DataFrame(items.data), pd.DataFrame(orders.data).rename(columns={"id": "order_id"}), on="order_id")
        df["ds"] = pd.to_datetime(df["created_at"]).dt.date
        return df.groupby("ds")["quantity"].sum().reset_index().rename(columns={"ds": "ds", "quantity": "y"})
    except: return pd.DataFrame()

def get_forecast(name: str, days_ahead: int = 7):
    sales_df = get_sales_from_order_items(name)
    if sales_df.empty or len(sales_df) < 2: return None
    sales_df['ds'] = pd.to_datetime(sales_df['ds'])
    m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name="ZA")
    m.fit(sales_df)
    future = m.make_future_dataframe(periods=days_ahead)
    forecast = m.predict(future)
    weather = get_klerksdorp_weather(days_ahead)
    forecast["yhat"] = forecast.apply(lambda x: x["yhat"] * weather.get(x["ds"].strftime("%Y-%m-%d"), 1.0), axis=1)
    return forecast[forecast["ds"] > pd.to_datetime(sales_df["ds"].max())]

@app.post("/api/forecast-meals")
async def forecast_meals():
    # Define your main menu items here
    meals = ["Kota King", "Special Kota", "Classic Burger", "Russian & Chips"]
    results = {}
    for meal in meals:
        f = get_forecast(meal, 1)
        results[meal] = round(float(f["yhat"].iloc[0]), 1) if f is not None else 10.0
    return results

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    out = []
    weather = get_klerksdorp_weather(7)
    for entry in req.items:
        name = entry.item_name
        sales_df = get_sales_from_order_items(name)
        weekly_demand = 70.0 # Default
        if not sales_df.empty:
            f = get_forecast(name, 7)
            if f is not None: weekly_demand = float(f["yhat"].sum())
        
        stock = entry.current_stock or 0
        daily = weekly_demand / 7.0
        days_left = stock / daily if daily > 0 else 99
        
        out.append({
            "item_name": name,
            "current_stock": round(stock, 1),
            "daily_usage": round(daily, 1),
            "weekly_demand": round(weekly_demand, 1),
            "days_left": round(days_left, 1),
            "urgency": "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW",
            "status": "CRITICAL" if days_left < 3 else "OK",
            "action": "REORDER" if days_left < 3 else "Monitor"
        })
    return {"summary": {"total_items": len(out), "critical_items": len([x for x in out if x["urgency"]=="HIGH"]), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, "items": out}

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("index.html", "r") as f: return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
