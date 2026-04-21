# main.py - KOTAai Intelligence with PROPHET Integration
import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from supabase import create_client, Client
from prophet import Prophet

# --- Setup & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KOTAai Ingredient Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Supabase Initialization ---
supabase: Optional[Client] = None

def init_supabase():
    global supabase
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if url and key:
        supabase = create_client(url, key)
        logger.info("✅ Supabase connection successful.")

@app.on_event("startup")
async def startup():
    init_supabase()

# --- Models ---
class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[float] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

# --- PROPHET FORECASTING ENGINE ---
def get_prophet_forecast(item_name: str) -> float:
    """
    Fetches historical sales from Supabase, trains a Prophet model, 
    and predicts units sold for today.
    """
    if not supabase:
        return 20.0 # Fallback if DB is not connected

    try:
        # 1. Fetch historical sales for this meal
        # Assuming table 'order_items' has 'item_name', 'quantity', and 'created_at'
        res = supabase.table("order_items") \
            .select("quantity, created_at") \
            .ilike("item_name", f"%{item_name}%") \
            .execute()

        if not res.data or len(res.data) < 5:
            return 15.0 # Fallback if insufficient data for training

        # 2. Prepare Dataframe for Prophet
        df = pd.DataFrame(res.data)
        df['ds'] = pd.to_datetime(df['created_at']).dt.date
        df = df.groupby('ds')['quantity'].sum().reset_index()
        df.columns = ['ds', 'y']

        # 3. Train Model
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df)

        # 4. Predict Today
        future = m.make_future_dataframe(periods=1)
        forecast = m.predict(future)
        
        prediction = forecast.iloc[-1]['yhat']
        return max(0, round(prediction, 1))

    except Exception as e:
        logger.error(f"Prophet training failed for {item_name}: {e}")
        return 10.0

def get_weather_impact() -> float:
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=-26.85&longitude=26.66&daily=precipitation_probability_max&forecast_days=1"
        r = requests.get(url, timeout=3).json()
        prob = r["daily"]["precipitation_probability_max"][0]
        return 0.75 if prob > 50 else 1.0
    except:
        return 1.0

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    base_path = Path(__file__).parent
    file_path = base_path / "index.html"
    if file_path.exists():
        return FileResponse(file_path)
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    try:
        weather_factor = get_weather_impact()
        
        # 1. Prophet Meal Sales Forecast (For the Bar Graph)
        # This replaces the simulated logic with actual Prophet predictions
        meals = ["Special Kota", "Cheese Burger", "Magwinya", "Full Chips"]
        meal_forecast = {}
        
        for meal in meals:
            p_val = get_prophet_forecast(meal)
            meal_forecast[meal] = round(p_val * weather_factor, 1)

        # 2. Detailed Ingredient Intelligence (For the Table)
        output_items = []
        critical_count = 0
        
        for entry in req.items:
            stock = entry.current_stock if entry.current_stock is not None else 0.0
            
            # Using fixed logic for usage (or pull from your separate usage function)
            daily_usage = 12.0 
            weekly_demand = daily_usage * 7
            days_left = round(stock / daily_usage, 1) if daily_usage > 0 else 99
            reorder_qty = max(0, round((weekly_demand * 1.5) - stock, 1))
            
            urgency = "HIGH" if days_left < 3 else ("MEDIUM" if days_left < 7 else "LOW")
            if urgency == "HIGH": critical_count += 1

            output_items.append({
                "item_name": entry.item_name,
                "current_stock": stock,
                "daily_usage": daily_usage,
                "weekly_demand": weekly_demand,
                "days_left": days_left,
                "recommended_order": reorder_qty,
                "urgency": urgency,
                "status": "CRITICAL" if urgency == "HIGH" else "STABLE",
                "action": "ORDER NOW" if urgency == "HIGH" else "Monitor"
            })

        return {
            "summary": {
                "total_items": len(output_items),
                "critical_items": critical_count,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S SAST")
            },
            "meal_forecast": meal_forecast,
            "items": output_items
        }
    except Exception as e:
        logger.exception("Dashboard API Processing Error")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
