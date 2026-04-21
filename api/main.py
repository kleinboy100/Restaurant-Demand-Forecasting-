import os
import logging
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
from prophet import Prophet
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from supabase import create_client, Client
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KOTAai Ingredient Intelligence", version="4.8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Optional[Client] = None

def init_supabase():
    global supabase
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if url and key:
        supabase = create_client(url, key)

@app.on_event("startup")
async def startup():
    init_supabase()

class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[float] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

def get_weather_impact():
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast?latitude=-26.85&longitude=26.66&daily=precipitation_probability_max&forecast_days=1&timezone=Africa/Johannesburg", timeout=3)
        prob = r.json()["daily"]["precipitation_probability_max"][0]
        return 0.75 if prob > 50 else 1.0
    except:
        return 1.0

def run_safe_forecast(name: str, days: int = 7):
    if not supabase: return None
    try:
        res = supabase.table("order_items").select("order_id, quantity").ilike("item_name", f"%{name}%").execute()
        if not res.data: return None
        
        order_ids = [it["order_id"] for it in res.data]
        orders = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        if not orders.data: return None
        
        df = pd.merge(pd.DataFrame(res.data), pd.DataFrame(orders.data).rename(columns={"id": "order_id"}), on="order_id")
        df["ds"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None).dt.date
        daily = df.groupby("ds")["quantity"].sum().reset_index().rename(columns={"ds": "ds", "quantity": "y"})
        
        if len(daily) < 2: return None
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(daily)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        return forecast.tail(days)
    except Exception as e:
        logger.error(f"Forecast Error [{name}]: {e}")
        return None

@app.post("/api/forecast-meals")
async def forecast_meals():
    # Key representative items for the chart
    top_meals = ["Original Dagwood", "Laprovance", "Tower of Terror", "N12_3 Loaf", "Combo 45", "Ext 10"]
    impact = get_weather_impact()
    results = {}
    for meal in top_meals:
        f = run_safe_forecast(meal, 1)
        val = float(f["yhat"].iloc[0]) if f is not None else 10.0 
        results[meal] = round(val * impact, 1)
    return results

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    out = []
    total_rec = 0.0
    for entry in req.items:
        name = entry.item_name
        # 1. Fetch stock directly from database
        stock = 0.0
        if supabase:
            stock_res = supabase.table("ingredient_stock").select("current_stock").ilike("ingredient_name", f"%{name}%").execute()
            if stock_res.data:
                stock = float(stock_res.data[0]["current_stock"])
        
        # 2. Run forecast for demand
        f = run_safe_forecast(name, 7)
        weekly = float(f["yhat"].sum()) if f is not None else 50.0
        daily = weekly / 7.0
        days_left = stock / daily if daily > 0 else 99.0
        
        # 3. Calculate Recommendation (Buffer logic: 1.5x weekly demand minus current stock)
        recommend = max(0.0, (weekly * 1.5) - stock)
        total_rec += recommend
        
        out.append({
            "item_name": name,
            "current_stock": round(stock, 1),
            "weekly_demand": round(weekly, 1),
            "days_left": round(days_left, 1),
            "recommended_order": round(recommend, 1),
            "urgency": "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW",
            "status": "CRITICAL" if days_left < 3 else "OK",
            "action": "REORDER NOW" if days_left < 3 else "Monitor stock"
        })
    
    return {
        "summary": {
            "total_items": len(out),
            "critical_items": len([x for x in out if x["urgency"] == "HIGH"]),
            "total_recommended": round(total_rec, 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "items": out
    }

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(file_path):
        with open(file_path, "r") as f: return f.read()
    return "<h1>KOTAai Active</h1><p>index.html missing.</p>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
