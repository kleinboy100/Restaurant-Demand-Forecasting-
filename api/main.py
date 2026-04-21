import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import pandas as pd
from prophet import Prophet
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from supabase import create_client, Client
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KOTAai Ingredient Intelligence", version="4.6.0")

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
    else:
        logger.warning("Supabase credentials not found in environment variables.")

@app.on_event("startup")
async def startup():
    init_supabase()

class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[float] = 0

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

def get_weather_impact():
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast?latitude=-26.85&longitude=26.66&daily=precipitation_probability_max&forecast_days=1&timezone=Africa/Johannesburg", timeout=3)
        prob = r.json()["daily"]["precipitation_probability_max"][0]
        return 0.7 if prob > 60 else 0.85 if prob > 20 else 1.0
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return 1.0

def run_safe_forecast(name: str, days: int = 7):
    if not supabase: return None
    try:
        res = supabase.table("order_items").select("order_id, quantity").ilike("item_name", f"%{name}%").execute()
        if not res.data: return None
        
        order_ids = [it["order_id"] for it in res.data]
        orders = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        if not orders.data: return None
        
        df_items = pd.DataFrame(res.data)
        df_orders = pd.DataFrame(orders.data).rename(columns={"id": "order_id"})
        df = pd.merge(df_items, df_orders, on="order_id")
        
        df["ds"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None).dt.date
        daily = df.groupby("ds")["quantity"].sum().reset_index().rename(columns={"ds": "ds", "quantity": "y"})
        
        if len(daily) < 2: return None

        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(daily)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        return forecast.tail(days)
    except Exception as e:
        logger.error(f"Forecast failed for {name}: {e}")
        return None

@app.post("/api/forecast-meals")
async def forecast_meals():
    meals = ["Kota King", "Special Kota", "Classic Burger", "Russian & Chips"]
    impact = get_weather_impact()
    results = {}
    for meal in meals:
        f = run_safe_forecast(meal, 1)
        val = float(f["yhat"].iloc[0]) if f is not None else 15.0 
        results[meal] = round(val * impact, 1)
    return results

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    out = []
    for entry in req.items:
        f = run_safe_forecast(entry.item_name, 7)
        weekly = float(f["yhat"].sum()) if f is not None else 70.0
        daily = weekly / 7.0
        stock = float(entry.current_stock or 0)
        days_left = stock / daily if daily > 0 else 99.0
        
        out.append({
            "item_name": entry.item_name,
            "current_stock": round(stock, 1),
            "days_left": round(days_left, 1),
            "urgency": "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW"
        })
    
    return {
        "summary": {
            "total_items": len(out),
            "critical_items": len([x for x in out if x["urgency"] == "HIGH"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "items": out
    }

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    # Attempt to serve the physical file
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read()
    
    # Fallback HTML in case index.html is missing
    return """
    <html>
        <head><title>KOTAai API Status</title></head>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
            <h1 style="color: #1a365d;">🍛 KOTAai API is Online</h1>
            <p>The dashboard file (index.html) was not found in the root directory.</p>
            <div style="margin-top: 20px; padding: 20px; background: #f7fafc; display: inline-block; border-radius: 10px;">
                <strong>API Endpoints:</strong><br>
                <code>POST /api/dashboard</code><br>
                <code>POST /api/forecast-meals</code>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    # Use environment PORT for cloud providers like Render or Railway
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
