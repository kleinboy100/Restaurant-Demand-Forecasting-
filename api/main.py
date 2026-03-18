import os
import logging
import pandas as pd
import requests
import holidays
from datetime import datetime, timedelta, date, timezone
from typing import Optional, List, Dict
from prophet import Prophet
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KOTAai Advanced Forecasting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

supabase: Optional[Client] = None

# Models
class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[float] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

# ---------------------------------------------------------------------------
# FORECASTING ENGINE
# ---------------------------------------------------------------------------
def get_weather_impact(days: int = 7) -> Dict[str, float]:
    try:
        # Klerksdorp coordinates
        url = "https://api.open-meteo.com/v1/forecast?latitude=-26.85&longitude=26.66&daily=precipitation_probability_max&forecast_days=7&timezone=Africa/Johannesburg"
        data = requests.get(url, timeout=5).json()
        impacts = {}
        for i, d in enumerate(data["daily"]["time"]):
            prob = data["daily"]["precipitation_probability_max"][i]
            impacts[d] = 1.0 if prob < 20 else (0.85 if prob < 60 else 0.7)
        return impacts
    except:
        return { (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"): 1.0 for i in range(days) }

def get_forecast(ingredient_name: str) -> float:
    try:
        # 1. Fetch data from Supabase
        res = supabase.table("order_items").select("order_id, quantity").ilike("item_name", f"%{ingredient_name}%").execute()
        if not res.data: return 50.0 # Fallback
        
        order_ids = [r["order_id"] for r in res.data]
        orders = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        
        df = pd.merge(pd.DataFrame(res.data), pd.DataFrame(orders.data).rename(columns={"id": "order_id"}), on="order_id")
        df["ds"] = pd.to_datetime(df["created_at"]).dt.date
        df = df.groupby("ds")["quantity"].sum().reset_index().rename(columns={"quantity": "y"})
        
        if len(df) < 5: return 50.0

        # 2. Train Prophet
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True)
        model.add_country_holidays(country_name="ZA")
        model.fit(df)
        
        # 3. Forecast with Weather Impact
        future = model.make_future_dataframe(periods=7)
        weather = get_weather_impact()
        future["impact"] = future["ds"].dt.strftime("%Y-%m-%d").map(weather).fillna(1.0)
        
        forecast = model.predict(future)
        return float((forecast["yhat"].tail(7) * forecast["impact"].tail(7)).sum())
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        return 50.0

# ---------------------------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global supabase
    supabase = create_client(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_ANON_KEY", ""))

@app.post("/api/dashboard")
async def get_dashboard(request: DashboardRequest):
    results = []
    for item in request.items:
        weekly_demand = get_forecast(item.item_name)
        stock = item.current_stock or 0.0
        daily_d = weekly_demand / 7
        days_left = stock / daily_d if daily_d > 0 else 999
        
        results.append({
            "item_name": item.item_name,
            "current_stock": round(stock, 1),
            "weekly_demand": round(weekly_demand, 1),
            "days_left": round(days_left, 1),
            "recommended_order": max(0, (weekly_demand * 1.5) - stock),
            "urgency": "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW"
        })
    
    return {"items": sorted(results, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["urgency"]])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
