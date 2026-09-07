import os
import logging
from datetime import datetime, timedelta
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

app = FastAPI(title="KOTAai Ingredient Intelligence", version="5.0.0")

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

# Standard Recipe (Bill of Materials) mapping meals to ingredient quantities
RECIPES: Dict[str, Dict[str, float]] = {
    "Original Dagwood": {"Bread": 3, "Bacon": 1, "Polony": 1, "Egg": 1, "Cheese": 1, "Atchar": 1, "Secret Sauce": 1},
    "Laprovance": {"Loaf": 0.25, "Chips": 1, "Vienna": 1, "Polony": 1, "Cheese": 1, "Atchar": 1},
    "Tower of Terror": {"Loaf": 0.5, "Chips": 1, "Burger": 1, "Bacon": 2, "Cheese": 2, "Egg": 1, "Nosty Sauce": 1},
    "N12_3 Loaf": {"Loaf": 0.33, "Chips": 1, "Boere Wors": 1, "Cheese": 1, "Atchar": 1},
    "Combo 45": {"Bread": 2, "Chicken Stripes": 1, "Chips": 1, "Cheese": 1},
    "Ext 10": {"Bread": 2, "Ham": 1, "Polony": 1, "Cheese": 1, "Atchar": 1}
}

def get_weather_impact() -> float:
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast?latitude=-26.85&longitude=26.66&daily=precipitation_probability_max&forecast_days=1&timezone=Africa/Johannesburg", 
            timeout=3
        )
        prob = r.json()["daily"]["precipitation_probability_max"][0]
        return 0.75 if prob > 50 else 1.0
    except Exception as e:
        logger.warning(f"Weather API unavailable: {e}")
        return 1.0

def get_sma_fallback(item_name: str, days: int = 7) -> float:
    """Computes a 14-day Simple Moving Average (SMA) fallback if Prophet fails."""
    if not supabase:
        return 0.0
    try:
        res = supabase.table("order_items").select("quantity, created_at").ilike("item_name", f"%{item_name}%").limit(100).execute()
        if not res.data:
            return 0.0
        df = pd.DataFrame(res.data)
        if df.empty:
            return 0.0
        
        total_qty = df["quantity"].sum()
        # Estimate average daily velocity over available records
        avg_daily = total_qty / max(len(df), 1)
        return max(0.0, avg_daily * days)
    except Exception as e:
        logger.error(f"SMA Fallback Error [{item_name}]: {e}")
        return 0.0

def run_safe_forecast(name: str, days: int = 7) -> Optional[pd.DataFrame]:
    """Runs a Prophet forecast with non-negative bounds constraints."""
    if not supabase: 
        return None
    try:
        res = supabase.table("order_items").select("order_id, quantity").ilike("item_name", f"%{name}%").execute()
        if not res.data: 
            return None
        
        order_ids = [it["order_id"] for it in res.data if "order_id" in it]
        if not order_ids:
            return None
            
        orders = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        if not orders.data: 
            return None
        
        df = pd.merge(pd.DataFrame(res.data), pd.DataFrame(orders.data).rename(columns={"id": "order_id"}), on="order_id")
        df["ds"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None).dt.date
        daily = df.groupby("ds")["quantity"].sum().reset_index().rename(columns={"ds": "ds", "quantity": "y"})
        
        # Require minimum 3 historical data points for Prophet time-series stability
        if len(daily) < 3: 
            return None
            
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(daily)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        
        # Clip negative predictions to 0
        forecast["yhat"] = forecast["yhat"].clip(lower=0.0)
        return forecast.tail(days)
    except Exception as e:
        logger.error(f"Forecast Error [{name}]: {e}")
        return None

def calculate_ingredient_demand(ingredient_name: str, days: int = 7) -> float:
    """Aggregates demand for an ingredient across all meals using Recipe BOM."""
    total_demand = 0.0
    matched_meals = 0
    
    for meal, recipe in RECIPES.items():
        if ingredient_name in recipe:
            matched_meals += 1
            f = run_safe_forecast(meal, days)
            if f is not None:
                predicted_meal_sales = float(f["yhat"].sum())
            else:
                predicted_meal_sales = get_sma_fallback(meal, days)
                
            total_demand += predicted_meal_sales * recipe[ingredient_name]
            
    # Direct fallback if ingredient is sold as a standalone item or not in preset recipes
    if matched_meals == 0:
        f = run_safe_forecast(ingredient_name, days)
        if f is not None:
            total_demand = float(f["yhat"].sum())
        else:
            total_demand = get_sma_fallback(ingredient_name, days)
            
    return max(0.0, total_demand)

@app.post("/api/forecast-meals")
async def forecast_meals():
    top_meals = list(RECIPES.keys())
    impact = get_weather_impact()
    results = {}
    
    for meal in top_meals:
        f = run_safe_forecast(meal, 1)
        if f is not None:
            val = float(f["yhat"].iloc[0])
        else:
            val = get_sma_fallback(meal, 1)
        results[meal] = round(max(0.0, val * impact), 1)
        
    return results

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    out = []
    total_rec = 0.0
    
    for entry in req.items:
        name = entry.item_name
        
        # 1. Fetch stock from database or payload fallback
        stock = 0.0
        if supabase:
            try:
                stock_res = supabase.table("ingredient_stock").select("current_stock").ilike("ingredient_name", f"%{name}%").execute()
                if stock_res.data:
                    stock = float(stock_res.data[0]["current_stock"])
                elif entry.current_stock is not None:
                    stock = float(entry.current_stock)
            except Exception as e:
                logger.error(f"Stock fetch error [{name}]: {e}")
                stock = float(entry.current_stock or 0.0)
        else:
            stock = float(entry.current_stock or 0.0)
        
        # 2. Run recipe-aware demand forecasting for 7 days
        weekly = calculate_ingredient_demand(name, days=7)
        daily = weekly / 7.0
        days_left = (stock / daily) if daily > 0 else (99.0 if stock > 0 else 0.0)
        
        # 3. Buffer logic: 1.5x weekly demand minus current stock
        recommend = max(0.0, (weekly * 1.5) - stock)
        total_rec += recommend
        
        urgency = "HIGH" if days_left < 3 else ("MEDIUM" if days_left < 7 else "LOW")
        status = "CRITICAL" if days_left < 3 else "OK"
        action = "REORDER NOW" if days_left < 3 else "Monitor stock"
        
        out.append({
            "item_name": name,
            "current_stock": round(stock, 1),
            "weekly_demand": round(weekly, 1),
            "days_left": round(days_left, 1),
            "recommended_order": round(recommend, 1),
            "urgency": urgency,
            "status": status,
            "action": action
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
        with open(file_path, "r") as f: 
            return f.read()
    return "<h1>KOTAai Active</h1><p>index.html missing.</p>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
