import os
import logging
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import numpy as np
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

app = FastAPI(title="KOTAai Ingredient Intelligence", version="5.2.0")

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

# Complete Bill of Materials (BOM) populated directly from the database recipe table
# Standard Portion Unit Base: 250g Chips = 1.0 Portion
RECIPES: Dict[str, Dict[str, float]] = {
    "BBL Tower of Terror": {
        "Bacon": 1.0, "Bread": 0.25, "Cheese": 1.0, "Chips": 0.2, "Egg": 1.0, 
        "Frankfurter": 1.0, "Lettuce": 0.005, "Polony": 0.0025, "Rib Burger": 1.0, 
        "Russian": 1.0, "Secret Sauce": 0.0005, "Vienna": 1.0
    },
    "Bosrand": {
        "Bread": 0.25, "Cheese": 1.0, "Chips": 0.4, "Egg": 1.0, 
        "Lettuce": 0.005, "Polony": 0.00025, "Secret Sauce": 0.005, "Unico Russian": 1.0
    },
    "Cheesy D": {
        "Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 0.2, 
        "Egg": 1.0, "Lettuce": 0.005, "Secret Sauce": 0.025, "Unico Russian": 1.0
    },
    "Chips Extra Large": {"Chips": 4.0},
    "Chips Large": {"Chips": 3.0},
    "Chips Medium": {"Chips": 2.0},
    "Chips Small": {"Chips": 1.0},
    "Combo 10": {"Chips": 0.2, "Magwenya": 3.0},
    "Combo 13": {"Chips": 0.3, "Magwenya": 4.0, "Polony": 0.0025},
    "Combo 15": {"Chips": 0.2, "Magwenya": 5.0, "Polony": 0.025},
    "Combo 25": {"Chips": 0.4, "Magwenya": 6.0, "Polony": 0.0025, "Vienna": 1.0},
    "Combo 35": {"Chips": 0.3, "Magwenya": 6.0, "Polony": 0.025, "Unico Russian": 1.0},
    "Combo 45": {"Atchar": 2.5, "Chips": 1.0, "Liver": 1.0, "Magwenya": 6.0, "Russian": 1.0},
    "Curry Fish Kota": {"Atchar": 1.0, "Bread": 0.25, "Curry Fish": 1.0},
    "Dark City": {
        "Bread": 0.25, "Cheese": 1.0, "Cheesy Russian": 1.0, "Chips": 1.0, 
        "Egg": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0
    },
    "Di_Y_Kota": {
        "Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 1.0, 
        "Egg": 1.0, "Ham": 1.0, "Lettuce": 1.0, "Russian": 1.0, "Secret Sauce": 1.0
    },
    "Di_Z_Kota": {
        "Bread": 0.25, "Cheese": 1.0, "Cheesy Russian": 1.0, "Chips": 1.0, 
        "Egg": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0
    },
    "Down": {
        "Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 1.0, 
        "Egg": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0
    },
    "Ext 10": {
        "Bread": 0.25, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, 
        "Nosty Sauce": 1.0, "Polony": 0.025
    },
    "Ext 6": {
        "Bacon": 1.0, "Bread": 0.25, "Chips": 1.0, "Egg": 1.0, "Fish Fillet": 1.0, 
        "Lettuce": 1.0, "Russian": 1.0, "Secret Sauce": 1.0
    },
    "Flamwood": {
        "Bacon": 1.0, "Bread": 0.25, "Cheese": 1.0, "Chicken Stripes": 1.0, 
        "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Russian": 1.0, "Secret Sauce": 1.0
    },
    "J_town": {
        "Atchar": 1.0, "Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, 
        "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Russian": 1.0, "Secret Sauce": 1.0, "Tomato": 0.167
    },
    "La Hof": {
        "Bacon": 1.0, "Bread": 0.25, "Cheesy Russian": 1.0, "Chicken Stripes": 1.0, 
        "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0
    },
    "Laprovance": {
        "Bacon": 1.0, "Boere Wors": 1.0, "Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, 
        "Club Stake": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Onion": 0.0125, "Polony": 0.025, 
        "Russian": 1.0, "Secret Sauce": 1.0
    },
    "Matlosana Dagwood": {
        "Bacon": 1.0, "Bread": 1.0, "Cheese": 2.0, "Cheesy Russian": 1.0, 
        "Club Stake": 1.0, "Egg": 1.0, "Nosty Sauce": 1.0, "Onion": 0.0125, "Tomato": 0.167
    },
    "Mince Kota": {"Atchar": 1.0, "Bread": 0.25, "Mince": 1.0},
    "Mofarasai Dagwood": {
        "Bacon": 1.0, "Bread": 1.0, "Cheese": 2.0, "Cheesy Russian": 1.0, 
        "Egg": 1.0, "Nosty Sauce": 1.0, "Rib Burger": 1.0, "Tomato": 0.167
    },
    "N12_1": {
        "Bread": 1.0, "Cheese": 2.0, "Chips": 1.0, "Egg": 2.0, "Loaf": 0.25, 
        "Nosty Sauce": 1.0, "Polony": 0.1, "Russian": 1.0, "Vienna": 1.0
    },
    "N12_2": {
        "Bread": 1.0, "Burger": 1.0, "Cheese": 2.0, "Chips": 1.0, "Egg": 2.0, 
        "Loaf": 0.25, "Nosty Sauce": 1.0, "Polony": 0.15, "Russian": 1.0, "Vienna": 1.0
    },
    "N12_3": {
        "Bacon": 1.0, "Bread": 1.0, "Burger": 2.0, "Cheese": 3.0, "Chips": 2.0, 
        "Egg": 3.0, "Loaf": 0.25, "Nosty Sauce": 1.0, "Polony": 0.2, "Russian": 2.0, "Vienna": 2.0
    },
    "N12_4": {
        "Bacon": 2.0, "Bread": 1.0, "Burger": 2.0, "Cheese": 4.0, "Chips": 2.0, 
        "Egg": 4.0, "Loaf": 0.25, "Nosty Sauce": 1.0, "Polony": 0.25, "Russian": 3.0, "Vienna": 3.0
    },
    "Original Dagwood": {
        "Bacon": 1.0, "Bread": 1.0, "Burger": 1.0, "Cheese": 1.0, "Egg": 1.0, 
        "Nosty Sauce": 1.0, "Russian": 1.0, "Tomato": 0.167
    },
    "Phelandaba": {
        "Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, 
        "Russian": 1.0, "Secret Sauce": 1.0
    },
    "Stop 1": {
        "Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, 
        "Polony": 0.025, "Secret Sauce": 1.0
    },
    "Stop 18": {
        "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, 
        "Lettuce": 1.0, "Polony": 0.025, "Russian": 1.0, "Secret Sauce": 1.0
    },
    "Stop 5_1": {
        "Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, 
        "Secret Sauce": 1.0, "Vienna": 1.0
    },
    "Stop 5+": {
        "Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Ham": 1.0, 
        "Lettuce": 1.0, "Secret Sauce": 1.0, "Vienna": 1.0
    },
    "Sun City": {
        "Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Frankfurter": 1.0, 
        "Lettuce": 1.0, "Secret Sauce": 1.0
    },
    "Tower of Terror": {
        "Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 1.0, 
        "Egg": 1.0, "Frankfurter": 1.0, "Lettuce": 1.0, "Polony": 0.025, "Russian": 1.0, "Secret Sauce": 1.0
    },
    "Turbo Dagwood": {
        "Bread": 1.0, "Cheese": 1.0, "Chicken Russian": 1.0, "Egg": 1.0, 
        "Nosty Sauce": 1.0, "Rib Burger": 1.0, "Tomato": 0.167
    }
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

def fetch_continuous_sales_df(item_name: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Fetches raw daily order data and reindexes across a continuous date grid.
    Fills zero-sales days with 0 to resolve zero-inflation/missing array errors.
    """
    today = pd.Timestamp.now().floor('D')
    start_date = today - pd.Timedelta(days=lookback_days - 1)
    full_date_range = pd.date_range(start=start_date, end=today, freq='D')
    
    if not supabase:
        return pd.DataFrame({"ds": full_date_range, "y": 0.0})
    
    try:
        res = supabase.table("order_items").select("order_id, quantity").ilike("item_name", f"%{item_name}%").execute()
        if not res.data:
            return pd.DataFrame({"ds": full_date_range, "y": 0.0})
            
        order_ids = [it["order_id"] for it in res.data if "order_id" in it]
        if not order_ids:
            return pd.DataFrame({"ds": full_date_range, "y": 0.0})
            
        orders = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        if not orders.data:
            return pd.DataFrame({"ds": full_date_range, "y": 0.0})
            
        df_items = pd.DataFrame(res.data)
        df_orders = pd.DataFrame(orders.data).rename(columns={"id": "order_id"})
        df = pd.merge(df_items, df_orders, on="order_id")
        
        df["ds"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None).dt.floor('D')
        daily_agg = df.groupby("ds")["quantity"].sum().reset_index().rename(columns={"quantity": "y"})
        
        # Merge with contiguous date range to explicitly model non-sale days
        continuous_df = pd.DataFrame({"ds": full_date_range})
        merged_df = pd.merge(continuous_df, daily_agg, on="ds", how="left").fillna(0.0)
        return merged_df
    except Exception as e:
        logger.error(f"Error fetching continuous time-series [{item_name}]: {e}")
        return pd.DataFrame({"ds": full_date_range, "y": 0.0})

def calculate_intermittent_fallback(df: pd.DataFrame, days_ahead: int = 7) -> float:
    """
    Calculates moving-average baseline demand for low-frequency/sparse items,
    preventing non-zero intermittent products from truncating to 0.
    """
    if df.empty or df["y"].sum() == 0:
        return 0.0

    recent_7d_avg = df.tail(7)["y"].mean()
    recent_30d_avg = df["y"].mean()
    nonzero_days = (df["y"] > 0).sum()

    if nonzero_days < 5:
        # High sparsity baseline
        predicted_daily_qty = max(recent_30d_avg, 0.5) if recent_30d_avg > 0 else 0.0
    else:
        # Weighted demand smoothing
        predicted_daily_qty = (0.6 * recent_7d_avg) + (0.4 * recent_30d_avg)
        predicted_daily_qty = max(0.5, predicted_daily_qty)

    return float(predicted_daily_qty * days_ahead)

def run_safe_forecast(name: str, days: int = 7) -> float:
    """
    Runs Prophet with floor controls or falls back to intermittent/moving-average logic.
    Returns total projected demand quantity for the specified period.
    """
    df = fetch_continuous_sales_df(name, lookback_days=30)
    nonzero_days = (df["y"] > 0).sum()

    # Fallback if historical points are sparse or Prophet won't converge
    if nonzero_days < 5:
        return calculate_intermittent_fallback(df, days_ahead=days)

    try:
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        
        # Take future period predictions and clip negative trends to 0
        future_yhat = forecast.tail(days)["yhat"].clip(lower=0.0)
        total_predicted = float(future_yhat.sum())

        # Safety baseline threshold if Prophet severely underpredicts active demand
        fallback_val = calculate_intermittent_fallback(df, days_ahead=days)
        return max(total_predicted, fallback_val)
    except Exception as e:
        logger.error(f"Prophet Forecast Exception [{name}]: {e}")
        return calculate_intermittent_fallback(df, days_ahead=days)

def calculate_ingredient_demand(ingredient_name: str, days: int = 7) -> float:
    """Aggregates total demand for an ingredient across all recipes in the BOM."""
    total_demand = 0.0
    matched_meals = 0
    clean_target = ingredient_name.strip().lower()
    
    for meal, recipe in RECIPES.items():
        # Match exact ingredient or exact substring
        for ing, qty in recipe.items():
            if ing.lower() == clean_target or clean_target in ing.lower():
                matched_meals += 1
                predicted_meal_sales = run_safe_forecast(meal, days)
                total_demand += (predicted_meal_sales * qty)
                break
            
    # Direct fallback if item is ordered standalone rather than via a meal recipe
    if matched_meals == 0:
        total_demand = run_safe_forecast(ingredient_name, days)
            
    return max(0.0, total_demand)

@app.post("/api/forecast-meals")
async def forecast_meals():
    top_meals = ["Original Dagwood", "Laprovance", "Tower of Terror", "N12_3", "Combo 45", "Ext 10"]
    impact = get_weather_impact()
    results = {}
    
    for meal in top_meals:
        predicted_qty = run_safe_forecast(meal, days=1)
        adjusted_val = predicted_qty * impact
        
        # Round up fractional demand to ensure low-frequency meals display baseline integer sales
        results[meal] = math.ceil(adjusted_val) if adjusted_val > 0 else 0
        
    return results

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    out = []
    total_rec = 0.0
    
    for entry in req.items:
        name = entry.item_name.strip()
        
        # 1. Fetch current stock from database or payload
        stock = 0.0
        if supabase:
            try:
                stock_res = supabase.table("ingredient_stock").select("current_stock").ilike("ingredient_name", name).execute()
                if stock_res.data:
                    stock = float(stock_res.data[0]["current_stock"])
                elif entry.current_stock is not None:
                    stock = float(entry.current_stock)
            except Exception as e:
                logger.error(f"Stock fetch error [{name}]: {e}")
                stock = float(entry.current_stock or 0.0)
        else:
            stock = float(entry.current_stock or 0.0)
        
        # 2. Demand calculation using the complete database BOM
        weekly = calculate_ingredient_demand(name, days=7)
        daily = weekly / 7.0
        days_left = (stock / daily) if daily > 0 else (99.0 if stock > 0 else 0.0)
        
        # 3. Buffer calculation (1.5x weekly demand - current stock)
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
