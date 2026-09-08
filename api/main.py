import os
import logging
import math
import re
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

app = FastAPI(title="KOTAai Ingredient Intelligence", version="5.3.0")

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

# Complete Bill of Materials (BOM)
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

def sanitize_query_term(term: str) -> str:
    """Cleans up search names (e.g., 'N12_3' -> 'N12') to prevent zero SQL matches."""
    cleaned = re.sub(r'[_+\-]', ' ', term).strip()
    parts = cleaned.split()
    return parts[0] if parts else term

def fetch_continuous_sales_df(item_name: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Fetches raw daily order data with resilient item matching and continuous dates.
    """
    today = pd.Timestamp.now().floor('D')
    start_date = today - pd.Timedelta(days=lookback_days - 1)
    full_date_range = pd.date_range(start=start_date, end=today, freq='D')
    default_df = pd.DataFrame({"ds": full_date_range, "y": 0.0})

    if not supabase:
        return default_df

    try:
        # Search using sanitized root term to ensure database matches
        clean_search = sanitize_query_term(item_name)
        res = supabase.table("order_items").select("*").ilike("item_name", f"%{clean_search}%").execute()
        
        if not res.data:
            # Fallback retry with raw item name
            res = supabase.table("order_items").select("*").ilike("item_name", f"%{item_name}%").execute()
            if not res.data:
                return default_df

        df_items = pd.DataFrame(res.data)

        # Check if created_at is directly on order_items table
        if "created_at" in df_items.columns:
            df_items["ds"] = pd.to_datetime(df_items["created_at"]).dt.tz_localize(None).dt.floor('D')
            daily_agg = df_items.groupby("ds")["quantity"].sum().reset_index().rename(columns={"quantity": "y"})
        elif "order_id" in df_items.columns:
            order_ids = df_items["order_id"].dropna().unique().tolist()
            if not order_ids:
                return default_df
            orders = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
            if not orders.data:
                return default_df
            df_orders = pd.DataFrame(orders.data).rename(columns={"id": "order_id"})
            df = pd.merge(df_items, df_orders, on="order_id")
            df["ds"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None).dt.floor('D')
            daily_agg = df.groupby("ds")["quantity"].sum().reset_index().rename(columns={"quantity": "y"})
        else:
            return default_df

        merged_df = pd.merge(pd.DataFrame({"ds": full_date_range}), daily_agg, on="ds", how="left").fillna(0.0)
        return merged_df

    except Exception as e:
        logger.error(f"Data Fetch Error [{item_name}]: {e}")
        return default_df

def run_safe_forecast(name: str, days: int = 7) -> float:
    """
    Computes daily/weekly demand projection using Prophet or moving average fallbacks.
    Guarantees a baseline prediction > 0 if historical sales exist anywhere.
    """
    df = fetch_continuous_sales_df(name, lookback_days=30)
    total_sales = df["y"].sum()

    # If no sales found in database for this specific item name, return reasonable baseline estimate
    if total_sales == 0:
        return float(days * 1.5)  # Baseline operational default (1.5 units/day)

    recent_7d = df.tail(7)["y"].mean()
    recent_30d = df["y"].mean()
    nonzero_days = (df["y"] > 0).sum()

    # Sparse sales logic (< 5 active days out of 30)
    if nonzero_days < 5 or len(df[df["y"] > 0]) < 3:
        avg_daily = max(recent_30d, recent_7d, 0.5)
        return float(avg_daily * days)

    try:
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)

        future_yhat = forecast.tail(days)["yhat"].clip(lower=0.0)
        predicted = float(future_yhat.sum())

        # Dynamic floor based on 7-day average
        baseline_floor = float(recent_7d * days)
        return max(predicted, baseline_floor, 1.0)
    except Exception as e:
        logger.error(f"Prophet Exception [{name}]: {e}")
        return max(float(recent_30d * days), 1.0)

def calculate_ingredient_demand(ingredient_name: str, days: int = 7) -> float:
    total_demand = 0.0
    matched_meals = 0
    clean_target = ingredient_name.strip().lower()

    for meal, recipe in RECIPES.items():
        for ing, qty in recipe.items():
            if ing.lower() == clean_target or clean_target in ing.lower():
                matched_meals += 1
                predicted_meal_sales = run_safe_forecast(meal, days)
                total_demand += (predicted_meal_sales * qty)
                break

    if matched_meals == 0:
        total_demand = run_safe_forecast(ingredient_name, days)

    return max(0.0, total_demand)

@app.post("/api/forecast-meals")
async def forecast_meals():
    top_meals = ["Original Dagwood", "Laprovance", "Tower of Terror", "N12_3", "Combo 45", "Ext 10"]
    impact = get_weather_impact()
    results = {}

    for meal in top_meals:
        predicted_daily = run_safe_forecast(meal, days=1)
        adjusted_val = predicted_daily * impact

        # Ensure expected daily sales display at least 1 unit if menu item exists
        results[meal] = max(1, math.ceil(adjusted_val))

    return results

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    out = []
    total_rec = 0.0

    for entry in req.items:
        name = entry.item_name.strip()

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

        weekly = calculate_ingredient_demand(name, days=7)
        daily = weekly / 7.0
        days_left = (stock / daily) if daily > 0 else (99.0 if stock > 0 else 0.0)

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
