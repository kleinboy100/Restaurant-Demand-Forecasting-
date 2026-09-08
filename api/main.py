import os
import logging
import math
import re
from datetime import datetime
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

app = FastAPI(title="KOTAai Ingredient Intelligence", version="5.5.0")

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

class MealForecastRequest(BaseModel):
    meals: Optional[List[str]] = None

# Complete Bill of Materials (BOM) including Meals, Combos, Chips, Loafs, and Tops
RECIPES: Dict[str, Dict[str, float]] = {
    # --- Tops & Add-ons (Quantity 1.0 per item) ---
    "Frankfurter Short": {"Frankfurter": 1.0},
    "Continental Russian Long": {"Russian": 1.0},
    "Continental Russian Short": {"Russian": 1.0},
    "Extra Sauce": {"Secret Sauce": 1.0},
    "Atchar": {"Atchar": 1.0},
    "Egg": {"Egg": 1.0},
    "Special Garlic": {"Special Garlic": 1.0},
    "Burger": {"Burger": 1.0},
    "Vienna": {"Vienna": 1.0},
    "Cheese": {"Cheese": 1.0},
    "Liver": {"Liver": 1.0},
    "Cheese Russian Long": {"Cheesy Russian": 1.0},
    "Cheese Russian Short": {"Cheesy Russian": 1.0},
    "Frankfurter Long": {"Frankfurter": 1.0},

    # --- Chips ---
    "Chips Extra Large": {"Chips": 4.0},
    "Chips Large": {"Chips": 3.0},
    "Chips Medium": {"Chips": 2.0},
    "Chips Small": {"Chips": 1.0},

    # --- Combos ---
    "Combo 10": {"Chips": 0.2, "Magwenya": 3.0},
    "Combo 13": {"Chips": 0.3, "Magwenya": 4.0, "Polony": 0.0025},
    "Combo 15": {"Chips": 0.2, "Magwenya": 5.0, "Polony": 0.025},
    "Combo 25": {"Chips": 0.4, "Magwenya": 6.0, "Polony": 0.0025, "Vienna": 1.0},
    "Combo 35": {"Chips": 0.3, "Magwenya": 6.0, "Polony": 0.025, "Unico Russian": 1.0},
    "Combo 45": {"Atchar": 2.5, "Chips": 1.0, "Liver": 1.0, "Magwenya": 6.0, "Russian": 1.0},

    # --- Dagwoods ---
    "Matlosana Dagwood": {"Bacon": 1.0, "Bread": 1.0, "Cheese": 2.0, "Cheesy Russian": 1.0, "Club Stake": 1.0, "Egg": 1.0, "Nosty Sauce": 1.0, "Onion": 0.0125, "Tomato": 0.167},
    "Mofarasai Dagwood": {"Bacon": 1.0, "Bread": 1.0, "Cheese": 2.0, "Cheesy Russian": 1.0, "Egg": 1.0, "Nosty Sauce": 1.0, "Rib Burger": 1.0, "Tomato": 0.167},
    "Original Dagwood": {"Bacon": 1.0, "Bread": 1.0, "Burger": 1.0, "Cheese": 1.0, "Egg": 1.0, "Nosty Sauce": 1.0, "Russian": 1.0, "Tomato": 0.167},
    "Turbo Dagwood": {"Bread": 1.0, "Cheese": 1.0, "Chicken Russian": 1.0, "Egg": 1.0, "Nosty Sauce": 1.0, "Rib Burger": 1.0, "Tomato": 0.167},

    # --- Loafs ---
    "N12_1": {"Bread": 1.0, "Cheese": 2.0, "Chips": 1.0, "Egg": 2.0, "Loaf": 0.25, "Nosty Sauce": 1.0, "Polony": 0.1, "Russian": 1.0, "Vienna": 1.0},
    "N12_2": {"Bread": 1.0, "Burger": 1.0, "Cheese": 2.0, "Chips": 1.0, "Egg": 2.0, "Loaf": 0.25, "Nosty Sauce": 1.0, "Polony": 0.15, "Russian": 1.0, "Vienna": 1.0},
    "N12_3": {"Bacon": 1.0, "Bread": 1.0, "Burger": 2.0, "Cheese": 3.0, "Chips": 2.0, "Egg": 3.0, "Loaf": 0.25, "Nosty Sauce": 1.0, "Polony": 0.2, "Russian": 2.0, "Vienna": 2.0},
    "N12_4": {"Bacon": 2.0, "Bread": 1.0, "Burger": 2.0, "Cheese": 4.0, "Chips": 2.0, "Egg": 4.0, "Loaf": 0.25, "Nosty Sauce": 1.0, "Polony": 0.25, "Russian": 3.0, "Vienna": 3.0},

    # --- Kota Menu ---
    "BBL Tower of Terror": {"Bacon": 1.0, "Bread": 0.25, "Cheese": 1.0, "Chips": 0.2, "Egg": 1.0, "Frankfurter": 1.0, "Lettuce": 0.005, "Polony": 0.0025, "Rib Burger": 1.0, "Russian": 1.0, "Secret Sauce": 0.0005, "Vienna": 1.0},
    "Bosrand": {"Bread": 0.25, "Cheese": 1.0, "Chips": 0.4, "Egg": 1.0, "Lettuce": 0.005, "Polony": 0.00025, "Secret Sauce": 0.005, "Unico Russian": 1.0},
    "Cheesy D": {"Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 0.2, "Egg": 1.0, "Lettuce": 0.005, "Secret Sauce": 0.025, "Unico Russian": 1.0},
    "Curry Fish Kota": {"Atchar": 1.0, "Bread": 0.25, "Curry Fish": 1.0},
    "Dark City": {"Bread": 0.25, "Cheese": 1.0, "Cheesy Russian": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0},
    "Di_Y_Kota": {"Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Ham": 1.0, "Lettuce": 1.0, "Russian": 1.0, "Secret Sauce": 1.0},
    "Di_Z_Kota": {"Bread": 0.25, "Cheese": 1.0, "Cheesy Russian": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0},
    "Down": {"Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0},
    "Ext 10": {"Bread": 0.25, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Nosty Sauce": 1.0, "Polony": 0.025},
    "Ext 6": {"Bacon": 1.0, "Bread": 0.25, "Chips": 1.0, "Egg": 1.0, "Fish Fillet": 1.0, "Lettuce": 1.0, "Russian": 1.0, "Secret Sauce": 1.0},
    "Flamwood": {"Bacon": 1.0, "Bread": 0.25, "Cheese": 1.0, "Chicken Stripes": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Russian": 1.0, "Secret Sauce": 1.0},
    "J_town": {"Atchar": 1.0, "Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Russian": 1.0, "Secret Sauce": 1.0, "Tomato": 0.167},
    "La Hof": {"Bacon": 1.0, "Bread": 0.25, "Cheesy Russian": 1.0, "Chicken Stripes": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0},
    "Laprovance": {"Bacon": 1.0, "Boere Wors": 1.0, "Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Club Stake": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Onion": 0.0125, "Polony": 0.025, "Russian": 1.0, "Secret Sauce": 1.0},
    "Mince Kota": {"Atchar": 1.0, "Bread": 0.25, "Mince": 1.0},
    "Phelandaba": {"Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Russian": 1.0, "Secret Sauce": 1.0},
    "Stop 1": {"Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Polony": 0.025, "Secret Sauce": 1.0},
    "Stop 18": {"Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Polony": 0.025, "Russian": 1.0, "Secret Sauce": 1.0},
    "Stop 5_1": {"Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0, "Vienna": 1.0},
    "Stop 5+": {"Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Ham": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0, "Vienna": 1.0},
    "Sun City": {"Bread": 0.25, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Frankfurter": 1.0, "Lettuce": 1.0, "Secret Sauce": 1.0},
    "Tower of Terror": {"Bacon": 1.0, "Bread": 0.25, "Burger": 1.0, "Cheese": 1.0, "Chips": 1.0, "Egg": 1.0, "Frankfurter": 1.0, "Lettuce": 1.0, "Polony": 0.025, "Russian": 1.0, "Secret Sauce": 1.0}
}

def clean_name(term: str) -> str:
    """Standardizes names by stripping symbols for fuzzy matching."""
    return re.sub(r'[^a-zA-Z0-9]', '', term).lower()

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
    today = pd.Timestamp.now().floor('D')
    start_date = today - pd.Timedelta(days=lookback_days - 1)
    full_date_range = pd.date_range(start=start_date, end=today, freq='D')
    default_df = pd.DataFrame({"ds": full_date_range, "y": 0.0})

    if not supabase:
        return default_df

    try:
        search_token = item_name.split()[0] if item_name.split() else item_name
        res = supabase.table("order_items").select("*").ilike("item_name", f"%{search_token}%").execute()
        
        if not res.data:
            return default_df

        df_items = pd.DataFrame(res.data)

        target_clean = clean_name(item_name)
        df_items["clean_name"] = df_items["item_name"].astype(str).apply(clean_name)
        df_matched = df_items[df_items["clean_name"].str.contains(target_clean, na=False) | df_items["clean_name"].apply(lambda x: target_clean in x)]

        if df_matched.empty:
            return default_df

        if "created_at" in df_matched.columns:
            df_matched["ds"] = pd.to_datetime(df_matched["created_at"]).dt.tz_localize(None).dt.floor('D')
            daily_agg = df_matched.groupby("ds")["quantity"].sum().reset_index().rename(columns={"quantity": "y"})
        elif "order_id" in df_matched.columns:
            order_ids = df_matched["order_id"].dropna().unique().tolist()
            if not order_ids:
                return default_df
            orders = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
            if not orders.data:
                return default_df
            df_orders = pd.DataFrame(orders.data).rename(columns={"id": "order_id"})
            df = pd.merge(df_matched, df_orders, on="order_id")
            df["ds"] = pd.to_datetime(df["created_at"]).dt.tz_localize(None).dt.floor('D')
            daily_agg = df.groupby("ds")["quantity"].sum().reset_index().rename(columns={"quantity": "y"})
        else:
            return default_df

        merged_df = pd.merge(pd.DataFrame({"ds": full_date_range}), daily_agg, on="ds", how="left").fillna(0.0)
        return merged_df

    except Exception as e:
        logger.error(f"Data Fetch Error [{item_name}]: {e}")
        return default_df

def run_safe_forecast(name: str, days: int = 1) -> float:
    """
    Computes exact historical daily forecast.
    Returns 0.0 if there are no historical sales recorded.
    """
    df = fetch_continuous_sales_df(name, lookback_days=30)
    total_sales = df["y"].sum()

    # Accurate Fallback: If no sales recorded, demand is 0.0
    if total_sales == 0:
        return 0.0

    recent_7d = df.tail(7)["y"].mean()
    recent_30d = df["y"].mean()
    nonzero_days = (df["y"] > 0).sum()

    if nonzero_days < 3:
        return float(recent_30d * days)

    try:
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)

        future_yhat = forecast.tail(days)["yhat"].clip(lower=0.0)
        predicted = float(future_yhat.sum())

        return max(0.0, predicted)
    except Exception as e:
        logger.error(f"Prophet Exception [{name}]: {e}")
        return max(0.0, float(recent_30d * days))

@app.post("/api/forecast-meals")
async def forecast_meals(req: Optional[MealForecastRequest] = None):
    if req and req.meals and len(req.meals) > 0:
        target_meals = req.meals
    else:
        target_meals = list(RECIPES.keys())

    impact = get_weather_impact()
    results = {}

    for meal in target_meals:
        predicted_daily = run_safe_forecast(meal, days=1)
        adjusted_val = predicted_daily * impact
        
        # Round to nearest integer (returns 0 if model predicts 0 or below 0.5)
        results[meal] = int(round(adjusted_val))

    return results

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    out = []
    total_rec = 0.0

    for entry in req.items:
        name = entry.item_name.strip()
        stock = float(entry.current_stock or 0.0)

        if supabase:
            try:
                stock_res = supabase.table("ingredient_stock").select("current_stock").ilike("ingredient_name", name).execute()
                if stock_res.data:
                    stock = float(stock_res.data[0]["current_stock"])
            except Exception as e:
                logger.error(f"Stock fetch error [{name}]: {e}")

        # Weekly ingredient requirement based on true forecast
        weekly_demand = 0.0
        clean_target = clean_name(name)
        for meal_name, recipe in RECIPES.items():
            for ing, qty in recipe.items():
                if clean_name(ing) == clean_target or clean_target in clean_name(ing):
                    meal_forecast = run_safe_forecast(meal_name, days=7)
                    weekly_demand += (meal_forecast * qty)
                    break

        # Fallback to direct ingredient history (defaults to 0.0 if not sold)
        if weekly_demand == 0.0:
            weekly_demand = run_safe_forecast(name, days=7)

        daily = weekly_demand / 7.0
        days_left = (stock / daily) if daily > 0 else (99.0 if stock > 0 else 0.0)

        recommend = max(0.0, (weekly_demand * 1.5) - stock)
        total_rec += recommend

        urgency = "HIGH" if days_left < 3 else ("MEDIUM" if days_left < 7 else "LOW")
        status = "CRITICAL" if days_left < 3 else "OK"
        action = "REORDER NOW" if days_left < 3 else "Monitor stock"

        out.append({
            "item_name": name,
            "current_stock": round(stock, 1),
            "weekly_demand": round(weekly_demand, 1),
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
