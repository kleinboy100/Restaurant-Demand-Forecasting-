import os
import logging
import asyncio
from datetime import datetime, timedelta, date, timezone
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from prophet import Prophet
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import requests

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FASTAPI APP INITIALIZATION
# ---------------------------------------------------------------------------
app = FastAPI(
    title="KOTAai Restaurant Demand Forecasting API",
    description="Advanced AI-powered demand forecasting for Kota King Klerksdorp",
    version="2.3.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------------------------------
supabase: Optional[Client] = None

# ---------------------------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------------------------
class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[int] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

class UsageHistoryRequest(BaseModel):
    target_date: str # Format: YYYY-MM-DD
    ingredient_name: Optional[str] = "All"

# ---------------------------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------------------------
def init_supabase():
    global supabase
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in environment variables")
            return False
        
        supabase = create_client(supabase_url, supabase_key)
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    init_supabase()

# ---------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------
def get_klerksdorp_weather(days_ahead: int = 7) -> Dict[str, float]:
    try:
        lat, lon = -26.85, 26.66
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "daily": "precipitation_probability_max", "forecast_days": days_ahead, "timezone": "Africa/Johannesburg"}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        weather_impact = {}
        for i, date_str in enumerate(data["daily"]["time"]):
            prob = data["daily"]["precipitation_probability_max"][i]
            impact = 1.0 if prob < 20 else (0.85 if prob < 60 else 0.7)
            weather_impact[date_str] = impact
        return weather_impact
    except Exception as e:
        future_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days_ahead + 1)]
        return {date_str: 1.0 for date_str in future_dates}

def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    if not supabase: return pd.DataFrame()
    try:
        order_items_result = supabase.table("order_items").select("order_id, item_name, quantity").eq("item_name", item_name).execute()
        if not order_items_result.data: return pd.DataFrame()
        
        order_ids = [item["order_id"] for item in order_items_result.data]
        if not order_ids: return pd.DataFrame()

        orders_result = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        if not orders_result.data: return pd.DataFrame()
        
        order_items_df = pd.DataFrame(order_items_result.data)
        orders_df = pd.DataFrame(orders_result.data).rename(columns={"id": "order_id"})
        merged_df = pd.merge(order_items_df, orders_df, on="order_id", how="inner")
        merged_df["created_at"] = pd.to_datetime(merged_df["created_at"])
        cutoff_date = datetime.now() - timedelta(days=days_back)
        merged_df = merged_df[merged_df["created_at"] >= cutoff_date]
        if merged_df.empty: return pd.DataFrame()
        
        merged_df["sale_date"] = merged_df["created_at"].dt.date
        daily_sales = merged_df.groupby("sale_date")["quantity"].sum().reset_index()
        daily_sales.rename(columns={"sale_date": "ds", "quantity": "y"}, inplace=True)
        daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])
        return daily_sales
    except Exception as e:
        logger.error(f"Error fetching sales data: {str(e)}")
        return pd.DataFrame()

def get_ingredient_usage(target_date: Optional[date] = None) -> Dict[str, float]:
    """
    Calculates total ingredient usage for ANY given date based on SAST.
    If no date is provided, defaults to today.
    """
    if not supabase: return {}
    
    try:
        sast_tz = timezone(timedelta(hours=2))
        
        if target_date is None:
            now_sast = datetime.now(sast_tz)
            target_date = now_sast.date()
            
        # Define 00:00:00 to 23:59:59 SAST for the requested date
        start_of_day = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=sast_tz)
        end_of_day = start_of_day + timedelta(days=1)
        
        # Fetch orders within this 24 hour window
        orders_res = supabase.table("orders").select("id").gte("created_at", start_of_day.isoformat()).lt("created_at", end_of_day.isoformat()).execute()
        
        if not orders_res.data: return {}
        order_ids = [o["id"] for o in orders_res.data]
        if not order_ids: return {}
            
        # Fetch order items
        items_res = supabase.table("order_items").select("item_name, quantity").in_("order_id", order_ids).execute()
        if not items_res.data: return {}
            
        # Fetch meal recipes
        recipes_res = supabase.table("meal_recipes").select("meal_name, ingredient_name, quantity_per_meal").execute()
        
        recipes = {}
        for r in recipes_res.data:
            meal = r["meal_name"].strip().lower()
            if meal not in recipes: recipes[meal] = []
            recipes[meal].append(r)
            
        # Calculate usage
        usage_dict = {}
        for item in items_res.data:
            meal_name = item["item_name"].strip().lower()
            qty = item["quantity"]
            
            if meal_name in recipes:
                for ing in recipes[meal_name]:
                    ing_name = ing["ingredient_name"].strip().lower()
                    ing_qty = float(ing["quantity_per_meal"])
                    usage_dict[ing_name] = usage_dict.get(ing_name, 0.0) + (ing_qty * qty)
                    
        return usage_dict
    except Exception as e:
        logger.error(f"Error calculating usage for {target_date}: {str(e)}")
        return {}

def generate_world_class_forecast(item_name: str, days_ahead: int) -> Optional[pd.DataFrame]:
    df = get_sales_from_order_items(item_name)
    if df.empty: return None

    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name="ZA")
    model.fit(df)

    future = model.make_future_dataframe(periods=days_ahead, freq='D')
    weather_impact = get_klerksdorp_weather(days_ahead)
    future["impact_score"] = [weather_impact.get(d.strftime("%Y-%m-%d"), 1.0) for d in future["ds"]]
    
    forecast = model.predict(future)
    forecast["final_prediction"] = (forecast["yhat"] * future["impact_score"]).clip(lower=0)
    return forecast[forecast["ds"] > df["ds"].max()].copy()

# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/api/usage-history")
async def historical_usage(request: UsageHistoryRequest):
    """New endpoint to check historical usage for specific dates and ingredients."""
    try:
        # Parse the string date into a Python date object
        target_date = datetime.strptime(request.target_date, "%Y-%m-%d").date()
        
        # Calculate usage for that specific date
        usage_dict = get_ingredient_usage(target_date)
        
        if request.ingredient_name and request.ingredient_name != "All":
            lookup_name = request.ingredient_name.strip().lower()
            amount_used = usage_dict.get(lookup_name, 0.0)
            return {
                "date": request.target_date,
                "ingredient": request.ingredient_name,
                "amount_used": round(amount_used, 2),
                "all_data": usage_dict
            }
        else:
            return {
                "date": request.target_date,
                "ingredient": "All",
                "all_data": {k: round(v, 2) for k, v in usage_dict.items()}
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    try:
        items_data = []
        total_recommended = 0
        critical_count = 0
        
        # Calculate TODAY's usage
        today_usage = get_ingredient_usage()
        
        fallback_demands = {"Chips": 500.0, "Melted Cheese": 50.0, "Russian": 12.0, "lettuce": 8.0, "Bread": 25.0, "tomato": 10.0, "atchar": 5.0, "Vienna": 15.0, "egg": 20.0, "steak": 6.0}
        
        for item in request.items:
            try:
                lookup_name = item.item_name.strip()
                current_stock = item.current_stock
                
                if current_stock is None and supabase:
                    stock_result = supabase.table("ingredient_stock").select("current_stock").ilike("ingredient_name", lookup_name).limit(1).execute()
                    current_stock = stock_result.data[0]["current_stock"] if stock_result.data else 0
                
                forecast_df = generate_world_class_forecast(item.item_name, 7)
                weekly_demand = forecast_df["final_prediction"].sum() if forecast_df is not None else fallback_demands.get(lookup_name, 5.0)
                
                daily_used = today_usage.get(lookup_name.lower(), 0.0)
                
                daily_demand = weekly_demand / 7
                days_left = current_stock / daily_demand if daily_demand > 0 else 999
                recommended_order = max(0, (weekly_demand * 1.5) - current_stock)
                
                if days_left < 3:
                    urgency, status = "HIGH", "CRITICAL"
                    critical_count += 1
                elif days_left < 7:
                    urgency, status = "MEDIUM", "LOW"
                else:
                    urgency, status = "LOW", "OK"
                
                total_recommended += recommended_order
                
                items_data.append({
                    "item_name": item.item_name,
                    "current_stock": current_stock,
                    "daily_usage": round(daily_used, 2),
                    "weekly_demand": round(weekly_demand, 1),
                    "days_left": round(days_left, 1),
                    "recommended_order": round(recommended_order, 1),
                    "urgency": urgency,
                    "status": status,
                    "action": "REORDER NOW" if urgency == "HIGH" else "Monitor stock"
                })
                
            except Exception as e:
                logger.error(f"Error processing {item.item_name}: {str(e)}")
        
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        items_data.sort(key=lambda x: urgency_order.get(x["urgency"], 3))
        
        return {
            "summary": {
                "total_items": len(items_data),
                "critical_items": critical_count,
                "total_recommended": round(total_recommended, 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "items": items_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
