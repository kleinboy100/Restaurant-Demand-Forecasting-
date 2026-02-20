import os
import logging
from datetime import datetime, timedelta, date, timezone
from typing import Optional, List, Dict
import pandas as pd
from prophet import Prophet
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KOTAai Ingredient Intelligence API",
    description="Ingredient-level demand forecasting for Kota King Klerksdorp",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Optional[Client] = None

class DashboardItem(BaseModel):
    item_name: str  # NOW EXCLUSIVELY INGREDIENT NAMES
    current_stock: Optional[float] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

class UsageHistoryRequest(BaseModel):
    target_date: str
    ingredient_name: str

@app.on_event("startup")
async def startup_event():
    global supabase
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials missing")
            return
        supabase = create_client(supabase_url, supabase_key)
        logger.info("✅ Database connected successfully")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")

def get_klerksdorp_weather(days_ahead: int = 7) -> Dict[str, float]:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": -26.85, 
            "longitude": 26.66, 
            "daily": "precipitation_probability_max", 
            "forecast_days": days_ahead, 
            "timezone": "Africa/Johannesburg"
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return {
            date_str: (1.0 if prob < 20 else (0.85 if prob < 60 else 0.7))
            for i, date_str in enumerate(data["daily"]["time"])
            for prob in [data["daily"]["precipitation_probability_max"][i]]
        }
    except Exception as e:
        logger.warning(f"Weather API failed: {str(e)}")
        return {}

def get_ingredient_usage(target_date: Optional[date] = None) -> Dict[str, float]:
    """Returns {ingredient_name_lower: total_units_used} for target date"""
    if not supabase:
        return {}
    
    try:
        sast_tz = timezone(timedelta(hours=2))
        if target_date is None:
            target_date = datetime.now(sast_tz).date()
        
        # Convert SAST boundaries → UTC for Supabase query
        start_sast = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=sast_tz)
        end_sast = start_sast + timedelta(days=1)
        start_utc = start_sast.astimezone(timezone.utc).isoformat()
        end_utc = end_sast.astimezone(timezone.utc).isoformat()
        
        # Get orders for date range
        orders = supabase.table("orders")\
            .select("id")\
            .gte("created_at", start_utc)\
            .lt("created_at", end_utc)\
            .execute()
        if not orders.data:
            return {}
        
        order_ids = [o["id"] for o in orders.data]
        if not order_ids:
            return {}
        
        # Get order items
        items = supabase.table("order_items")\
            .select("item_name, quantity")\
            .in_("order_id", order_ids)\
            .execute()
        if not items.data:
            return {}
        
        # Get ALL recipes (critical for ingredient mapping)
        recipes = supabase.table("meal_recipes")\
            .select("meal_name, ingredient_name, quantity_per_meal")\
            .execute()
        if not recipes.data:
            logger.warning("⚠️ No recipes found in meal_recipes table")
            return {}
        
        # Build recipe lookup: {meal_name_lower: [(ingredient, qty), ...]}
        recipe_map = {}
        for r in recipes.data:
            meal_key = r["meal_name"].strip().lower()
            if meal_key not in recipe_map:
                recipe_map[meal_key] = []
            recipe_map[meal_key].append((
                r["ingredient_name"].strip().lower(),
                float(r["quantity_per_meal"])
            ))
        
        # Calculate ingredient usage
        usage = {}
        for item in items.data:
            order_meal = str(item.get("item_name", "")).strip().lower()
            qty = float(item.get("quantity", 1))
            
            # Fuzzy match: find recipe where recipe name is substring of order name
            matched_recipe = None
            if order_meal in recipe_map:
                matched_recipe = order_meal
            else:
                for recipe_key in recipe_map.keys():
                    if recipe_key in order_meal or order_meal in recipe_key:
                        matched_recipe = recipe_key
                        break
            
            if matched_recipe:
                for ing_name, ing_qty_per_meal in recipe_map[matched_recipe]:
                    total_qty = ing_qty_per_meal * qty
                    usage[ing_name] = usage.get(ing_name, 0.0) + total_qty
        
        return usage
    except Exception as e:
        logger.error(f"UsageId calculation failed: {str(e)}", exc_info=True)
        return {}

def calculate_ingredient_weekly_demand(ingredient_name: str, days_ahead: int = 7) -> float:
    """
    Calculates projected weekly demand for an INGREDIENT by:
    1. Finding all meals containing this ingredient
    2. Forecasting each meal's demand
    3. Converting meal forecasts → ingredient demand
    """
    if not supabase:
        return 0.0
    
    try:
        # Get all meals using this ingredient
        recipes = supabase.table("meal_recipes")\
            .select("meal_name, quantity_per_meal")\
            .eq("ingredient_name", ingredient_name)\
            .execute()
        
        if not recipes.data:
            return 0.0
        
        total_demand = 0.0
        weather_impact = get_klerksdorp_weather(days_ahead)
        
        for recipe in recipes.data:
            meal_name = recipe["meal_name"]
            qty_per_meal = float(recipe["quantity_per_meal"])
            
            # Get historical sales for this meal
            sales_df = get_sales_from_order_items(meal_name, days_back=90)
            if sales_df.empty or len(sales_df) < 5:  # Need min data
                continue
            
            # Forecast meal demand
            model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
            model.add_country_holidays(country_name="ZA")
            model.fit(sales_df)
            
            future = model.make_future_dataframe(periods=days_ahead, freq='D')
            forecast = model.predict(future)
            
            # Sum next 7 days of forecasted meal sales
            meal_demand = forecast[forecast['ds'] > sales_df['ds'].max()]['yhat'][:days_ahead].sum()
            
            # Convert to ingredient demand
            ingredient_demand = meal_demand * qty_per_meal
            
            # Apply weather impact (average over period)
            avg_impact = sum(weather_impact.get((datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"), 1.0) 
                           for i in range(days_ahead)) / days_ahead
            total_demand += ingredient_demand * avg_impact
        
        return max(0, total_demand)
    except Exception as e:
        logger.warning(f"Forecast failed for {ingredient_name}: {str(e)}")
        return 0.0

def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    """Helper: Get historical sales for a MEAL (not ingredient)"""
    if not supabase:
        return pd.DataFrame()
    try:
        items = supabase.table("order_items")\
            .select("order_id, quantity")\
            .eq("item_name", item_name)\
            .execute()
        if not items.data:
            return pd.DataFrame()
        
        order_ids = [i["order_id"] for i in items.data]
        orders = supabase.table("orders")\
            .select("id, created_at")\
            .in_("id", order_ids)\
            .execute()
        if not orders.data:
            return pd.DataFrame()
        
        df_items = pd.DataFrame(items.data)
        df_orders = pd.DataFrame(orders.data).rename(columns={"id": "order_id"})
        merged = pd.merge(df_items, df_orders, on="order_id", how="inner")
        merged["created_at"] = pd.to_datetime(merged["created_at"])
        cutoff = datetime.now() - timedelta(days=days_back)
        merged = merged[merged["created_at"] >= cutoff]
        
        daily = merged.groupby(merged["created_at"].dt.date)["quantity"].sum().reset_index()
        daily.columns = ["ds", "y"]
        daily["ds"] = pd.to_datetime(daily["ds"])
        return daily
    except Exception as e:
        logger.error(f"Sales fetch error: {str(e)}")
        return pd.DataFrame()

@app.post("/api/usage-history")
async def usage_history(request: UsageHistoryRequest):
    try:
        target_date = datetime.strptime(request.target_date, "%Y-%m-%d").date()
        usage = get_ingredient_usage(target_date)
        amount = usage.get(request.ingredient_name.strip().lower(), 0.0)
        
        return {
            "date": request.target_date,
            "ingredient": request.ingredient_name,
            "amount_used": round(amount, 2),
            "unit": "units"  # Could be enhanced per ingredient later
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")

@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    try:
        today_usage = get_ingredient_usage()
        items_data = []
        total_recommended = 0.0
        critical_count = 0
        
        for item in request.items:
            ing_name = item.item_name.strip()
            ing_key = ing_name.lower()
            
            # 1. CURRENT STOCK
            current_stock = item.current_stock
            if current_stock is None and supabase:
                res = supabase.table("ingredient_stock")\
                    .select("current_stock")\
                    .ilike("ingredient_name", ing_name)\
                    .limit(1)\
                    .execute()
                current_stock = float(res.data[0]["current_stock"]) if res.data else 0.0
            
            # 2. DAILY USAGE (TODAY)
            daily_used = today_usage.get(ing_key, 0.0)
            
            # 3. WEEKLY DEMAND (CALCULATED FROM MEAL FORECASTS)
            weekly_demand = calculate_ingredient_weekly_demand(ing_name, days_ahead=7)
            if weekly_demand == 0:  # Fallback if forecast fails
                fallbacks = {
                    "chips": 500.0, "melted cheese": 50.0, "russian": 12.0, "lettuce": 8.0,
                    "bread": 25.0, "tomato": 10.0, "atchar": 5.0, "vienna": 15.0,
                    "egg": 20.0, "steak": 6.0
                }
                weekly_demand = fallbacks.get(ing_key, 10.0)
            
            # 4. METRICS
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
                "item_name": ing_name,
                "current_stock": round(current_stock, 1),
                "daily_usage": round(daily_used, 1),
                "weekly_demand": round(weekly_demand, 1),
                "days_left": round(days_left, 1),
                "recommended_order": round(recommended_order, 1),
                "urgency": urgency,
                "status": status,
                "action": "REORDER NOW" if urgency == "HIGH" else "Monitor stock"
            })
        
        # Sort by urgency
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
        logger.error(f"Dashboard error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if supabase else "disconnected"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
