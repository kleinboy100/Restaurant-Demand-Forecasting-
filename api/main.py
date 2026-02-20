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

app = FastAPI(title="KOTAai API", version="2.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Optional[Client] = None

class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[int] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

class UsageHistoryRequest(BaseModel):
    target_date: str
    ingredient_name: Optional[str] = "All"

@app.on_event("startup")
async def startup_event():
    global supabase
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Database connected successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {str(e)}")

def get_klerksdorp_weather(days_ahead: int = 7) -> Dict[str, float]:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": -26.85, "longitude": 26.66, "daily": "precipitation_probability_max", "forecast_days": days_ahead, "timezone": "Africa/Johannesburg"}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        weather_impact = {}
        for i, date_str in enumerate(data["daily"]["time"]):
            prob = data["daily"]["precipitation_probability_max"][i]
            weather_impact[date_str] = 1.0 if prob < 20 else (0.85 if prob < 60 else 0.7)
        return weather_impact
    except Exception as e:
        logger.warning(f"Weather API failed: {str(e)}")
        future_dates = [(datetime.now(tz=timezone.utc) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days_ahead + 1)]
        return {date_str: 1.0 for date_str in future_dates}

def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    if not supabase: return pd.DataFrame()
    try:
        order_items_result = supabase.table("order_items").select("order_id, item_name, quantity").eq("item_name", item_name).execute()
        if not order_items_result.data: return pd.DataFrame()
        
        order_ids = [item["order_id"] for item in order_items_result.data]
        orders_result = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        
        order_items_df = pd.DataFrame(order_items_result.data)
        orders_df = pd.DataFrame(orders_result.data).rename(columns={"id": "order_id"})
        merged_df = pd.merge(order_items_df, orders_df, on="order_id", how="inner")
        merged_df["created_at"] = pd.to_datetime(merged_df["created_at"])
        cutoff_date = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
        merged_df = merged_df[merged_df["created_at"] >= cutoff_date]

        daily_sales = merged_df.groupby(merged_df["created_at"].dt.date)["quantity"].sum().reset_index()
        daily_sales.columns = ["ds", "y"]
        daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])
        return daily_sales
    except Exception as e:
        logger.error(f"Error fetching sales data: {str(e)}")
        return pd.DataFrame()

def get_ingredient_usage(target_date: Optional[date] = None) -> tuple[Dict[str, float], Dict]:
    """Returns (usage_by_ingredient, debug_info). Uses UTC-aligned window for Supabase."""
    if not supabase:
        return {}, {"error": "Supabase not initialized"}
    
    try:
        sast_tz = timezone(timedelta(hours=2))
        if target_date is None:
            target_date = datetime.now(sast_tz).date()

        # Convert SAST date to UTC ISO strings for Supabase query
        start_sast = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=sast_tz)
        end_sast = start_sast + timedelta(days=1)
        start_utc_iso = start_sast.astimezone(timezone.utc).isoformat()
        end_utc_iso = end_sast.astimezone(timezone.utc).isoformat()

        # 1. Get orders from today (UTC window)
        orders_res = supabase.table("orders").select("id").gte("created_at", start_utc_iso).lt("created_at", end_utc_iso).execute()
        order_ids = [o["id"] for o in orders_res.data] if orders_res.data else []

        if not order_ids:
            return {}, {
                "orders_found": 0,
                "date_utc_start": start_utc_iso,
                "date_utc_end": end_utc_iso,
                "recipes_loaded": 0,
                "matches": 0,
                "unmatched_items": []
            }

        # 2. Get all order_items for these orders
        items_res = supabase.table("order_items").select("item_name, quantity").in_("order_id", order_ids).execute()
        order_items = items_res.data if items_res.data else []

        # 3. Load all recipes (RLS should allow read now)
        recipes_res = supabase.table("meal_recipes").select("meal_name, ingredient_name, quantity_per_meal").execute()
        recipes_data = recipes_res.data if recipes_res.data else []

        # Build meal → ingredients mapping
        recipes_map = {}
        for rec in recipes_data:
            meal = rec["meal_name"].strip().lower()
            if meal not in recipes_map:
                recipes_map[meal] = []
            recipes_map[meal].append({
                "ingredient": rec["ingredient_name"].strip().lower(),
                "qty": float(rec["quantity_per_meal"])
            })

        # 4. Match order_items to recipes (fuzzy matching)
        usage = {}
        unmatched_items = []
        matches = 0

        for item in order_items:
            order_meal = item.get("item_name", "").strip().lower()
            try: quantity = float(item.get("quantity", 1))
            except: quantity = 1.0

            # Try exact match
            if order_meal in recipes_map:
                matches += 1
                for ing in recipes_map[order_meal]:
                    ing_name = ing["ingredient"]
                    usage[ing_name] = usage.get(ing_name, 0.0) + (ing["qty"] * quantity)
            else:
                # Fuzzy: Does recipe name appear inside the order item name?
                matched = False
                for recipe_name in recipes_map.keys():
                    if recipe_name in order_meal:
                        matches += 1
                        matched = True
                        for ing in recipes_map[recipe_name]:
                            ing_name = ing["ingredient"]
                            usage[ing_name] = usage.get(ing_name, 0.0) + (ing["qty"] * quantity)
                        break
                if not matched:
                    unmatched_items.append(order_meal)

        debug_info = {
            "orders_found": len(order_ids),
            "items_in_orders": len(order_items),
            "recipes_loaded": len(recipes_data),
            "successful_matches": matches,
            "unmatched_items": list(set(unmatched_items))
        }

        return usage, debug_info

    except Exception as e:
        logger.error(f"Error in get_ingredient_usage: {str(e)}")
        return {}, {"error": str(e)}

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

@app.post("/api/usage-history")
async def historical_usage(request: UsageHistoryRequest):
    try:
        target_date = datetime.strptime(request.target_date, "%Y-%m-%d").date()
        usage_dict, debug_info = get_ingredient_usage(target_date)
        
        response = {
            "date": request.target_date,
            "all_data": {k: round(v, 2) for k, v in usage_dict.items()},
            "debug_info": debug_info
        }
        
        if request.ingredient_name != "All":
            key = request.ingredient_name.strip().lower()
            response["amount_used"] = round(usage_dict.get(key, 0.0), 2)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    try:
        items_data = []
        total_recommended, critical_count = 0, 0

        # --- NEW: Calculate daily usage for ALL ingredients used today ---
        daily_usage, debug_info = get_ingredient_usage()
        logger.info(f"Dashboard usage debug: {debug_info}")  # Check server logs!

        # Fallback demands (for forecast if no sales history)
        fallback_demands = {
            "chips": 500.0,
            "melted cheese": 50.0,
            "russian": 12.0,
            "lettuce": 8.0,
            "bread": 25.0,
            "tomato": 10.0,
            "atchar": 5.0,
            "vienna": 15.0,
            "egg": 20.0,
            "steak": 6.0
        }

        for item in request.items:
            try:
                item_name_clean = item.item_name.strip().lower()
                current_stock = item.current_stock
                if current_stock is None and supabase:
                    res = supabase.table("ingredient_stock").select("current_stock").ilike("ingredient_name", item.item_name.strip()).limit(1).execute()
                    current_stock = res.data[0]["current_stock"] if res.data else 0

                forecast_df = generate_world_class_forecast(item.item_name.strip(), 7)
                weekly_demand = forecast_df["final_prediction"].sum() if forecast_df is not None else fallback_demands.get(item_name_clean, 5.0)

                daily_used = daily_usage.get(item_name_clean, 0.0)
                daily_demand = weekly_demand / 7
                days_left = current_stock / daily_demand if daily_demand > 0 else 999
                recommended = max(0, (weekly_demand * 1.5) - current_stock)

                if days_left < 3:
                    urgency, status, critical_count = "HIGH", "CRITICAL", critical_count + 1
                elif days_left < 7:
                    urgency, status = "MEDIUM", "LOW"
                else:
                    urgency, status = "LOW", "OK"

                total_recommended += recommended
                items_data.append({
                    "item_name": item.item_name,
                    "current_stock": current_stock,
                    "daily_usage": round(daily_used, 2),  # <-- NEW! Daily usage since midnight SAST
                    "weekly_demand": round(weekly_demand, 1),
                    "days_left": round(days_left, 1),
                    "recommended_order": round(recommended, 1),
                    "urgency": urgency,
                    "status": status,
                    "action": "REORDER NOW" if urgency == "HIGH" else "Monitor stock"
                })

            except Exception as e:
                logger.error(f"Error processing {item.item_name}: {str(e)}")
                items_data.append({
                    "item_name": item.item_name,
                    "current_stock": 0,
                    "daily_usage": 0,
                    "weekly_demand": 0,
                    "days_left": 0,
                    "recommended_order": 0,
                    "urgency": "HIGH",
                    "status": "ERROR",
                    "action": "Check data"
                })

        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        items_data.sort(key=lambda x: urgency_order.get(x["urgency"], 3))

        return {
            "summary": {
                "total_items": len(items_data),
                "critical_items": critical_count,
                "total_recommended": round(total_recommended, 1),
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            },
            "items": items_data
        }
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "connected" if supabase else "disconnected"
    }
    try:
        if supabase:
            # Verify access to all critical tables
            supabase.table("ingredient_stock").select("*").limit(1).execute()
            supabase.table("meal_recipes").select("*").limit(1).execute()
            health_status["tables_check"] = "ingredient_stock, meal_recipes"
    except Exception as e:
        health_status["tables_check"] = f"error: {str(e)}"
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
