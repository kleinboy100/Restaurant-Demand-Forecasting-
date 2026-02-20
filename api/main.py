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
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GLOBAL VARIABLES & MODELS
# ---------------------------------------------------------------------------
supabase: Optional[Client] = None

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
@app.on_event("startup")
async def startup_event():
    global supabase
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in environment variables")
            return
        
        supabase = create_client(supabase_url, supabase_key)
        logger.info("Database connected successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {str(e)}")

# ---------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------
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

def get_ingredient_usage(target_date: Optional[date] = None):
    """Calculates usage using strict UTC time boundaries and Fuzzy String Matching"""
    if not supabase: return {}, {"error": "Database not connected"}
    
    try:
        sast_tz = timezone(timedelta(hours=2))
        if target_date is None:
            target_date = datetime.now(sast_tz).date()
            
        # 1. TIMEZONE HANDLING: Convert SAST boundaries strictly to UTC
        start_sast = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=sast_tz)
        end_sast = start_sast + timedelta(days=1)
        
        start_utc = start_sast.astimezone(timezone.utc).isoformat()
        end_utc = end_sast.astimezone(timezone.utc).isoformat()
        
        # 2. FETCH ORDERS
        orders_res = supabase.table("orders").select("id").gte("created_at", start_utc).lt("created_at", end_utc).execute()
        order_ids = [o["id"] for o in orders_res.data] if orders_res.data else []
        
        if not order_ids: 
            return {}, {"orders_found": 0, "date_checked": start_utc}
            
        # 3. FETCH ITEMS
        items_res = supabase.table("order_items").select("item_name, quantity").in_("order_id", order_ids).execute()
        order_items = items_res.data if items_res.data else []
            
        # 4. FETCH RECIPES
        recipes_res = supabase.table("meal_recipes").select("meal_name, ingredient_name, quantity_per_meal").execute()
        recipe_data = recipes_res.data if recipes_res.data else []
        
        # Group recipes by lowercased meal name
        recipes = {}
        for r in recipe_data:
            meal = r["meal_name"].strip().lower()
            if meal not in recipes: recipes[meal] = []
            recipes[meal].append(r)
            
        # 5. CALCULATE (WITH FUZZY MATCHING)
        usage_dict = {}
        unmatched_meals = []
        matched_meals_count = 0
        
        for item in order_items:
            # Safely grab strings and floats
            order_meal_name = str(item.get("item_name", "")).strip().lower()
            try: qty = float(item.get("quantity", 1))
            except: qty = 1.0
            
            matched_recipe_key = None
            
            # Direct match first (e.g. "flamwood" == "flamwood")
            if order_meal_name in recipes:
                matched_recipe_key = order_meal_name
            else:
                # Fuzzy Match: Check if the recipe name is INSIDE the order name 
                # e.g., recipe="flamwood", order="flamwood kota combo" -> match!
                for recipe_key in recipes.keys():
                    if recipe_key in order_meal_name or order_meal_name in recipe_key:
                        matched_recipe_key = recipe_key
                        break
            
            if matched_recipe_key:
                matched_meals_count += 1
                for ing in recipes[matched_recipe_key]:
                    ing_name = ing["ingredient_name"].strip().lower()
                    ing_qty = float(ing["quantity_per_meal"])
                    usage_dict[ing_name] = usage_dict.get(ing_name, 0.0) + (ing_qty * qty)
            else:
                unmatched_meals.append(order_meal_name)
                
        # Prepare debug info to send back to the dashboard
        debug_info = {
            "orders_found": len(order_ids),
            "total_items_in_orders": len(order_items),
            "recipes_loaded_from_db": len(recipe_data),
            "successful_meal_matches": matched_meals_count,
            "unmatched_meals_ignored": list(set(unmatched_meals))
        }
        
        return usage_dict, debug_info
        
    except Exception as e:
        logger.error(f"Error calculating usage: {str(e)}")
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

# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "KOTAai API", "status": "online", "version": "2.4.0"}

@app.get("/health")
async def health_check():
    health_status = {"status": "healthy", "timestamp": datetime.now().isoformat()}
    if supabase:
        try:
            supabase.table("ingredient_stock").select("*").limit(1).execute()
            health_status["database"] = "connected"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
    return health_status

@app.post("/api/usage-history")
async def historical_usage(request: UsageHistoryRequest):
    """Analyzes historical usage with debugging metrics attached"""
    try:
        target_date = datetime.strptime(request.target_date, "%Y-%m-%d").date()
        usage_dict, debug_info = get_ingredient_usage(target_date)
        
        response_data = {
            "date": request.target_date,
            "ingredient": request.ingredient_name,
            "all_data": {k: round(v, 2) for k, v in usage_dict.items()},
            "debug_info": debug_info
        }
        
        if request.ingredient_name and request.ingredient_name != "All":
            lookup = request.ingredient_name.strip().lower()
            response_data["amount_used"] = round(usage_dict.get(lookup, 0.0), 2)
            
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    """Main dashboard endpoint"""
    try:
        items_data = []
        total_recommended = 0
        critical_count = 0
        
        # 1. GET TODAY'S USAGE
        today_usage, debug_info = get_ingredient_usage()
        logger.info(f"Dashboard usage debug: {debug_info}")
        
        fallback_demands = {"Chips": 500.0, "Melted Cheese": 50.0, "Russian": 12.0, "lettuce": 8.0, "Bread": 25.0, "tomato": 10.0, "atchar": 5.0, "Vienna": 15.0, "egg": 20.0, "steak": 6.0}
        
        for item in request.items:
            try:
                lookup_name = item.item_name.strip()
                
                # STOCK
                current_stock = item.current_stock
                if current_stock is None and supabase:
                    stock_res = supabase.table("ingredient_stock").select("current_stock").ilike("ingredient_name", lookup_name).limit(1).execute()
                    current_stock = stock_res.data[0]["current_stock"] if stock_res.data else 0
                
                # FORECAST
                forecast_df = generate_world_class_forecast(item.item_name, 7)
                weekly_demand = forecast_df["final_prediction"].sum() if forecast_df is not None else fallback_demands.get(lookup_name, 5.0)
                
                # DAILY USAGE
                daily_used = today_usage.get(lookup_name.lower(), 0.0)
                
                # METRICS
                daily_demand = weekly_demand / 7
                days_left = current_stock / daily_demand if daily_demand > 0 else 999
                recommended_order = max(0, (weekly_demand * 1.5) - current_stock)
                
                if days_left < 3:
                    urgency, status, critical_count = "HIGH", "CRITICAL", critical_count + 1
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

@app.post("/reorder-recommendations") 
async def reorder_recommendations(request: DashboardRequest):
    return await dashboard_data(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
