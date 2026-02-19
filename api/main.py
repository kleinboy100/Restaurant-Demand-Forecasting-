import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from prophet import Prophet
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import requests
from uuid import uuid4

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
    version="2.1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; replace "*" with specific URLs if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------------------------------
supabase: Optional[Client] = None

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
        # Test connection with existing table
        test_result = supabase.table("ingredient_stock").select("id").limit(1).execute()
        logger.info(f"Database connected successfully. Test query returned {len(test_result.data)} rows")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    if init_supabase():
        asyncio.create_task(listen_for_new_orders())

# ---------------------------------------------------------------------------
# LISTEN FOR NEW ORDERS USING SUPABASE REALTIME
# ---------------------------------------------------------------------------
async def listen_for_new_orders():
    """Listen for new orders and process them automatically"""
    try:
        from supabase_py_async import create_client as create_async_client
        async_supabase = await create_async_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
        
        # Subscribe to changes in the 'orders' table
        subscription = await async_supabase.table('orders').on(
            'INSERT',
            lambda payload: process_new_order(payload['new'])
        ).subscribe()

        logger.info("Subscribed to 'orders' table for real-time updates.")
        
        # Keep the subscription alive
        while True:
            await asyncio.sleep(60)  # Keep the task running

    except Exception as e:
        logger.error(f"Error setting up Realtime listener: {str(e)}")

async def process_new_order(order):
    """Process a new order when detected via Realtime"""
    try:
        order_id = order.get('id')
        if not order_id:
            logger.warning("Received order without an ID. Skipping.")
            return
        
        # Fetch order items associated with this order
        order_items_result = supabase.table("order_items") \
                                    .select("*") \
                                    .eq("order_id", order_id) \
                                    .execute()
        
        if not order_items_result.data:
            logger.warning(f"No order items found for order ID: {order_id}")
            return
        
        for item in order_items_result.data:
            meal_name = item.get('item_name')
            quantity = item.get('quantity', 1)
            
            if not meal_name or quantity <= 0:
                continue
            
            # Get meal recipe from database
            recipe_result = supabase.table("meal_recipes") \
                                  .select("*") \
                                  .eq("meal_name", meal_name) \
                                  .execute()
            
            if not recipe_result.data:
                logger.warning(f"Meal {meal_name} not found in recipes. Skipping.")
                continue
            
            # Process each ingredient in the recipe
            for recipe in recipe_result.data:
                ingredient = recipe["ingredient_name"]
                quantity_per_meal = recipe["quantity_per_meal"]
                unit = recipe["unit"]
                
                # Calculate total quantity used
                quantity_used = quantity * quantity_per_meal
                
                # Get current stock for the ingredient
                stock_result = supabase.table("ingredient_stock") \
                                    .select("current_stock, unit") \
                                    .eq("ingredient_name", ingredient) \
                                    .execute()
                
                if stock_result.data:
                    current_stock = stock_result.data[0]["current_stock"]
                    stock_unit = stock_result.data[0]["unit"]
                    
                    # Handle unit conversions if needed
                    if unit != stock_unit:
                        logger.warning(f"Unit mismatch for {ingredient}: recipe {unit} vs stock {stock_unit}")
                    
                    # Update current_stock
                    new_stock = max(0, current_stock - quantity_used)
                    
                    supabase.table("ingredient_stock") \
                           .update({"current_stock": new_stock}) \
                           .eq("ingredient_name", ingredient) \
                           .execute()
                    
                    # Record ingredient usage
                    meal_ingredient_data = {
                        "id": str(uuid4()),
                        "order_id": order_id,
                        "meal_name": meal_name,
                        "ingredient_name": ingredient,
                        "quantity_used": quantity_used,
                        "unit": unit,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    supabase.table("meal_ingredients").insert(meal_ingredient_data).execute()
                    
                    logger.info(f"Updated {ingredient}: {current_stock} -> {new_stock} (used {quantity_used} {unit})")
                else:
                    logger.warning(f"Ingredient {ingredient} not found in stock")
        
        logger.info(f"Processed new order ID: {order_id}")

    except Exception as e:
        logger.error(f"Error processing new order: {str(e)}")

# ---------------------------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------------------------
class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[int] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "KOTAai API", "status": "online", "version": "2.1.0"}

@app.get("/health")
async def health_check():
    health_status = {"status": "healthy", "timestamp": datetime.now().isoformat()}
    if supabase:
        try:
            supabase.table("ingredient_stock").select("*").limit(1).execute()
            health_status["database"] = "connected"
            health_status["stock_table"] = "ingredient_stock"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
    return health_status

@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    """Main dashboard endpoint with real-time stock and depleted quantity"""
    try:
        items_data = []
        total_recommended = 0
        critical_count = 0
        
        # Get today's usage for all ingredients
        today = datetime.now().strftime("%Y-%m-%d")
        today_usage_result = supabase.table("meal_ingredients") \
                                   .select("*") \
                                   .ilike("created_at", f"{today}%") \
                                   .execute()
        
        today_usage = {}
        for record in today_usage_result.data:
            ingredient = record["ingredient_name"]
            if ingredient not in today_usage:
                today_usage[ingredient] = 0
            today_usage[ingredient] += record["quantity_used"]
        
        for item in request.items:
            try:
                # 1. FETCH CURRENT STOCK from ingredient_stock
                current_stock = item.current_stock
                if current_stock is None and supabase:
                    try:
                        lookup = item.item_name.strip()
                        stock_result = supabase.table("ingredient_stock") \
                            .select("ingredient_name, current_stock, unit") \
                            .ilike("ingredient_name", lookup) \
                            .limit(1) \
                            .execute()
                        
                        if stock_result.data:
                            current_stock = stock_result.data[0]["current_stock"]
                        else:
                            current_stock = 0
                    except Exception as e:
                        logger.warning(f"Error reading ingredient_stock for {item.item_name}: {str(e)}")
                        current_stock = 0
                
                # 2. GET TODAY'S USAGE
                today_usage_amount = today_usage.get(item.item_name.strip(), 0)
                
                # 3. GENERATE FORECAST
                forecast_df = generate_world_class_forecast(item.item_name, 7)
                weekly_demand = forecast_df["final_prediction"].sum() if forecast_df is not None else 5.0
                
                # 4. CALCULATE METRICS
                daily_demand = weekly_demand / 7
                days_left = (current_stock - today_usage_amount) / daily_demand if daily_demand > 0 else 999
                recommended_order = max(0, (weekly_demand * 1.5) - (current_stock - today_usage_amount))
                
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
                    "today_usage": round(today_usage_amount, 2),
                    "weekly_demand": round(weekly_demand, 1),
                    "days_left": round(days_left, 1),
                    "recommended_order": round(recommended_order, 1),
                    "urgency": urgency,
                    "status": status,
                    "action": "REORDER NOW" if urgency == "HIGH" else "Monitor stock"
                })
                
            except Exception as e:
                logger.error(f"Error processing {item.item_name}: {str(e)}")
                items_data.append({"item_name": item.item_name, "current_stock": 0, "today_usage": 0, "weekly_demand": 0, "days_left": 0, "recommended_order": 0, "urgency": "HIGH", "status": "ERROR", "action": "Check data"})
        
        # Sort items by urgency
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
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_world_class_forecast(item_name: str, days_ahead: int) -> Optional[pd.DataFrame]:
    """Generate forecast using Prophet"""
    df = get_sales_from_order_items(item_name)
    if df.empty:
        return None

    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name="ZA")
    model.fit(df)

    future = model.make_future_dataframe(periods=days_ahead, freq='D')
    weather_impact = get_klerksdorp_weather(days_ahead)
    future["impact_score"] = [weather_impact.get(d.strftime("%Y-%m-%d"), 1.0) for d in future["ds"]]
    
    forecast = model.predict(future)
    forecast["final_prediction"] = (forecast["yhat"] * future["impact_score"]).clip(lower=0)
    return forecast[forecast["ds"] > df["ds"].max()].copy()

def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    """Fetch sales data for an item"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    try:
        order_items_result = supabase.table("order_items").select("order_id, item_name, quantity").eq("item_name", item_name).execute()
        if not order_items_result.data:
            return pd.DataFrame()
        
        order_ids = [item["order_id"] for item in order_items_result.data]
        orders_result = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        if not orders_result.data:
            return pd.DataFrame()
        
        order_items_df = pd.DataFrame(order_items_result.data)
        orders_df = pd.DataFrame(orders_result.data).rename(columns={"id": "order_id"})
        merged_df = pd.merge(order_items_df, orders_df, on="order_id", how="inner")
        merged_df["created_at"] = pd.to_datetime(merged_df["created_at"])
        cutoff_date = datetime.now() - timedelta(days=days_back)
        merged_df = merged_df[merged_df["created_at"] >= cutoff_date]
        if merged_df.empty:
            return pd.DataFrame()
        
        merged_df["sale_date"] = merged_df["created_at"].dt.date
        daily_sales = merged_df.groupby("sale_date")["quantity"].sum().reset_index()
        daily_sales.rename(columns={"sale_date": "ds", "quantity": "y"}, inplace=True)
        daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])
        return daily_sales
    except Exception as e:
        logger.error(f"Error fetching sales data: {str(e)}")
        return pd.DataFrame()

def get_klerksdorp_weather(days_ahead: int = 7) -> Dict[str, float]:
    """Fetch weather impact scores for Klerksdorp"""
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
        logger.warning(f"Weather API failed: {str(e)}")
        future_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days_ahead + 1)]
        return {date_str: 1.0 for date_str in future_dates}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
