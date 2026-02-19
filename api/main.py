import os
import logging
import asyncio
from datetime import datetime, timedelta, date
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
# MEAL-INGREDIENT MAPPING
# ---------------------------------------------------------------------------
MEAL_INGREDIENTS = {
    "Flamwood": {"Chips": 150, "Melted Cheese": 2, "Russian": 1, "lettuce": 50, "Bread": 0.25, "tomato": 1},
    "Stop 5": {"Chips": 150, "Melted Cheese": 1, "Russian": 1, "lettuce": 50, "Bread": 0.25, "tomato": 1, "atchar": 50, "Vienna": 1, "egg": 1},
    "Stop 18": {"Chips": 150, "Melted Cheese": 1, "lettuce": 50, "Bread": 0.25, "tomato": 1, "atchar": 50},
    "Phelandaba": {"Chips": 150, "Melted Cheese": 1, "Russian": 1, "lettuce": 50, "Bread": 0.25, "tomato": 1, "atchar": 50, "egg": 1},
    "Steak": {"steak": 100},
    "Toast": {"Bread": 4, "Chips": 150}
}

# ---------------------------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    item_name: str
    days_ahead: int = 7

class RecommendationRequest(BaseModel):
    item_name: str
    current_stock: Optional[int] = None

class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[int] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

class OrderMealRequest(BaseModel):
    order_id: str
    meal_name: str
    quantity: int

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def get_today_date_range():
    """Get the start and end of today as ISO formatted strings"""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    return today_start.isoformat(), today_end.isoformat()

def get_today_usage_from_db():
    """Get today's ingredient usage from the meal_ingredients table"""
    if not supabase:
        return {}
    
    try:
        today_start, today_end = get_today_date_range()
        
        # Use gte and lt for proper timestamp filtering
        today_usage_result = supabase.table("meal_ingredients") \
                                   .select("*") \
                                   .gte("created_at", today_start) \
                                   .lt("created_at", today_end) \
                                   .execute()
        
        today_usage = {}
        for record in today_usage_result.data:
            ingredient = record["ingredient_name"]
            if ingredient not in today_usage:
                today_usage[ingredient] = 0
            today_usage[ingredient] += float(record["quantity_used"])
        
        return today_usage
    except Exception as e:
        logger.error(f"Error getting today's usage: {str(e)}")
        return {}

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

# ---------------------------------------------------------------------------
# WEATHER DATA INTEGRATION
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
        logger.warning(f"Weather API failed: {str(e)}")
        future_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days_ahead + 1)]
        return {date_str: 1.0 for date_str in future_dates}

# ---------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------
def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    if not supabase:
        logger.error("Database not configured")
        return pd.DataFrame()
    
    try:
        order_items_result = supabase.table("order_items").select("order_id, item_name, quantity").eq("item_name", item_name).execute()
        if not order_items_result.data:
            logger.info(f"No order items found for {item_name}")
            return pd.DataFrame()
        
        order_ids = [item["order_id"] for item in order_items_result.data]
        orders_result = supabase.table("orders").select("id, created_at").in_("id", order_ids).execute()
        if not orders_result.data:
            logger.info(f"No orders found for order IDs: {order_ids}")
            return pd.DataFrame()
        
        order_items_df = pd.DataFrame(order_items_result.data)
        orders_df = pd.DataFrame(orders_result.data).rename(columns={"id": "order_id"})
        merged_df = pd.merge(order_items_df, orders_df, on="order_id", how="inner")
        merged_df["created_at"] = pd.to_datetime(merged_df["created_at"])
        cutoff_date = datetime.now() - timedelta(days=days_back)
        merged_df = merged_df[merged_df["created_at"] >= cutoff_date]
        if merged_df.empty:
            logger.info(f"No sales data for {item_name} in the last {days_back} days")
            return pd.DataFrame()
        
        merged_df["sale_date"] = merged_df["created_at"].dt.date
        daily_sales = merged_df.groupby("sale_date")["quantity"].sum().reset_index()
        daily_sales.rename(columns={"sale_date": "ds", "quantity": "y"}, inplace=True)
        daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])
        logger.info(f"Sales data for {item_name}: {len(daily_sales)} days")
        return daily_sales
    except Exception as e:
        logger.error(f"Error fetching sales data for {item_name}: {str(e)}")
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# AI FORECAST ENGINE
# ---------------------------------------------------------------------------
def generate_world_class_forecast(item_name: str, days_ahead: int) -> Optional[pd.DataFrame]:
    if not supabase:
        logger.error("Database not configured")
        return None
    
    df = get_sales_from_order_items(item_name)
    if df.empty:
        logger.warning(f"No sales data available for {item_name}, using default forecast")
        return None

    try:
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        model.add_country_holidays(country_name="ZA")
        model.fit(df)

        future = model.make_future_dataframe(periods=days_ahead, freq='D')
        weather_impact = get_klerksdorp_weather(days_ahead)
        future["impact_score"] = [weather_impact.get(d.strftime("%Y-%m-%d"), 1.0) for d in future["ds"]]
        
        forecast = model.predict(future)
        forecast["final_prediction"] = (forecast["yhat"] * future["impact_score"]).clip(lower=0)
        return forecast[forecast["ds"] > df["ds"].max()].copy()
    except Exception as e:
        logger.error(f"Error generating forecast for {item_name}: {str(e)}")
        return None

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
            supabase.table("ingredient_stock").select("id").limit(1).execute()
            health_status["database"] = "connected"
            health_status["stock_table"] = "ingredient_stock"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
    else:
        health_status["database"] = "not connected"
    return health_status

@app.post("/api/process-meal-order")
async def process_meal_order(order: OrderMealRequest):
    """Process a meal order and update ingredient stock with unit handling"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        # Get meal recipe from database
        recipe_result = supabase.table("meal_recipes") \
                              .select("*") \
                              .eq("meal_name", order.meal_name) \
                              .execute()
        
        if not recipe_result.data:
            # Fall back to hardcoded recipes if not in database
            if order.meal_name in MEAL_INGREDIENTS:
                ingredients = MEAL_INGREDIENTS[order.meal_name]
            else:
                raise HTTPException(status_code=404, detail=f"Meal {order.meal_name} not found in recipes")
        else:
            ingredients = {r["ingredient_name"]: r["quantity_per_meal"] for r in recipe_result.data}
        
        # Process each ingredient in the recipe
        total_depleted = 0
        for ingredient, quantity_per_meal in ingredients.items():
            # Calculate total quantity used
            quantity_used = order.quantity * quantity_per_meal
            
            # Get current stock for the ingredient
            stock_result = supabase.table("ingredient_stock") \
                                .select("current_stock, unit") \
                                .eq("ingredient_name", ingredient) \
                                .execute()
            
            if stock_result.data:
                current_stock = float(stock_result.data[0]["current_stock"])
                unit = stock_result.data[0].get("unit", "units")
                
                # Update current_stock
                new_stock = max(0, current_stock - quantity_used)
                
                supabase.table("ingredient_stock") \
                       .update({"current_stock": new_stock, "last_updated": datetime.now().isoformat()}) \
                       .eq("ingredient_name", ingredient) \
                       .execute()
                
                # Record ingredient usage
                meal_ingredient_data = {
                    "id": str(uuid4()),
                    "order_id": order.order_id,
                    "meal_name": order.meal_name,
                    "ingredient_name": ingredient,
                    "quantity_used": quantity_used,
                    "unit": unit,
                    "created_at": datetime.now().isoformat()
                }
                
                supabase.table("meal_ingredients").insert(meal_ingredient_data).execute()
                
                total_depleted += quantity_used
                logger.info(f"Updated {ingredient}: {current_stock} -> {new_stock} (used {quantity_used} {unit})")
            else:
                logger.warning(f"Ingredient {ingredient} not found in stock")
        
        return {
            "status": "success",
            "order_id": order.order_id,
            "meal_name": order.meal_name,
            "quantity": order.quantity,
            "total_depleted": round(total_depleted, 2)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing meal order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/real-time-stock")
async def get_real_time_stock():
    """Get current stock levels for all ingredients"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        stock_result = supabase.table("ingredient_stock").select("*").execute()
        
        # Get usage for today using proper date range filtering
        today_usage = get_today_usage_from_db()
        
        # Combine stock and usage data
        stock_data = []
        for item in stock_result.data:
            ingredient_name = item["ingredient_name"]
            current_stock = float(item["current_stock"])
            usage = today_usage.get(ingredient_name, 0)
            
            stock_data.append({
                "ingredient_name": ingredient_name,
                "current_stock": current_stock,
                "unit": item.get("unit", "units"),
                "min_stock_level": item.get("min_stock_level"),
                "max_stock_level": item.get("max_stock_level"),
                "today_usage": usage,
                "remaining_stock": max(0, current_stock)
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "stock_data": stock_data
        }
    
    except Exception as e:
        logger.error(f"Error getting real-time stock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    """Main dashboard endpoint with real-time stock and depleted quantity"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        logger.info("Processing dashboard request")
        items_data = []
        total_recommended = 0
        critical_count = 0
        
        # Get today's usage using proper date range filtering
        today_usage = get_today_usage_from_db()
        logger.info(f"Today's usage data: {today_usage}")
        
        for item in request.items:
            try:
                logger.info(f"Processing item: {item.item_name}")
                # 1. FETCH CURRENT STOCK from ingredient_stock
                current_stock = item.current_stock
                if current_stock is None and supabase:
                    try:
                        lookup = item.item_name.strip()
                        logger.info(f"Looking up stock for: {lookup}")
                        stock_result = supabase.table("ingredient_stock") \
                            .select("ingredient_name, current_stock, unit") \
                            .eq("ingredient_name", lookup) \
                            .limit(1) \
                            .execute()
                        
                        if stock_result.data:
                            current_stock = float(stock_result.data[0]["current_stock"])
                            logger.info(f"Found stock: {current_stock}")
                        else:
                            current_stock = 0
                            logger.warning(f"No stock found for {lookup}")
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
                items_data.append({
                    "item_name": item.item_name, 
                    "current_stock": 0, 
                    "today_usage": 0, 
                    "weekly_demand": 0, 
                    "days_left": 0, 
                    "recommended_order": 0, 
                    "urgency": "HIGH", 
                    "status": "ERROR", 
                    "action": "Check data"
                })
        
        # Sort items by urgency
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        items_data.sort(key=lambda x: urgency_order.get(x["urgency"], 3))
        
        logger.info("Dashboard data processing completed")
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

@app.post("/api/auto-process-orders")
async def auto_process_orders():
    """Automatically process new orders and update ingredient stock"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        # Get today's date range for filtering
        today_start, today_end = get_today_date_range()
        
        # Get recent orders using proper date range filtering
        orders_result = supabase.table("orders") \
                               .select("*") \
                               .gte("created_at", today_start) \
                               .lt("created_at", today_end) \
                               .execute()
        
        # Get already processed orders to avoid duplicate processing
        processed_orders_result = supabase.table("meal_ingredients") \
                                        .select("order_id") \
                                        .gte("created_at", today_start) \
                                        .lt("created_at", today_end) \
                                        .execute()
        
        processed_order_ids = set(r["order_id"] for r in processed_orders_result.data)
        
        processed_count = 0
        skipped_count = 0
        
        for order in orders_result.data:
            order_id = str(order["id"])
            
            # Skip if already processed
            if order_id in processed_order_ids:
                skipped_count += 1
                continue
            
            # Get order items for this order
            order_items_result = supabase.table("order_items") \
                                       .select("*") \
                                       .eq("order_id", order["id"]) \
                                       .execute()
            
            for item in order_items_result.data:
                meal_name = item.get("item_name")
                quantity = item.get("quantity", 1)
                
                if not meal_name or quantity <= 0:
                    continue
                
                # Check if this meal is in our recipes
                if meal_name not in MEAL_INGREDIENTS:
                    # Check database recipes
                    recipe_check = supabase.table("meal_recipes") \
                                         .select("id") \
                                         .eq("meal_name", meal_name) \
                                         .limit(1) \
                                         .execute()
                    if not recipe_check.data:
                        logger.warning(f"Meal {meal_name} not found in recipes. Skipping.")
                        continue
                
                # Process the meal order
                try:
                    order_request = OrderMealRequest(
                        order_id=order_id,
                        meal_name=meal_name,
                        quantity=quantity
                    )
                    
                    # Call the existing process_meal_order function
                    await process_meal_order(order_request)
                    processed_count += 1
                    logger.info(f"Processed order {order_id}: {quantity}x {meal_name}")
                    
                except Exception as e:
                    logger.error(f"Error processing order item {item.get('id')}: {str(e)}")
        
        return {
            "status": "success",
            "processed_orders": processed_count,
            "skipped_orders": skipped_count,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in auto-process-orders: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reorder-recommendations") 
async def reorder_recommendations(request: DashboardRequest):
    return await dashboard_data(request)

# ---------------------------------------------------------------------------
# STARTUP EVENT
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    if init_supabase():
        logger.info("Database initialized successfully")
        # Start background task to process orders periodically
        asyncio.create_task(periodic_order_processing())
    else:
        logger.warning("Failed to initialize database connection")

async def periodic_order_processing():
    """Background task to process orders every 5 minutes"""
    while True:
        try:
            logger.info("Running periodic order processing...")
            result = await auto_process_orders()
            logger.info(f"Periodic processing completed: {result}")
        except Exception as e:
            logger.error(f"Periodic processing error: {str(e)}")
        
        # Wait 5 minutes before next check
        await asyncio.sleep(300)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
