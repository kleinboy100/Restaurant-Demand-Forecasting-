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
# MEAL-INGREDIENT MAPPING (for future use)
# ---------------------------------------------------------------------------
MEAL_INGREDIENTS = {
    "Flamwood": {
        "Chips": 150,  # grams
        "Melted Cheese": 2,  # portions
        "Russian": 1,  # portion
        "lettuce": 50,  # grams
        "Bread": 0.25,  # quarter loaf
        "tomato": 1  # slice
    },
    "Stop 5": {
        "Chips": 150,
        "Melted Cheese": 1,
        "Russian": 1,
        "lettuce": 50,
        "Bread": 0.25,
        "tomato": 1,
        "atchar": 50,
        "Vienna": 1,
        "egg": 1
    },
    "Stop 18": {
        "Chips": 150,
        "Melted Cheese": 1,
        "lettuce": 50,
        "Bread": 0.25,
        "tomato": 1,
        "atchar": 50
    },
    "Phelandaba": {
        "Chips": 150,
        "Melted Cheese": 1,
        "Russian": 1,
        "lettuce": 50,
        "Bread": 0.25,
        "tomato": 1,
        "atchar": 50,
        "egg": 1
    },
    "Steak": {
        "steak": 100  # grams
    },
    "Toast": {
        "Bread": 4,  # slices
        "Chips": 150  # grams
    }
}

# ---------------------------------------------------------------------------
# PYDANTIC MODELS (keeping exact same names and structure)
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

# ---------------------------------------------------------------------------
# DATABASE CONNECTION (exact same function name)
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
        
        # Test connection
        test_result = supabase.table("order_items").select("*").limit(1).execute()
        logger.info(f"Database connected successfully. Test query returned {len(test_result.data)} rows")
        return True
        
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {str(e)}")
        return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    init_supabase()

# ---------------------------------------------------------------------------
# WEATHER DATA INTEGRATION (exact same function name)
# ---------------------------------------------------------------------------
def get_klerksdorp_weather(days_ahead: int = 7) -> Dict[str, float]:
    """Get weather forecast for Klerksdorp from Open-Meteo API"""
    try:
        # Klerksdorp coordinates
        lat, lon = -26.85, 26.66
        
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_probability_max",
            "forecast_days": days_ahead,
            "timezone": "Africa/Johannesburg"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        weather_impact = {}
        
        dates = data["daily"]["time"]
        precip_prob = data["daily"]["precipitation_probability_max"]
        
        for i, date_str in enumerate(dates):
            prob = precip_prob[i] if i < len(precip_prob) else 0
            
            # Convert precipitation probability to demand impact
            if prob < 20:  # Low chance of rain
                impact = 1.0  # Normal demand
            elif prob < 60:  # Medium chance of rain
                impact = 0.85  # Slightly lower demand
            else:  # High chance of rain
                impact = 0.7  # Lower demand (people stay home)
                
            weather_impact[date_str] = impact
            
        return weather_impact
        
    except Exception as e:
        logger.warning(f"Weather API failed: {str(e)}, using default impact")
        # Default impact for all future dates
        future_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") 
                       for i in range(1, days_ahead + 1)]
        return {date_str: 1.0 for date_str in future_dates}

# ---------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS (exact same function name)
# ---------------------------------------------------------------------------
def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    """Fetch historical sales data for an item"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        # Get order_items data
        order_items_result = supabase.table("order_items").select(
            "order_id, item_name, quantity"
        ).eq("item_name", item_name).execute()
        
        if not order_items_result.data:
            logger.warning(f"No sales data found for {item_name}")
            return pd.DataFrame()
        
        # Get order dates
        order_ids = [item["order_id"] for item in order_items_result.data]
        orders_result = supabase.table("orders").select(
            "id, created_at"
        ).in_("id", order_ids).execute()
        
        if not orders_result.data:
            return pd.DataFrame()
        
        # Create dataframes
        order_items_df = pd.DataFrame(order_items_result.data)
        orders_df = pd.DataFrame(orders_result.data)
        orders_df.rename(columns={"id": "order_id"}, inplace=True)
        
        # Merge data
        merged_df = pd.merge(order_items_df, orders_df, on="order_id", how="inner")
        merged_df["created_at"] = pd.to_datetime(merged_df["created_at"])
        
        # Filter recent data
        cutoff_date = datetime.now() - timedelta(days=days_back)
        merged_df = merged_df[merged_df["created_at"] >= cutoff_date]
        
        if merged_df.empty:
            return pd.DataFrame()
        
        # Aggregate by date
        merged_df["sale_date"] = merged_df["created_at"].dt.date
        daily_sales = merged_df.groupby("sale_date")["quantity"].sum().reset_index()
        
        # Prepare for Prophet
        daily_sales.rename(columns={"sale_date": "ds", "quantity": "y"}, inplace=True)
        daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])
        
        logger.info(f"Found {len(daily_sales)} days of sales data for {item_name}")
        return daily_sales
        
    except Exception as e:
        logger.error(f"Error fetching sales data for {item_name}: {str(e)}")
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# AI FORECAST ENGINE (exact same function name)
# ---------------------------------------------------------------------------
def generate_world_class_forecast(item_name: str, days_ahead: int) -> Optional[pd.DataFrame]:
    """Generate AI-powered forecast using Prophet with weather integration"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    df = get_sales_from_order_items(item_name)

    if df.empty:  # Changed: Now only requires 1+ days of data
        logger.warning(f"No sales data available for {item_name}")
        return None

    # Add event impact (optional)
    df["impact_score"] = 1.0
    try:
        event_res = supabase.table("events").select("event_date, impact_score").execute()
        if event_res.data:
            events_df = pd.DataFrame(event_res.data)
            events_df["event_date"] = pd.to_datetime(events_df["event_date"])
            df = pd.merge(df, events_df, left_on="ds", right_on="event_date", how="left")
            df["impact_score"] = df["impact_score"].fillna(1.0)
    except Exception as e:
        logger.warning(f"Could not load events: {str(e)}")

    # Build Prophet model
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name="ZA")
    if "impact_score" in df and df["impact_score"].nunique() > 1:
        model.add_regressor("impact_score")

    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=days_ahead, freq='D')
    
    # Add weather impact to future dates
    weather_impact = get_klerksdorp_weather(days_ahead)
    future["impact_score"] = 1.0
    
    for i, row in future.iterrows():
        date_str = row["ds"].strftime("%Y-%m-%d")
        if date_str in weather_impact:
            future.at[i, "impact_score"] = weather_impact[date_str]

    # Generate predictions
    forecast = model.predict(future)
    
    # Apply final adjustments
    forecast["final_prediction"] = forecast["yhat"] * future["impact_score"]
    forecast["final_prediction"] = forecast["final_prediction"].clip(lower=0)
    
    # Return only future predictions
    future_forecast = forecast[forecast["ds"] > df["ds"].max()].copy()
    
    logger.info(f"Generated {len(future_forecast)} days forecast for {item_name}")
    return future_forecast[["ds", "final_prediction", "yhat_lower", "yhat_upper"]]

# ---------------------------------------------------------------------------
# API ENDPOINTS (all exact same endpoint names and function names)
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "KOTAai Restaurant Demand Forecasting API",
        "status": "online",
        "version": "2.1.0",
        "location": "Klerksdorp",
        "meals_available": ["Flamwood", "Stop 5", "Stop 18", "Phelandaba", "Steak", "Toast"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "location": "Klerksdorp",
        "timestamp": datetime.now().isoformat(),
        "dashboard_url": "https://kleinboy100.github.io/Dashboard/",
        "cors_enabled": True
    }
    
    if supabase:
        try:
            # Test database connection
            result = supabase.table("order_items").select("*").limit(1).execute()
            health_status["database"] = "connected"
            health_status["order_items_count"] = len(result.data)
            
            # Check if stock table exists
            try:
                stock_result = supabase.table("stock").select("*").limit(1).execute()
                health_status["stock_table_exists"] = True
                health_status["recommendation"] = "Stock table ready"
            except:
                health_status["stock_table_exists"] = False
                health_status["recommendation"] = "Use current_stock parameter in requests"
            
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
            health_status["recommendation"] = "Check database connection"
    else:
        health_status["database"] = "not configured"
        health_status["recommendation"] = "Configure SUPABASE_URL and SUPABASE_ANON_KEY"
    
    return health_status

@app.get("/api/order_items")
async def get_order_items_stats():
    """Get order items statistics"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        # Get all order items
        result = supabase.table("order_items").select("item_name, quantity").execute()
        
        if not result.data:
            return {"message": "No order items found", "items": []}
        
        # Calculate statistics
        df = pd.DataFrame(result.data)
        stats = df.groupby("item_name")["quantity"].agg(['count', 'sum', 'mean']).round(2)
        stats_dict = stats.to_dict('index')
        
        # Format response
        items_stats = []
        for item_name, stats in stats_dict.items():
            items_stats.append({
                "item_name": item_name,
                "total_orders": int(stats['count']),
                "total_quantity": int(stats['sum']),
                "average_per_order": float(stats['mean'])
            })
        
        # Sort by total quantity
        items_stats.sort(key=lambda x: x['total_quantity'], reverse=True)
        
        return {
            "total_items": len(items_stats),
            "total_orders": len(result.data),
            "items": items_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting order items stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get order items: {str(e)}")

@app.post("/api/forecast")
async def forecast_demand(request: ForecastRequest):
    """Generate demand forecast for a specific meal"""
    try:
        forecast_df = generate_world_class_forecast(request.item_name, request.days_ahead)
        
        if forecast_df is None or forecast_df.empty:
            # Fallback estimate based on item name
            fallback_estimates = {
                "Flamwood": 8.0,
                "Stop 5": 6.0,
                "Stop 18": 5.0,
                "Phelandaba": 7.0,
                "Steak": 4.0,
                "Toast": 3.0
            }
            weekly_total = fallback_estimates.get(request.item_name, 5.0) * request.days_ahead
            
            return {
                "item": request.item_name,
                "status": "insufficient_data",
                "message": "Using fallback estimate - need more sales records for AI forecast.",
                "days_ahead": request.days_ahead,
                "weekly_total": round(weekly_total, 1),
                "recommendation": "Add more sales data for better predictions."
            }
        
        # Format response
        forecast_list = []
        weekly_total = 0
        
        for _, row in forecast_df.iterrows():
            daily_prediction = max(0, round(row["final_prediction"], 1))
            weekly_total += daily_prediction
            
            forecast_list.append({
                "date": row["ds"].strftime("%Y-%m-%d"),
                "predicted": daily_prediction,
                "low_estimate": max(0, round(row["yhat_lower"], 1)),
                "high_estimate": max(0, round(row["yhat_upper"], 1))
            })
        
        return {
            "item": request.item_name,
            "days_ahead": request.days_ahead,
            "weekly_total": round(weekly_total, 1),
            "forecast": forecast_list,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Forecast error for {request.item_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.post("/api/recommend")
async def get_recommendation(request: RecommendationRequest):
    """Get reorder recommendation for a specific meal"""
    try:
        # Get current stock
        current_stock = request.current_stock
        if current_stock is None and supabase:
            try:
                stock_result = supabase.table("stock").select("current_stock").eq(
                    "item_name", request.item_name
                ).execute()
                if stock_result.data:
                    current_stock = stock_result.data[0]["current_stock"]
                else:
                    current_stock = 0
            except Exception as e:
                logger.warning(f"Could not fetch stock for {request.item_name}: {str(e)}")
                current_stock = 0
        elif current_stock is None:
            current_stock = 0
        
        # Generate forecast
        forecast_df = generate_world_class_forecast(request.item_name, 7)
        
        if forecast_df is None or forecast_df.empty:
            # Fallback calculation based on meal type
            fallback_demands = {
                "Flamwood": 8.0,
                "Stop 5": 6.0,
                "Stop 18": 5.0,
                "Phelandaba": 7.0,
                "Steak": 4.0,
                "Toast": 3.0
            }
            weekly_demand = fallback_demands.get(request.item_name, 5.0) * 7
        else:
            weekly_demand = forecast_df["final_prediction"].sum()
        
        # Calculate recommendations
        daily_demand = weekly_demand / 7
        days_left = current_stock / daily_demand if daily_demand > 0 else float('inf')
        recommended_order = max(0, (weekly_demand * 1.5) - current_stock)
        
        # Determine urgency
        if days_left < 3:
            urgency = "HIGH"
        elif days_left < 7:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        return {
            "item": request.item_name,
            "current_stock": current_stock,
            "predicted_weekly_demand": round(weekly_demand, 1),
            "days_of_stock_left": round(days_left, 1),
            "recommended_order": round(recommended_order, 1),
            "urgency": urgency,
            "reorder_now": urgency == "HIGH",
            "estimated_restock_days": 7
        }
        
    except Exception as e:
        logger.error(f"Recommendation error for {request.item_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    """Main dashboard endpoint - get recommendations for multiple meals"""
    try:
        items_data = []
        total_recommended = 0
        critical_count = 0
        
        # Fallback demands for meals (if no sales data)
        fallback_demands = {
            "Flamwood": 8.0,
            "Stop 5": 6.0,
            "Stop 18": 5.0,
            "Phelandaba": 7.0,
            "Steak": 4.0,
            "Toast": 3.0,
            # Legacy support for old ingredient names
            "Bread": 10.0,
            "Cheese": 15.0,
            "Polony": 8.0,
            "Atchar": 5.0,
            "Onions": 12.0,
            "Tomatoes": 10.0,
            "Chips": 20.0,
            "Sausage": 8.0,
            "Chicken": 12.0,
            "Beef": 10.0,
            "Potatoes": 1
