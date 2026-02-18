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
        test_result = supabase.table("order_items").select("*").limit(1).execute()
        logger.info(f"Database connected successfully. Test query returned {len(test_result.data)} rows")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    init_supabase()

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
    if not supabase: raise HTTPException(status_code=500, detail="Database not configured")
    try:
        order_items_result = supabase.table("order_items").select("order_id, item_name, quantity").eq("item_name", item_name).execute()
        if not order_items_result.data: return pd.DataFrame()
        
        order_ids = [item["order_id"] for item in order_items_result.data]
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

# ---------------------------------------------------------------------------
# AI FORECAST ENGINE
# ---------------------------------------------------------------------------
def generate_world_class_forecast(item_name: str, days_ahead: int) -> Optional[pd.DataFrame]:
    if not supabase: raise HTTPException(status_code=500, detail="Database not configured")
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
    return {"message": "KOTAai API", "status": "online", "version": "2.1.0"}

@app.get("/health")
async def health_check():
    health_status = {"status": "healthy", "timestamp": datetime.now().isoformat()}
    if supabase:
        try:
            # Check for ingredient_stock table
            supabase.table("ingredient_stock").select("*").limit(1).execute()
            health_status["database"] = "connected"
            health_status["stock_table"] = "ingredient_stock"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
    return health_status

@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    """Main dashboard endpoint - updated to use ingredient_stock table"""
    try:
        items_data = []
        total_recommended = 0
        critical_count = 0
        
        fallback_demands = {
            "Russian ": 12.0, "Bread": 25.0, "Lettuce": 8.0, "Cheese": 10.0, 
            "Steak": 6.0, "Atchar": 5.0, "Vienna": 15.0, "Tomatoes": 10.0, "Burger": 12.0
        }
        
        for item in request.items:
            try:
                # 1. FETCH FROM ingredient_stock INSTEAD OF stock
                current_stock = item.current_stock
                if current_stock is None and supabase:
                    try:
                        # Querying the specified ingredient_stock table
                        stock_result = supabase.table("ingredient_stock").select("current_stock").eq(
                            "item_name", item.item_name.strip()
                        ).execute()
                        
                        if stock_result.data:
                            current_stock = stock_result.data[0]["current_stock"]
                        else:
                            current_stock = 0
                    except Exception as e:
                        logger.warning(f"Error reading ingredient_stock for {item.item_name}: {str(e)}")
                        current_stock = 0
                
                # 2. GENERATE FORECAST
                forecast_df = generate_world_class_forecast(item.item_name, 7)
                weekly_demand = forecast_df["final_prediction"].sum() if forecast_df is not None else fallback_demands.get(item.item_name.strip(), 5.0)
                
                # 3. CALCULATE METRICS
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
                    "weekly_demand": round(weekly_demand, 1),
                    "days_left": round(days_left, 1),
                    "recommended_order": round(recommended_order, 1),
                    "urgency": urgency,
                    "status": status,
                    "action": "REORDER NOW" if urgency == "HIGH" else "Monitor stock"
                })
                
            except Exception as e:
                logger.error(f"Error processing {item.item_name}: {str(e)}")
                items_data.append({"item_name": item.item_name, "current_stock": 0, "weekly_demand": 0, "days_left": 0, "recommended_order": 0, "urgency": "HIGH", "status": "ERROR", "action": "Check data"})
        
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

@app.post("/reorder-recommendations") 
async def reorder_recommendations(request: DashboardRequest):
    return await dashboard_data(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
