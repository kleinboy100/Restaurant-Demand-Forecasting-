# main.py
# TEMPORARY DEBUG LOGS
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from prophet import Prophet
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# 0. LOAD ENVIRONMENT VARIABLES (FOR LOCAL DEVELOPMENT)
# ---------------------------------------------------------------------------
load_dotenv(override=False) 

# ---------------------------------------------------------------------------
# 1. SET UP STRUCTURED JSON LOGGING — NATIVE PYTHON 3.14+ COMPATIBLE
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "time": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "lineno": record.lineno
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)

logger = logging.getLogger("kota-ai")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# 3. NOW RUN THE DEBUG LOGS (logger is now defined)
logger.info(f"DEBUG: SUPABASE_URL in OS? {'SUPABASE_URL' in os.environ}")
logger.info(f"DEBUG: SUPABASE_ANON_KEY in OS? {'SUPABASE_ANON_KEY' in os.environ}")  # Loads .env for local dev — backend ignores VITE_ prefixes

# ---------------------------------------------------------------------------
# 2. SUPABASE CONFIG — MATCH RENDER ENVIRONMENT (NOT VITE_)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = (
    os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    or os.getenv("SUPABASE_PUBLISHABLE_KEY")
)

# Debug log: Confirm environment variables are loaded
logger.info(f"SUPABASE_URL set? {'✅ Yes' if SUPABASE_URL else '❌ No'}")
logger.info(f"SUPABASE_ANON_KEY set? {'✅ Yes' if SUPABASE_ANON_KEY else '❌ No'}")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    logger.error("⚠️⚠️ Supabase credentials are missing! Backend will not function.")
else:
    logger.info("✅ Supabase environment configured correctly (matches Render)")

# ---------------------------------------------------------------------------
# 3. SUPABASE CLIENT INITIALIZATION
# ---------------------------------------------------------------------------
def get_supabase_client() -> Optional[Client]:
    """
    Creates and returns a Supabase client if credentials are present.
    Logs errors or missing env vars.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None

    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        # Change these lines in get_supabase_client() and /health
        test = client.table("order_items").select("*", count="exact").limit(1).execute()
        # test.count is the special Supabase property for the TRUE total
        count = test.count if test.count is not None else 0

        logger.info(f"✅ Supabase connected. Total orders in DB: {count}")
        return client
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {str(e)}")
        return None

# Initialize globally — safe because logger and env vars are loaded
supabase = get_supabase_client()

# ---------------------------------------------------------------------------
# 4. CONSTANTS
# ---------------------------------------------------------------------------
KLERKSDORP_LAT = -26.86
KLERKSDORP_LON = 26.63

class ForecastRequest(BaseModel):
    item_name: str
    days_ahead: int = 7

class RecommendationRequest(BaseModel):
    item_name: str
    current_stock: Optional[int] = None

class StockUpdateRequest(BaseModel):
    item_name: str
    quantity: int
    transaction_type: str = "ADJUSTMENT"
    notes: Optional[str] = None

class DashboardRequest(BaseModel):
    items: List[Dict[str, Any]]

# ---------------------------------------------------------------------------
# 5. WEATHER SERVICE (GETS HISTORICAL IMPACT FROM OPEN METEO)
# ---------------------------------------------------------------------------
def get_klerksdorp_weather(days: int = 7) -> List[float]:
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={KLERKSDORP_LAT}&longitude={KLERKSDORP_LON}"
        f"&daily=precipitation_probability&forecast_days={days}"
    )
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            probs = response.json().get("daily", {}).get("precipitation_probability", [])
            return [1.0 if p < 25 else 0.7 if p < 60 else 0.4 for p in probs]
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")
    return [0.9] * days  # Fallback: neutral impact

# ---------------------------------------------------------------------------
# 6. STOCK MANAGEMENT (SAFE HANDLING FOR MISSING SUPABASE)
# ---------------------------------------------------------------------------
def get_current_stock_from_table(item_name: str) -> Optional[int]:
    if not supabase:
        logger.warning("get_current_stock_from_table: Supabase not available")
        return None

    try:
        res = supabase.table("stock").select("current_stock").eq("item_name", item_name).execute()
        if res.data:
            return res.data[0]["current_stock"]
    except Exception as e:
        logger.error(f"Error fetching stock for {item_name}: {str(e)}")
    return None

def update_stock_in_table(item_name: str, quantity_change: int, transaction_type: str, notes: str = "") -> bool:
    if not supabase:
        logger.warning("update_stock_in_table: Supabase not available")
        return False

    try:
        current = get_current_stock_from_table(item_name)
        if current is None:
            return False

        new_stock = max(0, current + quantity_change)

        supabase.table("stock").update({
            "current_stock": new_stock,
            "last_updated": datetime.now().isoformat()
        }).eq("item_name", item_name).execute()

        supabase.table("stock_transactions").insert({
            "item_name": item_name,
            "transaction_type": transaction_type,
            "quantity": quantity_change,
            "previous_stock": current,
            "new_stock": new_stock,
            "notes": notes,
            "created_at": datetime.now().isoformat()
        }).execute()

        logger.info(f"Stock updated: {item_name} +{quantity_change} → {new_stock}")
        return True
    except Exception as e:
        logger.error(f"Stock update failed for {item_name}: {str(e)}")
        return False

# ---------------------------------------------------------------------------
# 7. SALES DATA EXTRACTION (USE SUPABASE)
# ---------------------------------------------------------------------------
def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    if not supabase:
        logger.error("get_sales_from_order_items: Supabase not available")
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        res = supabase.table("order_items").select("*").eq("item_name", item_name).execute()
        if not res.data:
            return pd.DataFrame()

        df = pd.DataFrame(res.data)

        # Link to orders table for timestamps
        order_ids = df["order_id"].unique().tolist()
        if not order_ids:
            return pd.DataFrame()

        orders_res = supabase.table("orders").select("id, created_at, order_date").in_("id", order_ids).execute()
        if orders_res.data:
            orders_df = pd.DataFrame(orders_res.data)
            date_col = "order_date" if "order_date" in orders_df.columns else "created_at"
            df = df.merge(orders_df[["id", date_col]], left_on="order_id", right_on="id", how="left")
            df["sale_date"] = pd.to_datetime(df[date_col])
        else:
            df["sale_date"] = pd.Timestamp.now()

        # Filter last N days
        cutoff = pd.Timestamp.now() - timedelta(days=days_back)
        df = df[df["sale_date"] >= cutoff]

        if not df.empty and "sale_date" in df:
            grouped = df.groupby("sale_date")["quantity"].sum().reset_index()
            grouped.columns = ["ds", "y"]
            return grouped

        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching sales for {item_name}: {str(e)}")
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# 8. AI FORECAST ENGINE (Prophet + Weather Impact)
# ---------------------------------------------------------------------------
def generate_world_class_forecast(item_name: str, days_ahead: int) -> Optional[pd.DataFrame]:
    df = get_sales_from_order_items(item_name)

    if df.empty or len(df) < 1:  # Changed to 1
        # if fewer than 1 daily point, abort
        return None

    # ... (rest of the function remains the same)
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

    # Forecast next N days
    future = model.make_future_dataframe(periods=days_ahead)
    future["impact_score"] = 1.0  # default

    # Optionally: Add weather impact
    weather_mult = get_klerksdorp_weather(days_ahead)
    future_tail = future.tail(days_ahead)
    future_tail = future_tail.copy()
    future_tail["impact_score"] = [(weather_mult[i] if i < len(weather_mult) else 0.9) for i in range(len(future_tail))]

    # Merge back impact scores
    future.update(future_tail)

    forecast = model.predict(future)
    results = forecast.tail(days_ahead).copy()
    results["final_prediction"] = results["yhat"] * results["impact_score"].clip(lower=0.4)
    results["final_prediction"] = results["final_prediction"].clip(lower=0)

    return results[["ds", "final_prediction", "yhat_lower", "yhat_upper"]]

# ---------------------------------------------------------------------------
# 9. FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kota AI: Klerksdorp Edition",
    description="AI-powered inventory forecasting for restaurant ingredients",
    version="1.0.0"
)

# ---------------------------------------------------------------------------
# 10. CORS CONFIGURATION - UPDATED WITH YOUR GITHUB PAGES URL
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",  # For your local dashboard testing
        "https://kleinboy100.github.io",  # Your GitHub Pages domain
        "https://restaurant-demand-forecasting-1.onrender.com",  # API self-origin
        "*"  # Allow all origins for development (remove in production)
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# ---------------------------------------------------------------------------
# 11. ROUTES
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Kota AI Forecasting API - Klerksdorp Edition", 
        "status": "online",
        "version": "1.0.0",
        "dashboard_url": "https://kleinboy100.github.io/Dashboard/"
    }

@app.get("/health")
async def health():
    db_status = "not configured"
    item_count = 0
    stock_table_exists = False

    if supabase:
        try:
            test = supabase.table("order_items").select("*", count="exact").limit(1).execute()
            db_status = "connected"
            item_count = test.count if test.count is not None else 0

            stock_res = supabase.table("stock").select("*", count="exact").limit(1).execute()
            stock_table_exists = bool(stock_res.data)
        except Exception as e:
            db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "location": "Klerksdorp",
        "database": db_status,
        "order_items_count": item_count,
        "stock_table_exists": stock_table_exists,
        "dashboard_url": "https://kleinboy100.github.io/Dashboard/",
        "cors_enabled": True,
        "recommendation": "Use current_stock param" if not stock_table_exists else "Stock table ready"
    }

@app.post("/api/forecast")
async def get_forecast(request: ForecastRequest):
    try:
        data = generate_world_class_forecast(request.item_name, request.days_ahead)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    if data is None:
        sales = get_sales_from_order_items(request.item_name)
        avg_daily = sales["y"].mean() if not sales.empty else 10
        weekly = avg_daily * 7

        return {
            "item": request.item_name,
            "status": "insufficient_data",
            "message": "Need ≥5 sales records for AI forecast.",
            "total_sold_to_date": round(avg_daily * 90, 1),
            "recommendation": "Use manual estimate or add sales data."
        }

    forecast = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "predicted": round(row["final_prediction"], 1),
            "low_estimate": round(row["yhat_lower"], 1),
            "high_estimate": round(row["yhat_upper"], 1)
        }
        for _, row in data.iterrows()
    ]

    return {
        "item": request.item_name,
        "days_ahead": request.days_ahead,
        "weekly_total": round(data["final_prediction"].sum(), 1),
        "forecast": forecast
    }

@app.post("/api/recommend")
async def get_recommendation(request: RecommendationRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    current_stock = request.current_stock
    if current_stock is None:
        current_stock = get_current_stock_from_table(request.item_name)
        if current_stock is None:
            raise HTTPException(
                status_code=400,
                detail="Must provide 'current_stock' (product not in stock table or DB down)."
            )

    forecast_data = generate_world_class_forecast(request.item_name, 7)
    if forecast_data is None:
        sales = get_sales_from_order_items(request.item_name)
        weekly_need = sales["y"].mean() * 7 if not sales.empty else 70
    else:
        weekly_need = forecast_data["final_prediction"].sum()

    days_left = current_stock / (weekly_need / 7) if weekly_need > 0 else 999
    recommended_order = max(0, (weekly_need * 1.5) - current_stock)
    urgency = "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW"

    return {
        "item": request.item_name,
        "current_stock": current_stock,
        "predicted_weekly_demand": round(weekly_need, 1),
        "days_of_stock_left": round(days_left, 1),
        "recommended_order": round(recommended_order, 1),
        "urgency": urgency,
        "reorder_now": urgency == "HIGH",
        "estimated_restock_days": round(recommended_order / (weekly_need / 7), 1) if weekly_need else 7
    }

@app.post("/api/dashboard")
async def get_dashboard(request: DashboardRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    results = []
    for item_data in request.items:
        item = item_data.get("item_name")
        if not item:
            continue

        current_stock = item_data.get("current_stock") or get_current_stock_from_table(item) or 0
        forecast = generate_world_class_forecast(item, 7)

        if forecast is None:
            sales = get_sales_from_order_items(item)
            weekly_need = sales["y"].mean() * 7 if not sales.empty else 70
        else:
            weekly_need = forecast["final_prediction"].sum()

        days_left = current_stock / (weekly_need / 7) if weekly_need > 0 else 999
        recommended = max(0, (weekly_need * 1.5) - current_stock)
        urgency = "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW"

        results.append({
            "item_name": item,
            "current_stock": current_stock,
            "weekly_demand": round(weekly_need, 1),
            "days_left": round(days_left, 1),
            "recommended_order": round(recommended, 1),
            "urgency": urgency,
            "status": "CRITICAL" if days_left < 2 else "OK" if days_left > 14 else "LOW"
        })

    results.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["urgency"]])
    return {
        "summary": {
            "total_items": len(results),
            "critical_items": sum(1 for r in results if r["urgency"] == "HIGH"),
            "total_recommended": round(sum(r["recommended_order"] for r in results), 1),
            "timestamp": datetime.now().isoformat()
        },
        "items": results
    }

# ---------------------------------------------------------------------------
# 12. STARTUP EVENT — LOGS API STATUS ON DEPLOY
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Kota AI API Starting...")
    logger.info(f"📊 Dashboard URL: https://kleinboy100.github.io/Dashboard/")
    if supabase:
        try:
            test = supabase.table("order_items").select("*", count="exact").limit(5).execute()
            logger.info(f"✅ Found {test.count if test.count else 0} order items")
            stock_test = supabase.table("stock").select("*", count="exact").limit(1).execute()
            logger.info(f"✅ Stock table exists: {bool(stock_test.data)}")
        except Exception as e:
            logger.error(f"⚠️ Database query failed: {str(e)}")
    else:
        logger.error("⚠️⚠️⚠️ Supabase not configured — backend features disabled")

# ---------------------------------------------------------------------------
# 13. LOCAL DEV SERVER — RUN WITH: python main.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
