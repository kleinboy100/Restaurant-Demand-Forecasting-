# main.py
import os
import logging
from pythonjsonlogger import jsonlogger
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from prophet import Prophet

# ---------------------------------------------------------------------------
# 0. ----------------------------------------------------------------------
#    CRITICAL: SET UP LOGGER FIRST (BEFORE ANY TOP-LEVEL CODE)
# ---------------------------------------------------------------------------
logger = logging.getLogger("kota-ai")
logger.setLevel(logging.INFO)

# JSON logging – works perfectly with Render's log stream
handler = logging.StreamHandler()
handler.setFormatter(
    jsonlogger.JsonFormatter(
        "%(asctime)s | %(levelname)-8s | %(message)s | %(module)s:%(lineno)d",
        rename_fields={"asctime": "time"}
    )
)
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# 1. LOAD .env (MUST BE AFTER LOGGER SETUP)
# ---------------------------------------------------------------------------
load_dotenv()  # Loads VITE_SUPABASE_URL & VITE_SUPABASE_ANON_KEY from .env
logger.info("✅ .env loaded")

# ---------------------------------------------------------------------------
# 2. SUPABASE CONFIG (VITE_ PREFIXES — MATCHES YOUR .env)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("VITE_SUPABASE_ANON_KEY")

# ---------------------------------------------------------------------------
# 3. SUPABASE CLIENT INITIALIZATION (SAFE — LOGGER EXISTS NOW)
# ---------------------------------------------------------------------------
def get_supabase_client() -> Optional[Client]:
    """
    Creates Supabase client. Logs missing vars *before* trying to connect.
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        missing = []
        if not SUPABASE_URL:
            missing.append("VITE_SUPABASE_URL")
        if not SUPABASE_ANON_KEY:
            missing.append("VITE_SUPABASE_ANON_KEY")
        logger.warning(f"Supabase credentials missing: {', '.join(missing)}")
        return None

    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        # Test connection
        test = client.table("order_items").select("*", count="exact").limit(1).execute()
        count = len(test.data) if test.data else 0
        logger.info(f"✅ Supabase connected. Found {count} order items")
        return client
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {str(e)}")
        return None

# Initialize client AT MODULE LEVEL (now safe — logger exists)
supabase = get_supabase_client()

# ---------------------------------------------------------------------------
# 4. CONSTANTS & DATA MODELS
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
# 5. WEATHER SERVICE
# ---------------------------------------------------------------------------
def get_klerksdorp_weather(days: int = 7) -> List[float]:
    """Returns weather impact multipliers [0.4-1.0] based on precipitation."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={KLERKSDORP_LAT}&longitude={KLERKSDORP_LON}"
        f"&daily=precipitation_probability&forecast_days={days}"
    )
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            probs = response.json().get("daily", {}).get("precipitation_probability", [])
            return [
                1.0 if p < 25 else 0.7 if p < 60 else 0.4
                for p in probs
            ]
    except Exception as e:
        logger.error(f"Weather API error: {str(e)}")

    # Fallback = no impact
    return [0.9] * days

# ---------------------------------------------------------------------------
# 6. STOCK MANAGEMENT (SAFE GUARDS AGAINST MISSING SUPABASE)
# ---------------------------------------------------------------------------
def get_current_stock_from_table(item_name: str) -> Optional[int]:
    if not supabase:
        logger.warning("get_current_stock_from_table: Supabase not ready")
        return None

    try:
        res = supabase.table("stock").select("current_stock") \
            .eq("item_name", item_name).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]["current_stock"]
    except Exception as e:
        logger.error(f"Error fetching stock for {item_name}: {str(e)}")
    return None


def update_stock_in_table(item_name: str, quantity_change: int,
                          transaction_type: str, notes: str = "") -> bool:
    if not supabase:
        logger.warning("update_stock_in_table: Supabase not ready")
        return False

    try:
        current = get_current_stock_from_table(item_name)
        if current is None:
            return False

        new_stock = max(0, current + quantity_change)

        # Update stock record
        supabase.table("stock").update({
            "current_stock": new_stock,
            "last_updated": datetime.now().isoformat()
        }).eq("item_name", item_name).execute()

        # Log transaction
        supabase.table("stock_transactions").insert({
            "item_name": item_name,
            "transaction_type": transaction_type,
            "quantity": quantity_change,
            "previous_stock": current,
            "new_stock": new_stock,
            "notes": notes
        }).execute()

        logger.info(f"Stock updated: {item_name} +{quantity_change} → {new_stock}")
        return True
    except Exception as e:
        logger.error(f"Stock update failed for {item_name}: {str(e)}")
        return False

# ---------------------------------------------------------------------------
# 7. SALES DATA EXTRACTION
# ---------------------------------------------------------------------------
def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    if not supabase:
        logger.error("get_sales_from_order_items: Supabase not ready")
        raise HTTPException(status_code=500, detail="Database not configured")

    try:
        # Fetch all order items for this product
        res = supabase.table("order_items").select("*") \
            .eq("item_name", item_name).execute()

        if not res.data:
            return pd.DataFrame()

        df = pd.DataFrame(res.data)

        # Link to orders table for dates
        order_ids = df["order_id"].unique().tolist()
        orders_res = supabase.table("orders").select("id, created_at, order_date") \
            .in_("id", order_ids).execute()

        if orders_res.data:
            orders_df = pd.DataFrame(orders_res.data)
            date_col = "order_date" if "order_date" in orders_df.columns else "created_at"
            df = df.merge(orders_df[["id", date_col]], on="order_id", how="left")
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
# 8. AI FORECAST ENGINE
# ---------------------------------------------------------------------------
def generate_world_class_forecast(item_name: str, days_ahead: int) -> Optional[pd.DataFrame]:
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    df = get_sales_from_order_items(item_name)

    if df.empty or len(df) < 5:
        logger.warning(f"Insufficient data for {item_name} (needs ≥5 rows)")
        return None

    # Add event impacts if table exists
    df["impact_score"] = 1.0
    try:
        event_res = supabase.table("events").select("event_date, impact_score").execute()
        if event_res.data:
            events_df = pd.DataFrame(event_res.data)
            if not events_df.empty:
                events_df["event_date"] = pd.to_datetime(events_df["event_date"])
                df = pd.merge(df, events_df, left_on="ds", right_on="event_date", how="left")
                df["impact_score"] = df["impact_score"].fillna(1.0)
    except Exception:
        pass

    # Build Prophet model
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name="ZA")

    if "impact_score" in df and df["impact_score"].nunique() > 1:
        model.add_regressor("impact_score")

    model.fit(df)

    # Forecast future
    future = model.make_future_dataframe(periods=days_ahead)
    future["impact_score"] = future["ds"].apply(lambda x: 1.5 if x.day >= 25 else 1.0)

    forecast = model.predict(future)
    weather_mult = get_klerksdorp_weather(days_ahead)

    results = forecast.tail(days_ahead).copy()
    results["final_prediction"] = np.array(weather_mult) * results["yhat"]
    results["final_prediction"] = results["final_prediction"].clip(lower=0)

    return results[["ds", "final_prediction", "yhat_lower", "yhat_upper"]]

# ---------------------------------------------------------------------------
# 9. FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI(title="Kota AI: Klerksdorp Edition")

# ---------------------------------------------------------------------------
# 10. ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Kota AI Forecasting API", "status": "online"}

@app.get("/health")
async def health():
    db_status = "not configured"
    item_count = 0
    stock_table_exists = False

    if supabase:
        try:
            test = supabase.table("order_items").select("*", count="exact").limit(1).execute()
            db_status = "connected"
            item_count = len(test.data) if test.data else 0

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
        "recommendation": "Use current_stock param" if not stock_table_exists else "Stock table ready"
    }

@app.post("/api/forecast")
async def get_forecast(request: ForecastRequest):
    try:
        data = generate_world_class_forecast(request.item_name, request.days_ahead)
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    if data is None:
        # Fallback to simple average
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

    # Get current stock
    current_stock = request.current_stock
    if current_stock is None:
        current_stock = get_current_stock_from_table(request.item_name)
        if current_stock is None:
            raise HTTPException(
                status_code=400,
                detail="Must provide 'current_stock' (stock table missing or item not found)."
            )

    # Get forecast
    forecast_data = generate_world_class_forecast(request.item_name, 7)
    if forecast_data is None:
        sales = get_sales_from_order_items(request.item_name)
        weekly_need = sales["y"].mean() * 7 if not sales.empty else 70
    else:
        weekly_need = forecast_data["final_prediction"].sum()

    # Calculate metrics
    days_left = current_stock / (weekly_need / 7) if weekly_need > 0 else 999
    recommended = max(0, (weekly_need * 1.5) - current_stock)
    urgency = "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW"

    return {
        "item": request.item_name,
        "current_stock": current_stock,
        "predicted_weekly_demand": round(weekly_need, 1),
        "days_of_stock_left": round(days_left, 1),
        "recommended_order": round(recommended, 1),
        "urgency": urgency,
        "reorder_now": urgency == "HIGH",
        "estimated_restock_days": round(recommended / (weekly_need / 7), 1) if weekly_need else 7
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

        stock = item_data.get("current_stock") or get_current_stock_from_table(item) or 0
        forecast = generate_world_class_forecast(item, 7)

        if forecast is None:
            sales = get_sales_from_order_items(item)
            weekly = sales["y"].mean() * 7 if not sales.empty else 70
        else:
            weekly = forecast["final_prediction"].sum()

        days = stock / (weekly / 7) if weekly > 0 else 999
        rec = max(0, (weekly * 1.5) - stock)
        urgency = "HIGH" if days < 3 else "MEDIUM" if days < 7 else "LOW"

        results.append({
            "item_name": item,
            "current_stock": stock,
            "weekly_demand": round(weekly, 1),
            "days_left": round(days, 1),
            "recommended_order": round(rec, 1),
            "urgency": urgency,
            "status": "CRITICAL" if days < 2 else "OK" if days > 14 else "LOW"
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
# 11. STARTUP CHECK (LOGS STATUS TO RENDER)
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Kota AI API Starting...")
    if supabase:
        try:
            count = len(supabase.table("order_items").select("*", count="exact").limit(5).execute().data)
            logger.info(f"✅ Found {count} order items")

            try:
                stock_test = supabase.table("stock").select("*", count="exact").limit(1).execute()
                logger.info("✅ Stock table exists")
            except:
                logger.warning("⚠️ Stock table missing — using manual stock input")
        except Exception as e:
            logger.error(f"⚠️ Database issue: {str(e)}")
    else:
        logger.error("⚠️⚠️⚠️ Supabase not configured — core features disabled")

# ---------------------------------------------------------------------------
# 12. RUN SERVER (for local dev)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
