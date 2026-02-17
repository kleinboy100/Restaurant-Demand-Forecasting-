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
from fastapi.middleware.cors import CORSMiddleware       # ← ADD THIS IMPORT
from pydantic import BaseModel
from supabase import create_client, Client
from prophet import Prophet

# ---------------------------------------------------------------------------
# 0. LOAD ENVIRONMENT VARIABLES (FOR LOCAL DEVELOPMENT)
# ---------------------------------------------------------------------------
load_dotenv(override=False)

# 2. SET UP LOGGER (Define logger FIRST)
logger = logging.getLogger("kota-ai")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
# … (your existing formatter configuration) …
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# DEBUG ENV VARS
# ---------------------------------------------------------------------------
logger.info(f"DEBUG: SUPABASE_URL in OS? {'SUPABASE_URL' in os.environ}")
logger.info(f"DEBUG: SUPABASE_ANON_KEY in OS? {'SUPABASE_ANON_KEY' in os.environ}")

# ---------------------------------------------------------------------------
# 1. STRUCTURED JSON LOGGING
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

# Re-configure logger to use JSON formatter
logger = logging.getLogger("kota-ai")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# 2. SUPABASE CONFIG — MATCH RENDER ENVIRONMENT (NOT VITE_)
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = (
    os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    or os.getenv("SUPABASE_PUBLISHABLE_KEY")
)

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
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        return None

    try:
        client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        test = client.table("order_items").select("*", count="exact").limit(1).execute()
        count = test.count if test.count is not None else 0
        logger.info(f"✅ Supabase connected. Total orders in DB: {count}")
        return client
    except Exception as e:
        logger.error(f"❌ Supabase connection failed: {e}")
        return None

supabase = get_supabase_client()

# ---------------------------------------------------------------------------
# 4. CONSTANTS & MODELS
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
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={KLERKSDORP_LAT}&longitude={KLERKSDORP_LON}"
        f"&daily=precipitation_probability&forecast_days={days}"
    )
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            probs = resp.json().get("daily", {}).get("precipitation_probability", [])
            return [1.0 if p < 25 else 0.7 if p < 60 else 0.4 for p in probs]
    except Exception as e:
        logger.error(f"Weather API error: {e}")
    return [0.9] * days

# ---------------------------------------------------------------------------
# 6. STOCK MANAGEMENT
# ---------------------------------------------------------------------------
def get_current_stock_from_table(item_name: str) -> Optional[int]:
    if not supabase:
        logger.warning("Supabase not available for stock lookup")
        return None
    try:
        res = supabase.table("stock").select("current_stock").eq("item_name", item_name).execute()
        return res.data[0]["current_stock"] if res.data else None
    except Exception as e:
        logger.error(f"Error fetching stock for {item_name}: {e}")
        return None

def update_stock_in_table(item_name: str, quantity_change: int, transaction_type: str, notes: str = "") -> bool:
    if not supabase:
        logger.warning("Supabase not available for stock update")
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
        logger.error(f"Stock update failed for {item_name}: {e}")
        return False

# ---------------------------------------------------------------------------
# 7. SALES DATA EXTRACTION
# ---------------------------------------------------------------------------
def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    try:
        res = supabase.table("order_items").select("*").eq("item_name", item_name).execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        if df.empty:
            return df
        order_ids = df["order_id"].unique().tolist()
        if order_ids:
            orders_res = supabase.table("orders").select("id, created_at, order_date").in_("id", order_ids).execute()
            orders_df = pd.DataFrame(orders_res.data) if orders_res.data else pd.DataFrame()
            date_col = "order_date" if "order_date" in orders_df.columns else "created_at"
            df = df.merge(orders_df[["id", date_col]], left_on="order_id", right_on="id", how="left")
            df["sale_date"] = pd.to_datetime(df[date_col])
        else:
            df["sale_date"] = pd.Timestamp.now()
        cutoff = pd.Timestamp.now() - timedelta(days=days_back)
        df = df[df["sale_date"] >= cutoff]
        if not df.empty:
            grouped = df.groupby("sale_date")["quantity"].sum().reset_index()
            grouped.columns = ["ds", "y"]
            return grouped
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching sales for {item_name}: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# 8. AI FORECAST ENGINE
# ---------------------------------------------------------------------------
def generate_world_class_forecast(item_name: str, days_ahead: int) -> Optional[pd.DataFrame]:
    df = get_sales_from_order_items(item_name)
    if df.empty or len(df) < 5:
        logger.warning(f"Insufficient data for {item_name}")
        return None
    df["impact_score"] = 1.0
    try:
        evt = supabase.table("events").select("event_date, impact_score").execute()
        events_df = pd.DataFrame(evt.data) if evt.data else pd.DataFrame()
        if not events_df.empty:
            events_df["event_date"] = pd.to_datetime(events_df["event_date"])
            df = df.merge(events_df, left_on="ds", right_on="event_date", how="left")
            df["impact_score"] = df["impact_score"].fillna(1.0)
    except Exception as e:
        logger.warning(f"Could not load events: {e}")

    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name="ZA")
    if "impact_score" in df and df["impact_score"].nunique() > 1:
        model.add_regressor("impact_score")
    model.fit(df)

    future = model.make_future_dataframe(periods=days_ahead)
    future["impact_score"] = 1.0
    weather_mult = get_klerksdorp_weather(days_ahead)
    tail = future.tail(days_ahead).copy()
    tail["impact_score"] = [weather_mult[i] if i < len(weather_mult) else 0.9 for i in range(days_ahead)]
    future.update(tail)

    forecast = model.predict(future)
    results = forecast.tail(days_ahead).copy()
    results["final_prediction"] = (results["yhat"] * results["impact_score"]).clip(lower=0)
    return results[["ds", "final_prediction", "yhat_lower", "yhat_upper"]]

# ---------------------------------------------------------------------------
# 9. FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI(title="Kota AI: Klerksdorp Edition")

# ─── ADD CORS MIDDLEWARE HERE ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",                            # Local dashboard
        "https://YOUR_GITHUB_USERNAME.github.io",            # GitHub Pages (future)
        "https://restaurant-demand-forecasting-1.onrender.com"  # Render API origin
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# 10. ROUTES
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Kota AI Forecasting API", "status": "online"}

@app.get("/health")
async def health():
    db_status = "not configured"
    item_count = 0
    stock_exists = False

    if supabase:
        try:
            test = supabase.table("order_items").select("*", count="exact").limit(1).execute()
            db_status = "connected"
            item_count = len(test.data) if test.data else 0
            stock_res = supabase.table("stock").select("*", count="exact").limit(1).execute()
            stock_exists = bool(stock_res.data)
        except Exception as e:
            db_status = f"error: {e}"

    return {
        "status": "healthy",
        "location": "Klerksdorp",
        "database": db_status,
        "order_items_count": item_count,
        "stock_table_exists": stock_exists,
        "recommendation": "Add stock table to enable recommendations"
    }

@app.post("/api/forecast")
async def get_forecast(request: ForecastRequest):
    try:
        df = generate_world_class_forecast(request.item_name, request.days_ahead)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if df is None:
        sales = get_sales_from_order_items(request.item_name)
        avg_daily = sales["y"].mean() if not sales.empty else 10
        weekly = avg_daily * 7
        return {
            "item": request.item_name,
            "status": "insufficient_data",
            "total_sold_90d": round(avg_daily * 90, 1),
            "recommendation": "Use manual estimate or add more data"
        }

    forecast_list = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "predicted": round(row["final_prediction"], 1),
            "low": round(row["yhat_lower"], 1),
            "high": round(row["yhat_upper"], 1)
        }
        for _, row in df.iterrows()
    ]
    return {
        "item": request.item_name,
        "days_ahead": request.days_ahead,
        "weekly_total": round(df["final_prediction"].sum(), 1),
        "forecast": forecast_list
    }

@app.post("/api/recommend")
async def get_recommendation(request: RecommendationRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    stock = request.current_stock or get_current_stock_from_table(request.item_name)
    if stock is None:
        raise HTTPException(status_code=400, detail="current_stock required or DB lookup failed")

    df = generate_world_class_forecast(request.item_name, 7)
    weekly_need = df["final_prediction"].sum() if df is not None else get_sales_from_order_items(request.item_name)["y"].mean() * 7
    days_left = stock / (weekly_need / 7) if weekly_need > 0 else 999
    rec = max(0, (weekly_need * 1.5) - stock)
    urgency = "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW"

    return {
        "item": request.item_name,
        "current_stock": stock,
        "weekly_demand": round(weekly_need, 1),
        "days_left": round(days_left, 1),
        "recommended_order": round(rec, 1),
        "urgency": urgency
    }

@app.post("/api/dashboard")
async def get_dashboard(request: DashboardRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")

    results = []
    for itm in request.items:
        name = itm.get("item_name")
        if not name:
            continue
        stock = itm.get("current_stock") or get_current_stock_from_table(name) or 0
        df = generate_world_class_forecast(name, 7)
        weekly = df["final_prediction"].sum() if df is not None else get_sales_from_order_items(name)["y"].mean() * 7
        days_left = stock / (weekly / 7) if weekly > 0 else 999
        rec = max(0, (weekly * 1.5) - stock)
        urg = "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW"
        status = "CRITICAL" if days_left < 2 else "OK" if days_left > 14 else "LOW"

        results.append({
            "item_name": name,
            "current_stock": stock,
            "weekly_demand": round(weekly, 1),
            "days_left": round(days_left, 1),
            "recommended_order": round(rec, 1),
            "urgency": urg,
            "status": status
        })

    results.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["urgency"]])
    return {
        "summary": {
            "total_items": len(results),
            "critical_items": sum(r["urgency"] == "HIGH" for r in results),
            "total_recommended": round(sum(r["recommended_order"] for r in results), 1),
            "timestamp": datetime.now().isoformat()
        },
        "items": results
    }

# ---------------------------------------------------------------------------
# 11. STARTUP EVENT — LOGS API STATUS ON DEPLOY
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Kota AI API Starting...")
    if supabase:
        try:
            test = supabase.table("order_items").select("*", count="exact").limit(5).execute()
            logger.info(f"✅ Found {len(test.data)} order items")
            stock_test = supabase.table("stock").select("*", count="exact").limit(1).execute()
            logger.info(f"✅ Stock table exists: {bool(stock_test.data)}")
        except Exception as e:
            logger.error(f"⚠️ Database query failed: {e}")
    else:
        logger.error("⚠️⚠️ Supabase not configured — backend disabled")

# ---------------------------------------------------------------------------
# 12. LOCAL DEV SERVER — RUN WITH: python main.py
# --------------------------------------------------------------------------- 
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
