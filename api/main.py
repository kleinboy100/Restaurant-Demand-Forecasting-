import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from prophet import Prophet

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demand_forecaster")

app = FastAPI(
    title="Demand Forecasting API",
    description="Prophet-backed demand forecasting and zero-safe model evaluation.",
    version="2.1.0",
    redirect_slashes=True  # Redirects /api/forecast/ to /api/forecast automatically
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase Client setup
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://your-supabase-url.supabase.co")
SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "your-supabase-key")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    logger.error(f"Supabase connection warning: {str(e)}")
    supabase = None


# --- SCHEMAS ---

class MetricDetail(BaseModel):
    accuracy: float
    mape: float
    mae: float
    rmse: float
    active_days_evaluated: int
    total_days_in_window: int
    status_note: Optional[str] = None

class PerformanceResponse(BaseModel):
    revenue: MetricDetail
    meals: MetricDetail


# --- ETL & DATA PIPELINE ---

def fetch_aggregated_daily_data() -> pd.DataFrame:
    if not supabase:
        return pd.DataFrame(columns=["ds", "revenue", "meals"])

    try:
        orders_res = supabase.table("orders").select("id, created_at, total_price, status").execute()
        orders_data = orders_res.data or []

        if not orders_data:
            return pd.DataFrame(columns=["ds", "revenue", "meals"])

        df_orders = pd.DataFrame(orders_data)
        
        if "status" in df_orders.columns:
            df_orders = df_orders[df_orders["status"].str.lower() != "cancelled"]

        if df_orders.empty:
            return pd.DataFrame(columns=["ds", "revenue", "meals"])

        df_orders["ds"] = pd.to_datetime(df_orders["created_at"]).dt.tz_localize(None).dt.floor("D")
        df_orders["total_price"] = pd.to_numeric(df_orders["total_price"], errors="coerce").fillna(0.0)

        try:
            items_res = supabase.table("order_items").select("order_id, quantity").execute()
            items_data = items_res.data or []
            if items_data:
                df_items = pd.DataFrame(items_data)
                df_items["quantity"] = pd.to_numeric(df_items["quantity"], errors="coerce").fillna(1)
                order_qty = df_items.groupby("order_id")["quantity"].sum().reset_index()
                df_orders = df_orders.merge(order_qty, left_on="id", right_on="order_id", how="left")
                df_orders["quantity"] = df_orders["quantity"].fillna(1)
            else:
                df_orders["quantity"] = 1
        except Exception:
            df_orders["quantity"] = 1

        daily_summary = df_orders.groupby("ds").agg(
            revenue=("total_price", "sum"),
            meals=("quantity", "sum")
        ).reset_index()

        min_date = daily_summary["ds"].min()
        max_date = daily_summary["ds"].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        
        full_df = pd.DataFrame({"ds": full_date_range}).merge(daily_summary, on="ds", how="left").fillna(0.0)
        return full_df.sort_values("ds").reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error reading Supabase data: {str(e)}")
        return pd.DataFrame(columns=["ds", "revenue", "meals"])


def train_prophet(df: pd.DataFrame, target_column: str) -> Prophet:
    train_data = df[["ds", target_column]].rename(columns={target_column: "y"})
    
    if len(train_data) < 2 or train_data["y"].nunique() <= 1:
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        fallback_df = pd.DataFrame({
            "ds": [pd.Timestamp.now() - pd.Timedelta(days=1), pd.Timestamp.now()],
            "y": [1.0, 1.0]
        })
        model.fit(fallback_df)
        return model

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(train_data)
    return model


# --- ROUTE HANDLERS ---

@app.get("/")
def read_root():
    """Root endpoint to verify operational status."""
    return {
        "status": "online",
        "service": "Demand Forecasting Engine",
        "interactive_docs": "/docs",
        "endpoints": {
            "performance": "/api/model-performance",
            "forecast": "/api/forecast",
            "history": "/api/history"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/model-performance", response_model=PerformanceResponse)
def get_model_performance(backtest_days: int = Query(default=14, ge=1, le=90)):
    df = fetch_aggregated_daily_data()

    if df.empty or len(df) <= backtest_days + 2:
        empty_metric = MetricDetail(
            accuracy=0.0, mape=0.0, mae=0.0, rmse=0.0,
            active_days_evaluated=0, total_days_in_window=0,
            status_note="Insufficient transaction history for requested backtest window."
        )
        return PerformanceResponse(revenue=empty_metric, meals=empty_metric)

    max_date = df["ds"].max()
    cutoff_date = max_date - pd.Timedelta(days=backtest_days)

    train_df = df[df["ds"] <= cutoff_date]
    test_df = df[df["ds"] > cutoff_date]

    if train_df.empty or test_df.empty:
        raise HTTPException(status_code=400, detail="Backtest window exceeds date range.")

    rev_model = train_prophet(train_df, "revenue")
    meal_model = train_prophet(train_df, "meals")

    future_dates = test_df[["ds"]].copy()
    pred_rev = rev_model.predict(future_dates)
    pred_meals = meal_model.predict(future_dates)

    act_rev = test_df["revenue"].values
    prd_rev = np.maximum(0.0, pred_rev["yhat"].values)

    act_m = test_df["meals"].values
    prd_m = np.maximum(0.0, pred_meals["yhat"].values)

    mae_rev = float(np.mean(np.abs(prd_rev - act_rev)))
    mae_meals = float(np.mean(np.abs(prd_m - act_m)))

    rmse_rev = float(np.sqrt(np.mean((prd_rev - act_rev) ** 2)))
    rmse_meals = float(np.sqrt(np.mean((prd_m - act_m) ** 2)))

    # Zero-safe masking for MAPE
    rev_mask = act_rev > 0
    meal_mask = act_m > 0

    mape_rev = float(np.mean(np.abs((act_rev[rev_mask] - prd_rev[rev_mask]) / act_rev[rev_mask])) * 100.0) if np.any(rev_mask) else 0.0
    mape_meals = float(np.mean(np.abs((act_m[meal_mask] - prd_m[meal_mask]) / act_m[meal_mask])) * 100.0) if np.any(meal_mask) else 0.0

    accuracy_rev = float(np.clip(100.0 - mape_rev, 0.0, 100.0))
    accuracy_meals = float(np.clip(100.0 - mape_meals, 0.0, 100.0))

    return PerformanceResponse(
        revenue=MetricDetail(
            accuracy=round(accuracy_rev, 2),
            mape=round(mape_rev, 2),
            mae=round(mae_rev, 2),
            rmse=round(rmse_rev, 2),
            active_days_evaluated=int(np.sum(rev_mask)),
            total_days_in_window=len(test_df),
            status_note="Zero-sale operating days masked during MAPE calculation."
        ),
        meals=MetricDetail(
            accuracy=round(accuracy_meals, 2),
            mape=round(mape_meals, 2),
            mae=round(mae_meals, 2),
            rmse=round(rmse_meals, 2),
            active_days_evaluated=int(np.sum(meal_mask)),
            total_days_in_window=len(test_df),
            status_note="Zero-sale operating days masked during MAPE calculation."
        )
    )


@app.get("/api/forecast")
def get_forecast(days: int = Query(default=7, ge=1, le=30)):
    df = fetch_aggregated_daily_data()

    if df.empty:
        return {"forecast": [], "note": "No transaction records found."}

    rev_model = train_prophet(df, "revenue")
    meal_model = train_prophet(df, "meals")

    future_dates = rev_model.make_future_dataframe(periods=days, freq="D")
    forecast_rev = rev_model.predict(future_dates)
    forecast_meals = meal_model.predict(future_dates)

    future_rev_slice = forecast_rev.tail(days)
    future_meal_slice = forecast_meals.tail(days)

    results: List[Dict[str, Any]] = []

    for r_row, m_row in zip(future_rev_slice.to_dict("records"), future_meal_slice.to_dict("records")):
        results.append({
            "ds": r_row["ds"].strftime("%Y-%m-%d"),
            "predicted_revenue": round(max(0.0, float(r_row["yhat"])), 2),
            "revenue_lower": round(max(0.0, float(r_row["yhat_lower"])), 2),
            "revenue_upper": round(max(0.0, float(r_row["yhat_upper"])), 2),
            "predicted_meals": int(round(max(0.0, float(m_row["yhat"])))),
            "meals_lower": int(round(max(0.0, float(m_row["yhat_lower"])))),
            "meals_upper": int(round(max(0.0, float(m_row["yhat_upper"]))))
        })

    return {"forecast": results}


@app.get("/api/history")
def get_historical_data(days: int = Query(default=30, ge=7, le=365)):
    df = fetch_aggregated_daily_data()
    if df.empty:
        return {"history": []}

    slice_df = df.tail(days)
    slice_df["ds"] = slice_df["ds"].dt.strftime("%Y-%m-%d")
    return {"history": slice_df.to_dict("records")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
