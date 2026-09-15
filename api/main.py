import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from prophet import Prophet

# ==========================================
# 1. LOGGING & APP CONFIGURATION
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demand_forecaster")

app = FastAPI(
    title="Demand Forecasting & Backtesting Engine",
    description="Production-grade FastAPI service providing Prophet-backed forecasts and zero-safe model backtesting.",
    version="2.1.0"
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase Client Instantiation
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://your-supabase-project.supabase.co")
SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_ANON_KEY", "your-supabase-key"))

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {str(e)}")
    supabase = None

# ==========================================
# 2. SCHEMAS
# ==========================================

class ForecastItem(BaseModel):
    ds: str = Field(..., description="Date formatted as YYYY-MM-DD")
    predicted_revenue: float = Field(..., description="Estimated total revenue (ZAR)")
    revenue_lower: float = Field(..., description="Lower confidence bound for revenue")
    revenue_upper: float = Field(..., description="Upper confidence bound for revenue")
    predicted_meals: int = Field(..., description="Estimated total meal units required")
    meals_lower: int = Field(..., description="Lower confidence bound for meals")
    meals_upper: int = Field(..., description="Upper confidence bound for meals")

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

# ==========================================
# 3. DATA ACCESS & ETL LAYER
# ==========================================

def fetch_aggregated_daily_data() -> pd.DataFrame:
    """
    Retrieves sales and item records from Supabase, aggregates daily total revenue 
    and meal quantities, and fills missing timeline gaps with explicit zeroes.
    """
    if not supabase:
        logger.warning("Supabase client is uninitialized. Returning empty DataFrame.")
        return pd.DataFrame(columns=["ds", "revenue", "meals"])

    try:
        # Fetch orders
        orders_res = supabase.table("orders").select("id, created_at, total_price, status").execute()
        orders_data = orders_res.data or []

        if not orders_data:
            return pd.DataFrame(columns=["ds", "revenue", "meals"])

        df_orders = pd.DataFrame(orders_data)
        
        # Exclude cancelled transactions if status column exists
        if "status" in df_orders.columns:
            df_orders = df_orders[df_orders["status"].str.lower() != "cancelled"]

        if df_orders.empty:
            return pd.DataFrame(columns=["ds", "revenue", "meals"])

        # Parse timestamps into normalized dates
        df_orders["ds"] = pd.to_datetime(df_orders["created_at"]).dt.tz_localize(None).dt.floor("D")
        df_orders["total_price"] = pd.to_numeric(df_orders["total_price"], errors="coerce").fillna(0.0)

        # Attempt to aggregate actual item counts from order_items table
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
        except Exception as item_err:
            logger.warning(f"Failed to join order_items, falling back to 1 unit per order: {str(item_err)}")
            df_orders["quantity"] = 1

        # Aggregate metrics per single calendar day
        daily_summary = df_orders.groupby("ds").agg(
            revenue=("total_price", "sum"),
            meals=("quantity", "sum")
        ).reset_index()

        # Reindex across a contiguous calendar range to ensure accurate backtesting time windows
        min_date = daily_summary["ds"].min()
        max_date = daily_summary["ds"].max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        
        full_df = pd.DataFrame({"ds": full_date_range}).merge(daily_summary, on="ds", how="left").fillna(0.0)
        return full_df.sort_values("ds").reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error fetching historical records: {str(e)}")
        return pd.DataFrame(columns=["ds", "revenue", "meals"])

# ==========================================
# 4. PROPHET MODELING PIPELINE
# ==========================================

def train_prophet(df: pd.DataFrame, target_column: str) -> Prophet:
    """
    Fits a Facebook Prophet time-series model on target series ('revenue' or 'meals').
    Handles low sample edge cases with synthetic stabilization.
    """
    train_data = df[["ds", target_column]].rename(columns={target_column: "y"})
    
    # Require at least 2 non-null historical points
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
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    model.fit(train_data)
    return model

# ==========================================
# 5. ENDPOINTS & BUSINESS LOGIC
# ==========================================

@app.get("/health", status_code=200)
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database_connected": supabase is not None
    }


@app.get("/api/model-performance", response_model=PerformanceResponse)
def get_model_performance(backtest_days: int = Query(default=14, ge=1, le=90)):
    """
    Evaluates forecasting performance on a sliding historical backtest window.
    Filters zero-activity days out of MAPE calculations to eliminate infinite errors.
    """
    df = fetch_aggregated_daily_data()

    if df.empty or len(df) <= backtest_days + 2:
        empty_metric = MetricDetail(
            accuracy=0.0, mape=0.0, mae=0.0, rmse=0.0,
            active_days_evaluated=0, total_days_in_window=0,
            status_note="Insufficient transaction history for evaluation window."
        )
        return PerformanceResponse(revenue=empty_metric, meals=empty_metric)

    # Cutoff splitting logic
    max_date = df["ds"].max()
    cutoff_date = max_date - pd.Timedelta(days=backtest_days)

    train_df = df[df["ds"] <= cutoff_date]
    test_df = df[df["ds"] > cutoff_date]

    if train_df.empty or test_df.empty:
        raise HTTPException(status_code=400, detail="Requested backtest window too large for available dataset.")

    # Train independent models
    rev_model = train_prophet(train_df, "revenue")
    meal_model = train_prophet(train_df, "meals")

    # Generate out-of-sample predictions
    future_dates = test_df[["ds"]].copy()
    pred_rev = rev_model.predict(future_dates)
    pred_meals = meal_model.predict(future_dates)

    # Actuals vs Predictions
    act_rev = test_df["revenue"].values
    prd_rev = np.maximum(0.0, pred_rev["yhat"].values)

    act_m = test_df["meals"].values
    prd_m = np.maximum(0.0, pred_meals["yhat"].values)

    # --- ERROR METRICS CALCULATION ---
    
    # 1. Standard Linear Metrics (MAE & RMSE)
    mae_rev = float(np.mean(np.abs(prd_rev - act_rev)))
    mae_meals = float(np.mean(np.abs(prd_m - act_m)))

    rmse_rev = float(np.sqrt(np.mean((prd_rev - act_rev) ** 2)))
    rmse_meals = float(np.sqrt(np.mean((prd_m - act_m) ** 2)))

    # 2. Masking Zero-Actual Days for Safe MAPE Calculation
    rev_mask = act_rev > 0
    meal_mask = act_m > 0

    if np.any(rev_mask):
        mape_rev = float(np.mean(np.abs((act_rev[rev_mask] - prd_rev[rev_mask]) / act_rev[rev_mask])) * 100.0)
    else:
        mape_rev = 0.0

    if np.any(meal_mask):
        mape_meals = float(np.mean(np.abs((act_m[meal_mask] - prd_m[meal_mask]) / act_m[meal_mask])) * 100.0)
    else:
        mape_meals = 0.0

    # 3. Bounded Accuracy Metrics (Bounded within 0.0% - 100.0%)
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
            status_note="Metrics evaluated safely by excluding zero-sale operating days."
        ),
        meals=MetricDetail(
            accuracy=round(accuracy_meals, 2),
            mape=round(mape_meals, 2),
            mae=round(mae_meals, 2),
            rmse=round(rmse_meals, 2),
            active_days_evaluated=int(np.sum(meal_mask)),
            total_days_in_window=len(test_df),
            status_note="Metrics evaluated safely by excluding zero-sale operating days."
        )
    )


@app.get("/api/forecast")
def get_forecast(days: int = Query(default=7, ge=1, le=30)):
    """
    Generates forward-looking daily demand forecasts for revenue and total meals.
    """
    df = fetch_aggregated_daily_data()

    if df.empty:
        return {"forecast": [], "note": "No transaction records found to generate baseline forecast."}

    # Train model on full complete dataset
    rev_model = train_prophet(df, "revenue")
    meal_model = train_prophet(df, "meals")

    future_dates = rev_model.make_future_dataframe(periods=days, freq="D")

    forecast_rev = rev_model.predict(future_dates)
    forecast_meals = meal_model.predict(future_dates)

    # Extract future slice
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
    """
    Returns sanitized daily historical totals for visualization and audit checks.
    """
    df = fetch_aggregated_daily_data()
    if df.empty:
        return {"history": []}

    slice_df = df.tail(days)
    slice_df["ds"] = slice_df["ds"].dt.strftime("%Y-%m-%d")
    return {"history": slice_df.to_dict("records")}

# ==========================================
# 6. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
