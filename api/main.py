# ----------------------------------------------------------
# main.py – KOTAai Ingredient Intelligence API (Weather + Prophet)
# --------------------------------------------------------------
import os
import logging
from datetime import datetime, timedelta, date, timezone
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
from prophet import Prophet
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from supabase import create_client, Client
import requests

# -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
app = FastAPI(
    title="KOTAai Restaurant Demand Forecasting API",
    description="Advanced AI‑powered demand forecasting for Kota King Klerksdorp",
    version="4.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
supabase: Optional[Client] = None               # Supabase client
model_cache: Dict[str, Prophet] = {}            # Cached Prophet models

# -----------------------------------------------------------------
class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[int] = None


class DashboardRequest(BaseModel):
    items: List[DashboardItem]


class UsageHistoryRequest(BaseModel):
    target_date: str               # YYYY‑MM‑DD
    ingredient_name: Optional[str] = "All"


# -----------------------------------------------------------------
FALLBACK_DEMANDS = {
    "chips": 500.0, "cheese": 50.0, "russian": 12.0, "lettuce": 8.0,
    "bread": 25.0, "tomato": 10.0, "atchaar": 5.0, "vienna": 15.0,
    "eggs": 20.0, "steak": 6.0, "bacon": 8.0, "polony": 10.0
}

# -----------------------------------------------------------------
def init_supabase() -> bool:
    global supabase
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials missing")
            return False
        supabase = create_client(supabase_url, supabase_key)
        logger.info("✅ Connected to Supabase")
        return True
    except Exception as exc:
        logger.error(f"Supabase connection failed: {exc}")
        return False


@app.on_event("startup")
async def startup():
    init_supabase()
    model_cache.clear()
    logger.info("Prophet model cache cleared – next request will trigger training")

# -----------------------------------------------------------------
def get_klerksdorp_weather(days_ahead: int = 7) -> Dict[str, float]:
    """
    Return {ISO_DATE: impact_factor} where impact_factor < 1.0 when rain is forecast.
    """
    try:
        lat, lon = -26.85, 26.66
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_probability_max",
            "forecast_days": days_ahead,
            "timezone": "Africa/Johannesburg",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        impact = {}
        for i, iso in enumerate(data["daily"]["time"]):
            prob = data["daily"]["precipitation_probability_max"][i]
            if prob < 20:
                factor = 1.0
            elif prob < 60:
                factor = 0.85
            else:
                factor = 0.70
            impact[iso] = factor
        return impact
    except Exception as exc:
        logger.exception(f"Weather fetch failed: {exc}")
        return {
            (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"): 1.0
            for i in range(1, days_ahead + 1)
        }

# -----------------------------------------------------------------
def get_sales_from_order_items(item_name: str, days_back: int = 90) -> pd.DataFrame:
    """
    Historical quantity sold for *item_name* → DataFrame with columns ds, y.
    """
    if not supabase:
        return pd.DataFrame()
    try:
        items = (
            supabase.table("order_items")
            .select("order_id, item_name, quantity")
            .eq("item_name", item_name)
            .execute()
        )
        if not items.data:
            return pd.DataFrame()
        order_ids = [it["order_id"] for it in items.data]

        orders = (
            supabase.table("orders")
            .select("id, created_at")
            .in_("id", order_ids)
            .execute()
        )
        if not orders.data:
            return pd.DataFrame()

        items_df = pd.DataFrame(items.data)
        orders_df = pd.DataFrame(orders.data).rename(columns={"id": "order_id"})
        merged = pd.merge(items_df, orders_df, on="order_id", how="inner")
        merged["created_at"] = pd.to_datetime(merged["created_at"])

        cutoff = datetime.utcnow() - timedelta(days=days_back)
        merged = merged[merged["created_at"] >= cutoff]
        if merged.empty:
            return pd.DataFrame()

        merged["sale_date"] = merged["created_at"].dt.date
        daily = (
            merged.groupby("sale_date")["quantity"]
            .sum()
            .reset_index()
            .rename(columns={"sale_date": "ds", "quantity": "y"})
        )
        daily["ds"] = pd.to_datetime(daily["ds"])
        return daily
    except Exception as exc:
        logger.exception(f"Sales extraction failed for {item_name}: {exc}")
        return pd.DataFrame()

# -----------------------------------------------------------------
def train_prophet_model(df: pd.DataFrame, name: str) -> Prophet:
    m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name="ZA")
    m.fit(df)
    path = f"models/{name.replace(' ', '_')}_prophet.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        import pickle
        pickle.dump(m, f)
    return m


def load_prophet_model(name: str) -> Optional[Prophet]:
    path = f"models/{name.replace(' ', '_')}_prophet.pkl"
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        import pickle
        return pickle.load(f)


def generate_world_class_forecast(item_name: str, days_ahead: int = 7) -> Optional[pd.DataFrame]:
    sales_df = get_sales_from_order_items(item_name)
    if sales_df.empty:
        return None
    model_name = f"{item_name} Forecast"
    cached = load_prophet_model(model_name)
    if cached is None:
        cached = train_prophet_model(sales_df, model_name)
    model_cache[item_name] = cached

    future = cached.make_future_dataframe(periods=days_ahead, freq="D")
    weather = get_klerksdorp_weather(days_ahead)
    future["impact_score"] = [
        weather.get(d.strftime("%Y-%m-%d"), 1.0) for d in future["ds"]
    ]
    forecast = cached.predict(future)
    forecast["final_prediction"] = (forecast["yhat"] * forecast["impact_score"]).clip(lower=0)
    future_only = forecast[forecast["ds"] > sales_df["ds"].max()].copy()
    return future_only[["ds", "final_prediction"]]

# -----------------------------------------------------------------
def get_ingredient_usage(target_date: Optional[date] = None) -> Dict[str, float]:
    if not supabase:
        return {}
    try:
        sast_tz = timezone(timedelta(hours=2))
        if target_date is None:
            target_date = datetime.now(sast_tz).date()
        start = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            0,
            0,
            0,
            tzinfo=sast_tz,
        )
        end = start + timedelta(days=1)

        orders = (
            supabase.table("orders")
            .select("id")
            .gte("created_at", start.isoformat())
            .lt("created_at", end.isoformat())
            .execute()
        )
        if not orders.data:
            return {}
        order_ids = [o["id"] for o in orders.data]

        items = (
            supabase.table("order_items")
            .select("item_name, quantity")
            .in_("order_id", order_ids)
            .execute()
        )
        if not items.data:
            return {}

        recipes = (
            supabase.table("meal_recipes")
            .select("meal_name, ingredient_name, quantity_per_meal")
            .execute()
        )
        if not recipes.data:
            return {}

        recipe_map: Dict[str, List[tuple]] = {}
        for r in recipes.data:
            key = " ".join(r["meal_name"].strip().lower().split())
            if key not in recipe_map:
                recipe_map[key] = []
            recipe_map[key].append(
                (
                    " ".join(r["ingredient_name"].strip().lower().split()),
                    float(r["quantity_per_meal"]),
                )
            )

        usage: Dict[str, float] = {}
        for itm in items.data:
            order_meal = " ".join(str(itm.get("item_name", "")).strip().lower().split())
            qty = float(itm.get("quantity", 1))
            for meal_key, ing_list in recipe_map.items():
                if meal_key in order_meal or order_meal in meal_key:
                    for ing_name, ing_qty in ing_list:
                        usage[ing_name] = usage.get(ing_name, 0.0) + (ing_qty * qty)
        return usage
    except Exception as exc:
        logger.exception(f"Usage calculation failed for {target_date}: {exc}")
        return {}

# -----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    try:
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body style='font-family:sans-serif; text-align:center; padding:50px;'>
                <h1>Dashboard Not Found</h1>
                <p>Place an <code>index.html</code> file in the repo root.</p>
            </body>
        </html>
        """

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "4.1.0",
        "database": "connected" if supabase else "disconnected",
        "prophet_models": list(model_cache.keys()),
    }

@app.post("/api/usage-history")
async def usage_history(req: UsageHistoryRequest):
    try:
        target = datetime.strptime(req.target_date, "%Y-%m-%d").date()
        usage = get_ingredient_usage(target)
        key = req.ingredient_name.strip().lower() if req.ingredient_name else None
        if key:
            amount = usage.get(key, 0.0)
            return {
                "date": req.target_date,
                "ingredient": req.ingredient_name,
                "amount_used": round(amount, 2),
                "unit": "units",
                "all_data": {k: round(v, 2) for k, v in usage.items()},
            }
        else:
            return {
                "date": req.target_date,
                "ingredient": "All",
                "all_data": {k: round(v, 2) for k, v in usage.items()},
            }
    except Exception as exc:
        logger.exception(f"Usage‑history error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/dashboard")
async def dashboard(req: DashboardRequest):
    try:
        today_usage = get_ingredient_usage()
        sast_tz = timezone(timedelta(hours=2))
        today = datetime.now(sast_tz).date()
        # historic 7‑day totals (fallback)
        hist = {}
        for d in range(7):
            h = get_ingredient_usage(today - timedelta(days=d))
            for k, v in h.items():
                hist[k] = hist.get(k, 0.0) + v

        out: List[Dict] = []
        rec_total = 0.0
        critical = 0

        for entry in req.items:
            name = entry.item_name.strip()
            key = " ".join(name.lower().split())

            # stock
            stock = entry.current_stock if entry.current_stock is not None else 0.0
            if supabase and stock == 0:
                res = (
                    supabase.table("ingredient_stock")
                    .select("current_stock")
                    .ilike("ingredient_name", f"%{name}%")
                    .limit(1)
                    .execute()
                )
                stock = float(res.data[0]["current_stock"]) if res.data else 0.0

            # forecast
            forecast_df = generate_world_class_forecast(name, 7)
            if forecast_df is not None:
                weekly_demand = float(forecast_df["final_prediction"].sum())
            else:
                weekly_demand = hist.get(key, FALLBACK_DEMANDS.get(key, 10.0))

            daily = weekly_demand / 7.0
            days_left = stock / daily if daily > 0 else 999.0
            recommend = max(0.0, (weekly_demand * 1.5) - stock)

            if days_left < 3:
                urg, st = "HIGH", "CRITICAL"
                critical += 1
            elif days_left < 7:
                urg, st = "MEDIUM", "LOW"
            else:
                urg, st = "LOW", "OK"

            rec_total += recommend
            out.append(
                {
                    "item_name": name,
                    "current_stock": round(stock, 1),
                    "daily_usage": round(today_usage.get(key, 0.0), 1),
                    "weekly_demand": round(weekly_demand, 1),
                    "days_left": round(days_left, 1),
                    "recommended_order": round(recommend, 1),
                    "urgency": urg,
                    "status": st,
                    "action": "REORDER NOW" if urg == "HIGH" else "Monitor stock",
                }
            )

        out.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["urgency"], 3))

        return {
            "summary": {
                "total_items": len(out),
                "critical_items": critical,
                "total_recommended": round(rec_total, 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "items": out,
        }
    except Exception as exc:
        logger.exception(f"Dashboard error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

# -----------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
