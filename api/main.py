import os
import logging
from datetime import datetime, timedelta, date, timezone
from typing import Optional, List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KOTAai Ingredient Intelligence API",
    description="Real-time ingredient tracking with historical demand forecasting",
    version="4.1.0"  # Updated version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase: Optional[Client] = None

class DashboardItem(BaseModel):
    item_name: str
    current_stock: Optional[float] = None

class DashboardRequest(BaseModel):
    items: List[DashboardItem]

class UsageHistoryRequest(BaseModel):
    target_date: str
    ingredient_name: str

# Fallback demands for new ingredients
FALLBACK_DEMANDS = {
    "chips": 500.0, "cheese": 50.0, "russian": 12.0, "lettuce": 8.0,
    "bread": 25.0, "tomato": 10.0, "atchaar": 5.0, "vienna": 15.0,
    "eggs": 20.0, "steak": 6.0, "bacon": 8.0, "polony": 10.0
}

@app.on_event("startup")
async def startup_event():
    global supabase
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        if not supabase_url or not supabase_key:
            logger.error("❌ Supabase credentials missing")
            return
        supabase = create_client(supabase_url, supabase_key)
        logger.info("✅ Database connected successfully")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {str(e)}")

def get_ingredient_usage(target_date: Optional[date] = None) -> Dict[str, float]:
    """Calculates ingredient usage from Supabase tables"""
    if not supabase:
        return {}
    
    try:
        sast_tz = timezone(timedelta(hours=2))
        if target_date is None:
            target_date = datetime.now(sast_tz).date()
        
        start_sast = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=sast_tz)
        end_sast = start_sast + timedelta(days=1)
        start_utc = start_sast.astimezone(timezone.utc).isoformat()
        end_utc = end_sast.astimezone(timezone.utc).isoformat()
        
        # Get orders for target date
        orders = supabase.table("orders")\
            .select("id")\
            .gte("created_at", start_utc)\
            .lt("created_at", end_utc)\
            .execute()
        
        if not orders.data:
            return {}

        order_ids = [o["id"] for o in orders.data]
        
        # Get meals sold
        items = supabase.table("order_items")\
            .select("item_name, quantity")\
            .in_("order_id", order_ids)\
            .execute()
        
        if not items.data:
            return {}

        # Load recipe mappings
        recipes = supabase.table("meal_recipes")\
            .select("meal_name, ingredient_name, quantity_per_meal")\
            .execute()
        
        if not recipes.data:
            return {}

        # Calculate usage
        recipe_map = {}
        for r in recipes.data:
            meal_key = " ".join(r["meal_name"].strip().lower().split())
            if meal_key not in recipe_map:
                recipe_map[meal_key] = []
            recipe_map[meal_key].append((
                " ".join(r["ingredient_name"].strip().lower().split()),
                float(r["quantity_per_meal"])
            ))
        
        usage = {}
        for item in items.data:
            order_meal = " ".join(str(item.get("item_name", "")).strip().lower().split())
            qty = float(item.get("quantity", 1))
            
            for recipe_key, ingredients in recipe_map.items():
                if recipe_key in order_meal or order_meal in recipe_key:
                    for ing_name, ing_qty in ingredients:
                        usage[ing_name] = usage.get(ing_name, 0.0) + (ing_qty * qty)
        
        return usage
        
    except Exception as e:
        logger.exception(f"Usage calculation failed: {str(e)}")
        return {}

@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    try:
        today_usage = get_ingredient_usage()
        sast_tz = timezone(timedelta(hours=2))
        today_date = datetime.now(sast_tz).date()
        historical_usage = {}

        # Calculate historical demand
        for days_ago in range(7):
            check_date = today_date - timedelta(days=days_ago)
            daily_usage = get_ingredient_usage(check_date)
            for ing_key_hist, qty in daily_usage.items():
                historical_usage[ing_key_hist] = historical_usage.get(ing_key_hist, 0.0) + qty

        items_data = []
        for item in request.items:
            ing_name = item.item_name.strip()
            ing_key = " ".join(ing_name.lower().split())
            
            # Get current stock - either from request or Supabase
            current_stock = item.current_stock if item.current_stock is not None else 0.0
            if supabase and current_stock == 0:
                try:
                    res = supabase.table("ingredient_stock")\
                        .select("current_stock")\
                        .ilike("ingredient_name", f"%{ing_name}%")\
                        .limit(1)\
                        .execute()
                    if res.data:
                        current_stock = float(res.data[0]["current_stock"])
                except Exception as e:
                    logger.error(f"Failed to fetch stock for {ing_name}: {str(e)}")

            daily_used = today_usage.get(ing_key, 0.0)
            total_used_7_days = historical_usage.get(ing_key, 0.0)
            
            # Calculate metrics
            weekly_demand = total_used_7_days if total_used_7_days > 0 else FALLBACK_DEMANDS.get(ing_key, 10.0)
            daily_demand = weekly_demand / 7
            days_left = current_stock / daily_demand if daily_demand > 0 else 999
            recommended_order = max(0, (weekly_demand * 1.5) - current_stock)
            
            if days_left < 3:
                urgency, status = "HIGH", "CRITICAL"
            elif days_left < 7:
                urgency, status = "MEDIUM", "LOW"
            else:
                urgency, status = "LOW", "OK"
            
            items_data.append({
                "item_name": ing_name,
                "current_stock": round(current_stock, 1),
                "daily_usage": round(daily_used, 1),
                "weekly_demand": round(weekly_demand, 1),
                "days_left": round(days_left, 1),
                "recommended_order": round(recommended_order, 1),
                "urgency": urgency,
                "status": status,
                "action": "REORDER NOW" if urgency == "HIGH" else "Monitor stock"
            })
        
        # Sort by urgency
        items_data.sort(key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["urgency"], 3))
        
        return {
            "summary": {
                "total_items": len(items_data),
                "critical_items": sum(1 for item in items_data if item["urgency"] == "HIGH"),
                "total_recommended": round(sum(item["recommended_order"] for item in items_data), 1),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "items": items_data
        }
    except Exception as e:
        logger.exception(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ... (keep other endpoints the same as in your original code)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
