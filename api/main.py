import os
import logging
from datetime import datetime, timedelta, date, timezone
from typing import Optional, List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KOTAai Ingredient Intelligence API",
    description="Real-time ingredient tracking with historical demand forecasting",
    version="4.0.0"
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
    """
    CALCULATES INGREDIENT USAGE FROM 3 SUPABASE TABLES:
    1. orders → Orders created on target_date (SAST timezone)
    2. order_items → Meals sold (item_name, quantity)
    3. meal_recipes → Ingredient mappings (meal_name → ingredient + qty)
    
    RETURNS: { "chips": 450.0, "bread": 30.0, ... } (normalized lowercase keys)
    """
    if not supabase:
        return {}
    
    try:
        sast_tz = timezone(timedelta(hours=2))
        if target_date is None:
            target_date = datetime.now(sast_tz).date()
        
        # Convert SAST date boundaries → UTC for Supabase query
        start_sast = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=sast_tz)
        end_sast = start_sast + timedelta(days=1)
        start_utc = start_sast.astimezone(timezone.utc).isoformat()
        end_utc = end_sast.astimezone(timezone.utc).isoformat()
        
        # STEP 1: Get orders for target date
        orders = supabase.table("orders")\
            .select("id")\
            .gte("created_at", start_utc)\
            .lt("created_at", end_utc)\
            .execute()
        if not orders.data:
            logger.info(f"ℹ️ No orders found for {target_date} (SAST)")
            return {}
        
        order_ids = [o["id"] for o in orders.data]
        logger.info(f"✅ Found {len(order_ids)} orders for {target_date}")
        
        # STEP 2: Get meals sold
        items = supabase.table("order_items")\
            .select("item_name, quantity")\
            .in_("order_id", order_ids)\
            .execute()
        if not items.data:
            logger.warning("⚠️ Orders exist but no order_items found")
            return {}
        logger.info(f"✅ Found {len(items.data)} order items")
        
        # STEP 3: Load recipe mappings
        recipes = supabase.table("meal_recipes")\
            .select("meal_name, ingredient_name, quantity_per_meal")\
            .execute()
        if not recipes.data:
            logger.error("❌ CRITICAL: meal_recipes table is EMPTY!")
            return {}
        logger.info(f"✅ Loaded {len(recipes.data)} recipe entries")
        
        # Build normalized recipe map
        recipe_map = {}
        for r in recipes.data:
            meal_key = " ".join(r["meal_name"].strip().lower().split())
            if meal_key not in recipe_map:
                recipe_map[meal_key] = []
            recipe_map[meal_key].append((
                " ".join(r["ingredient_name"].strip().lower().split()),
                float(r["quantity_per_meal"])
            ))
        
        # STEP 4: Calculate usage with BIDIRECTIONAL MATCHING
        usage = {}
        unmatched = []
        
        for item in items.data:
            order_meal = " ".join(str(item.get("item_name", "")).strip().lower().split())
            qty = float(item.get("quantity", 1))
            
            # SMART MATCH: Handles "Toast" vs "Toast and chips" in BOTH directions
            matched_recipe = None
            for recipe_key in recipe_map.keys():
                if recipe_key in order_meal or order_meal in recipe_key:
                    matched_recipe = recipe_key
                    break
            
            if matched_recipe:
                for ing_name, ing_qty in recipe_map[matched_recipe]:
                    total_qty = ing_qty * qty
                    usage[ing_name] = usage.get(ing_name, 0.0) + total_qty
            else:
                unmatched.append(order_meal)
        
        if unmatched:
            logger.warning(f"⚠️ Unmatched meals ({len(unmatched)}): {list(set(unmatched))[:5]}")
        logger.info(f"✅ Calculated usage for {len(usage)} ingredients")
        return usage
        
    except Exception as e:
        logger.exception(f"Usage calculation failed: {str(e)}")
        return {}

# ============================================================================
# ✅ DEBUG ENDPOINT: Verify data flow
# ============================================================================
@app.get("/api/debug/ingredient-flow")
async def debug_ingredient_flow():
    """VERIFICATION ENDPOINT - Shows EXACTLY what the API sees"""
    try:
        today = datetime.now(timezone(timedelta(hours=2))).date()
        usage = get_ingredient_usage(today)
        
        # Get sample data
        sast_tz = timezone(timedelta(hours=2))
        start_sast = datetime(today.year, today.month, today.day, 0, 0, 0, tzinfo=sast_tz)
        end_sast = start_sast + timedelta(days=1)
        start_utc = start_sast.astimezone(timezone.utc).isoformat()
        end_utc = end_sast.astimezone(timezone.utc).isoformat()
        
        # Sample order meals
        orders = supabase.table("orders").select("id").gte("created_at", start_utc).lt("created_at", end_utc).execute()
        order_ids = [o["id"] for o in orders.data] if orders.data else []
        items_sample = []
        if order_ids:
            items_res = supabase.table("order_items").select("item_name").in_("order_id", order_ids).limit(5).execute()
            items_sample = [i["item_name"] for i in items_res.data] if items_res.data else []
        
        # Sample recipe meals
        recipes_res = supabase.table("meal_recipes").select("meal_name").limit(5).execute()
        recipes_sample = [r["meal_name"] for r in recipes_res.data] if recipes_res.data else []
        
        return {
            "status": "success",
            "date_checked": today.isoformat(),
            "timezone": "SAST (UTC+2)",
            "data_sources": {
                "orders_table_query": f"UTC range: {start_utc} to {end_utc}",
                "order_items_sample": items_sample,
                "meal_recipes_sample": recipes_sample
            },
            "diagnosis": {
                "orders_found": len(order_ids),
                "order_items_count": len(items_sample),
                "recipes_loaded": len(recipes_res.data) if recipes_res else 0,
                "ingredient_usage": {k: round(v, 2) for k, v in usage.items()},
                "troubleshooting": [
                    "✅ If orders_found=0: No orders today in SAST timezone",
                    "✅ If recipes_loaded=0: RLS policy blocking meal_recipes access",
                    "✅ If usage empty but orders exist: Meal names don't match between tables",
                    "💡 FIX: Run SQL patch to TRIM meal names + add missing recipes (see docs)"
                ]
            }
        }
    except Exception as e:
        logger.exception("Debug endpoint failed")
        return {"status": "error", "message": str(e)}

# ============================================================================
# 🔑 MAIN DASHBOARD ENDPOINT (WITH HISTORICAL DEMAND CALCULATION)
# ============================================================================
@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    try:
        # Get today's usage
        today_usage = get_ingredient_usage()
        
        # 🔥 CRITICAL FIX: Calculate demand from ACTUAL SALES (last 7 days)
        sast_tz = timezone(timedelta(hours=2))
        today_date = datetime.now(sast_tz).date()
        historical_usage = {}  # {ingredient_key: total_quantity_used_in_7_days}
        
        logger.info(f"📊 Calculating historical demand (last 7 days)...")
        for days_ago in range(7):
            check_date = today_date - timedelta(days=days_ago)
            daily_usage = get_ingredient_usage(check_date)
            for ing_key_hist, qty in daily_usage.items():
                historical_usage[ing_key_hist] = historical_usage.get(ing_key_hist, 0.0) + qty
        
        logger.info(f"✅ Historical usage calculated for {len(historical_usage)} ingredients")
        
        # Fallback ONLY for brand-new ingredients with zero history
        fallback_demands = {
            "chips": 500.0, "melted cheese": 50.0, "russian": 12.0, "lettuce": 8.0,
            "bread": 25.0, "tomato": 10.0, "atchar": 5.0, "vienna": 15.0,
            "egg": 20.0, "steak": 6.0
        }
        
        items_data = []
        total_recommended = 0.0
        critical_count = 0
        
        for item in request.items:
            ing_name = item.item_name.strip()
            ing_key = " ".join(ing_name.lower().split())
            
            # Current stock
            current_stock = item.current_stock
            if current_stock is None and supabase:
                res = supabase.table("ingredient_stock")\
                    .select("current_stock")\
                    .ilike("ingredient_name", ing_name)\
                    .limit(1)\
                    .execute()
                current_stock = float(res.data[0]["current_stock"]) if res.data else 0.0
            
            # Daily usage (today)
            daily_used = today_usage.get(ing_key, 0.0)
            
            # ✅ FIXED: Use REAL SALES DATA for demand forecasting
            total_used_7_days = historical_usage.get(ing_key, 0.0)
            if total_used_7_days > 0:
                weekly_demand = total_used_7_days  # Project same usage for next week
                logger.info(f"✅ {ing_name}: Historical demand = {weekly_demand:.1f} (from actual sales)")
            else:
                weekly_demand = fallback_demands.get(ing_key, 10.0)
                logger.warning(f"⚠️ {ing_name}: Using fallback demand = {weekly_demand:.1f} (no historical data)")
            
            # Metrics
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
        logger.exception(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# USAGE HISTORY ENDPOINT
# ============================================================================
@app.post("/api/usage-history")
async def usage_history(request: UsageHistoryRequest):
    try:
        target_date = datetime.strptime(request.target_date, "%Y-%m-%d").date()
        usage = get_ingredient_usage(target_date)
        amount = usage.get(" ".join(request.ingredient_name.strip().lower().split()), 0.0)
        return {
            "date": request.target_date,
            "ingredient": request.ingredient_name,
            "amount_used": round(amount, 2),
            "unit": "units"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")

# ============================================================================
# HEALTH CHECK
# ============================================================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "4.0.0",
        "database": "connected" if supabase else "disconnected",
        "features": {
            "realtime_usage": "enabled",
            "historical_demand": "enabled",
            "debug_endpoint": "/api/debug/ingredient-flow"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
