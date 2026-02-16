import os
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from prophet import Prophet

app = FastAPI(title="Kota AI: Klerksdorp Edition")

# --- SUPABASE CONFIG ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validate and create Supabase client
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("⚠️ Supabase credentials not set!")
        return None
    
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test connection
        test = client.table("order_items").select("*", count="exact").limit(1).execute()
        print(f"✅ Supabase connected. Found {len(test.data) if test.data else 0} order items")
        return client
    except Exception as e:
        print(f"❌ Supabase connection failed: {str(e)}")
        return None

supabase = get_supabase_client()

# --- COORDINATES FOR KLERKSDORP/JOUBERTON ---
KLERKSDORP_LAT = -26.86
KLERKSDORP_LON = 26.63

# --- DATA MODELS ---
class ForecastRequest(BaseModel):
    item_name: str
    days_ahead: int = 7

class RecommendationRequest(BaseModel):
    item_name: str
    current_stock: Optional[int] = None  # Make optional

class StockUpdateRequest(BaseModel):
    item_name: str
    quantity: int
    transaction_type: str = "ADJUSTMENT"  # PURCHASE, SALE, ADJUSTMENT, WASTAGE
    notes: Optional[str] = None

class DashboardRequest(BaseModel):
    items: List[Dict[str, Any]]  # List of items with current_stock

# --- WEATHER SERVICE ---
def get_klerksdorp_weather(days: int = 7):
    """Fetches precipitation probability for Klerksdorp/Jouberton."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={KLERKSDORP_LAT}&longitude={KLERKSDORP_LON}&daily=precipitation_probability&forecast_days={days}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            probs = response.json().get('daily', {}).get('precipitation_probability', [])
            return [1.0 if p < 25 else 0.7 if p < 60 else 0.4 for p in probs]
    except Exception as e:
        print(f"Weather error: {e}")
    return [0.9] * days

# --- STOCK MANAGEMENT FUNCTIONS ---
def get_current_stock_from_table(item_name: str) -> Optional[int]:
    """Get current stock from stock table if it exists."""
    if not supabase:
        return None
    
    try:
        res = supabase.table("stock").select("current_stock") \
            .eq("item_name", item_name).execute()
        
        if res.data and len(res.data) > 0:
            return res.data[0]["current_stock"]
    except Exception:
        # Stock table doesn't exist or error
        pass
    
    return None

def update_stock_in_table(item_name: str, quantity_change: int, transaction_type: str, notes: str = ""):
    """Update stock in stock table if it exists."""
    if not supabase:
        return False
    
    try:
        # Check if stock table exists
        current_stock = get_current_stock_from_table(item_name)
        
        if current_stock is not None:
            new_stock = max(0, current_stock + quantity_change)
            
            # Update stock table
            supabase.table("stock") \
                .update({
                    "current_stock": new_stock,
                    "last_updated": datetime.now().isoformat()
                }) \
                .eq("item_name", item_name).execute()
            
            # Record transaction
            supabase.table("stock_transactions").insert({
                "item_name": item_name,
                "transaction_type": transaction_type,
                "quantity": quantity_change,
                "previous_stock": current_stock,
                "new_stock": new_stock,
                "notes": notes
            }).execute()
            
            return True
    except Exception as e:
        print(f"Error updating stock table: {e}")
    
    return False

# --- HELPER: GET SALES DATA FROM ORDER_ITEMS ---
def get_sales_from_order_items(item_name: str, days_back: int = 90):
    """Extract sales history from order_items table."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    # Get all order items for this item
    res = supabase.table("order_items") \
        .select("*") \
        .eq("item_name", item_name) \
        .execute()
    
    if not res.data:
        return pd.DataFrame()
    
    df = pd.DataFrame(res.data)
    
    # Try to get order dates from orders table
    try:
        order_ids = df['order_id'].unique().tolist()
        orders_res = supabase.table("orders") \
            .select("id, created_at, order_date") \
            .in_("id", order_ids) \
            .execute()
        
        if orders_res.data:
            orders_df = pd.DataFrame(orders_res.data)
            date_col = 'order_date' if 'order_date' in orders_df.columns else 'created_at'
            
            df = df.merge(orders_df[['id', date_col]], 
                         left_on='order_id', 
                         right_on='id',
                         how='left')
            
            df['sale_date'] = pd.to_datetime(df[date_col])
        else:
            df['sale_date'] = pd.Timestamp.now()
    except Exception:
        df['sale_date'] = pd.Timestamp.now()
    
    if 'sale_date' in df.columns:
        cutoff_date = pd.Timestamp.now() - timedelta(days=days_back)
        df = df[df['sale_date'] >= cutoff_date]
    
    if not df.empty and 'sale_date' in df.columns:
        df_grouped = df.groupby('sale_date')['quantity'].sum().reset_index()
        df_grouped.columns = ['ds', 'y']
        return df_grouped
    
    return pd.DataFrame()

# --- CORE AI ENGINE ---
def generate_world_class_forecast(item_name: str, days_ahead: int):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    df = get_sales_from_order_items(item_name)
    
    if df.empty or len(df) < 5:
        return None
    
    # Add event impact if available
    df['impact_score'] = 1.0
    try:
        event_res = supabase.table("events").select("event_date, impact_score").execute()
        if event_res.data:
            df_events = pd.DataFrame(event_res.data)
            if not df_events.empty:
                df_events['event_date'] = pd.to_datetime(df_events['event_date'])
                df = pd.merge(df, df_events, left_on='ds', right_on='event_date', how='left')
                df['impact_score'] = df['impact_score'].fillna(1.0)
    except Exception:
        pass

    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name='ZA')
    
    if 'impact_score' in df.columns and df['impact_score'].nunique() > 1:
        model.add_regressor('impact_score')
    
    model.fit(df)

    future = model.make_future_dataframe(periods=days_ahead)
    future['impact_score'] = future['ds'].apply(lambda x: 1.5 if x.day >= 25 else 1.0)
    
    forecast = model.predict(future)
    weather_multipliers = get_klerksdorp_weather(days_ahead)
    
    results = forecast.tail(days_ahead).copy()
    results['final_prediction'] = results['yhat'] * np.array(weather_multipliers)
    results['final_prediction'] = results['final_prediction'].clip(lower=0)
    
    return results[['ds', 'final_prediction', 'yhat_lower', 'yhat_upper']]

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"message": "Kota AI Forecasting API", "status": "online"}

@app.get("/health")
async def health():
    if supabase:
        try:
            test = supabase.table("order_items").select("*", count="exact").limit(1).execute()
            db_status = "connected"
            item_count = len(test.data) if test.data else 0
            
            # Check if stock table exists
            try:
                stock_test = supabase.table("stock").select("*", count="exact").limit(1).execute()
                stock_table_exists = len(stock_test.data) > 0
            except:
                stock_table_exists = False
                
        except Exception as e:
            db_status = f"error: {str(e)}"
            item_count = 0
            stock_table_exists = False
    else:
        db_status = "not configured"
        item_count = 0
        stock_table_exists = False
    
    return {
        "status": "healthy", 
        "location": "Klerksdorp",
        "database": db_status,
        "order_items_count": item_count,
        "stock_table_exists": stock_table_exists,
        "recommendation": "Use current_stock parameter for recommendations" if not stock_table_exists else "Stock table available"
    }

@app.get("/api/order-stats")
async def get_order_stats():
    """Get statistics about order_items table."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        res = supabase.table("order_items").select("*").execute()
        
        if not res.data:
            return {"total_items": 0, "unique_items": []}
        
        df = pd.DataFrame(res.data)
        
        item_stats = df.groupby('item_name').agg({
            'quantity': 'sum',
            'id': 'count',
            'price': 'mean'
        }).reset_index()
        
        item_stats.columns = ['item_name', 'total_quantity', 'order_count', 'avg_price']
        
        return {
            "total_items": len(res.data),
            "unique_items_count": len(item_stats),
            "items": item_stats.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting order stats: {str(e)}")

@app.post("/api/forecast")
async def get_forecast(request: ForecastRequest):
    try:
        data = generate_world_class_forecast(request.item_name, request.days_ahead)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    if data is None:
        try:
            res = supabase.table("order_items") \
                .select("*") \
                .eq("item_name", request.item_name) \
                .execute()
            
            if res.data:
                df = pd.DataFrame(res.data)
                total_sold = df['quantity'].sum()
                
                return {
                    "item": request.item_name,
                    "status": "insufficient_data",
                    "message": f"Only {len(df)} sales records found. Need at least 5 for forecasting.",
                    "total_sold_to_date": int(total_sold),
                    "recommendation": "Add more sales data or use manual estimation"
                }
        except:
            pass
        
        raise HTTPException(status_code=404, detail="Insufficient historical data for this item.")
    
    formatted_data = [
        {
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted": round(max(0, row['final_prediction']), 1),
            "low_estimate": round(max(0, row['yhat_lower']), 1),
            "high_estimate": round(max(0, row['yhat_upper']), 1)
        } 
        for _, row in data.iterrows()
    ]
    
    return {
        "item": request.item_name,
        "days_ahead": request.days_ahead,
        "weekly_total": round(data['final_prediction'].sum(), 1),
        "forecast": formatted_data
    }

@app.post("/api/recommend")
async def get_recommendation(request: RecommendationRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        # Try to get current stock from table if not provided
        if request.current_stock is None:
            current_stock = get_current_stock_from_table(request.item_name)
            if current_stock is None:
                raise HTTPException(
                    status_code=400, 
                    detail="current_stock parameter required (stock table doesn't exist or item not in stock table)"
                )
        else:
            current_stock = request.current_stock
        
        # Get forecast
        forecast_data = generate_world_class_forecast(request.item_name, 7)
        
        if forecast_data is None:
            sales_data = get_sales_from_order_items(request.item_name)
            if sales_data.empty:
                avg_daily = 10
            else:
                avg_daily = sales_data['y'].mean()
            weekly_need = avg_daily * 7
        else:
            weekly_need = forecast_data['final_prediction'].sum()
        
        # Calculate metrics
        if current_stock <= 0:
            days_left = 0
        else:
            days_left = current_stock / (weekly_need / 7)
        
        safety_factor = 1.5
        recommended_order = max(0, (weekly_need * safety_factor) - current_stock)
        
        if days_left < 3:
            urgency = "HIGH"
        elif days_left < 7:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        return {
            "item": request.item_name,
            "current_stock": current_stock,
            "predicted_weekly_demand": round(weekly_need, 1),
            "days_of_stock_left": round(days_left, 1),
            "recommended_order": round(recommended_order, 1),
            "urgency": urgency,
            "reorder_now": urgency == "HIGH",
            "estimated_restock_days": round(recommended_order / (weekly_need / 7), 1) if weekly_need > 0 else 7
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@app.post("/api/dashboard")
async def get_dashboard(request: DashboardRequest):
    """Get dashboard data for multiple items with their current stock."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    results = []
    try:
        for item_data in request.items:
            item_name = item_data.get("item_name")
            current_stock = item_data.get("current_stock")
            
            if not item_name:
                continue
            
            # Try to get stock from table if not provided
            if current_stock is None:
                current_stock = get_current_stock_from_table(item_name) or 0
            
            # Get forecast
            forecast_data = generate_world_class_forecast(item_name, 7)
            
            if forecast_data is None:
                sales_data = get_sales_from_order_items(item_name)
                if sales_data.empty:
                    avg_daily = 10
                else:
                    avg_daily = sales_data['y'].mean()
                weekly_need = avg_daily * 7
            else:
                weekly_need = forecast_data['final_prediction'].sum()
            
            # Calculate metrics
            if current_stock <= 0:
                days_left = 0
            else:
                days_left = current_stock / (weekly_need / 7) if weekly_need > 0 else 999
            
            safety_factor = 1.5
            recommended_order = max(0, (weekly_need * safety_factor) - current_stock)
            
            if days_left < 3:
                urgency = "HIGH"
            elif days_left < 7:
                urgency = "MEDIUM"
            else:
                urgency = "LOW"
            
            results.append({
                "item_name": item_name,
                "current_stock": current_stock,
                "weekly_demand": round(weekly_need, 1),
                "days_left": round(days_left, 1),
                "recommended_order": round(recommended_order, 1),
                "urgency": urgency,
                "status": "CRITICAL" if days_left < 2 else "LOW" if days_left > 14 else "OK"
            })
        
        # Sort by urgency
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        results.sort(key=lambda x: urgency_order.get(x["urgency"], 3))
        
        # Calculate summary
        critical_items = sum(1 for r in results if r["urgency"] == "HIGH")
        total_recommended = sum(r["recommended_order"] for r in results)
        
        return {
            "summary": {
                "total_items": len(results),
                "critical_items": critical_items,
                "total_recommended_stock": round(total_recommended, 1),
                "timestamp": datetime.now().isoformat()
            },
            "items": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")

# Optional: Stock management endpoints (only work if stock table exists)
@app.get("/api/stock/{item_name}")
async def get_stock(item_name: str):
    """Get current stock for an item."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        current_stock = get_current_stock_from_table(item_name)
        
        if current_stock is None:
            raise HTTPException(status_code=404, detail="Item not found in stock table")
        
        # Get recommendations for context
        forecast_data = generate_world_class_forecast(item_name, 7)
        
        if forecast_data is None:
            weekly_need = 70  # Default estimate
        else:
            weekly_need = forecast_data['final_prediction'].sum()
        
        return {
            "item_name": item_name,
            "current_stock": current_stock,
            "estimated_weekly_demand": round(weekly_need, 1),
            "days_of_supply": round(current_stock / (weekly_need / 7), 1) if weekly_need > 0 else 999,
            "last_updated": "Now"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stock: {str(e)}")

@app.post("/api/stock/update")
async def update_stock(request: StockUpdateRequest):
    """Update stock for an item."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    success = update_stock_in_table(
        request.item_name, 
        request.quantity, 
        request.transaction_type,
        request.notes or ""
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Stock table doesn't exist or item not found")
    
    return {
        "status": "success",
        "message": f"Stock updated for {request.item_name}"
    }

# Startup check
@app.on_event("startup")
async def startup_event():
    print("🚀 Kota AI API Starting...")
    if supabase:
        try:
            res = supabase.table("order_items").select("*", count="exact").limit(5).execute()
            count = len(res.data) if res.data else 0
            print(f"✅ Order items found: {count}")
            
            # Check for stock table
            try:
                stock_res = supabase.table("stock").select("*", count="exact").limit(1).execute()
                if stock_res.data:
                    print("✅ Stock table exists")
                else:
                    print("⚠️ Stock table doesn't exist - using manual stock input")
            except:
                print("⚠️ Stock table doesn't exist - using manual stock input")
                
        except Exception as e:
            print(f"⚠️ Database issue: {e}")
    else:
        print("⚠️ Warning: Supabase not configured")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
