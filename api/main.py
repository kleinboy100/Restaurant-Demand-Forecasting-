import os
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
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
    current_stock: int

# --- WEATHER SERVICE ---
def get_klerksdorp_weather(days: int = 7):
    """Fetches precipitation probability for Klerksdorp/Jouberton."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={KLERKSDORP_LAT}&longitude={KLERKSDORP_LON}&daily=precipitation_probability&forecast_days={days}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            probs = response.json().get('daily', {}).get('precipitation_probability', [])
            # Map rain probability to a demand multiplier (Rain = lower demand)
            return [1.0 if p < 25 else 0.7 if p < 60 else 0.4 for p in probs]
    except Exception as e:
        print(f"Weather error: {e}")
    return [0.9] * days  # Default safe multiplier

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
    
    # Convert to DataFrame
    df = pd.DataFrame(res.data)
    
    # Need to get order dates from orders table
    # First, get all unique order_ids
    order_ids = df['order_id'].unique().tolist()
    
    # Get order dates (assuming you have an 'orders' table with created_at or order_date)
    try:
        orders_res = supabase.table("orders") \
            .select("id, created_at, order_date") \
            .in_("id", order_ids) \
            .execute()
        
        if orders_res.data:
            orders_df = pd.DataFrame(orders_res.data)
            # Determine which date column to use
            if 'order_date' in orders_df.columns:
                date_col = 'order_date'
            else:
                date_col = 'created_at'
            
            # Merge with order_items
            df = df.merge(orders_df[['id', date_col]], 
                         left_on='order_id', 
                         right_on='id',
                         how='left')
            
            df['sale_date'] = pd.to_datetime(df[date_col])
        else:
            # If no orders table or dates, use current date as fallback
            df['sale_date'] = pd.Timestamp.now()
    except Exception as e:
        print(f"⚠️ Could not get order dates: {e}")
        # Fallback: assume orders were today if we can't get dates
        df['sale_date'] = pd.Timestamp.now()
    
    # Filter to last X days (if we have dates)
    if 'sale_date' in df.columns:
        cutoff_date = pd.Timestamp.now() - timedelta(days=days_back)
        df = df[df['sale_date'] >= cutoff_date]
    
    # Group by date and sum quantities
    if not df.empty and 'sale_date' in df.columns:
        df_grouped = df.groupby('sale_date')['quantity'].sum().reset_index()
        df_grouped.columns = ['ds', 'y']
        return df_grouped
    
    return pd.DataFrame()

# --- CORE AI ENGINE ---
def generate_world_class_forecast(item_name: str, days_ahead: int):
    # Check if Supabase is configured
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    # 1. Get sales data from order_items table
    df = get_sales_from_order_items(item_name)
    
    if df.empty or len(df) < 5:
        return None  # Not enough data to train

    # 2. Fetch Events (optional - if you have events table)
    df['impact_score'] = 1.0  # Default
    try:
        event_res = supabase.table("events").select("event_date, impact_score").execute()
        if event_res.data:
            df_events = pd.DataFrame(event_res.data)
            if not df_events.empty:
                df_events['event_date'] = pd.to_datetime(df_events['event_date'])
                df = pd.merge(df, df_events, left_on='ds', right_on='event_date', how='left')
                df['impact_score'] = df['impact_score'].fillna(1.0)
    except Exception as e:
        print(f"⚠️ Could not fetch events: {e}")

    # 3. Initialize & Train Prophet
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name='ZA')
    
    # Only add impact_score as regressor if it exists and varies
    if 'impact_score' in df.columns and df['impact_score'].nunique() > 1:
        model.add_regressor('impact_score')
    
    model.fit(df)

    # 4. Create Future Dataframe
    future = model.make_future_dataframe(periods=days_ahead)
    
    # 5. Apply Future Regressors (Predicting Payday impact)
    future['impact_score'] = future['ds'].apply(lambda x: 1.5 if x.day >= 25 else 1.0)
    
    # 6. Predict & Apply Weather Adjustment
    forecast = model.predict(future)
    weather_multipliers = get_klerksdorp_weather(days_ahead)
    
    results = forecast.tail(days_ahead).copy()
    results['final_prediction'] = results['yhat'] * np.array(weather_multipliers)
    
    # Ensure predictions are not negative
    results['final_prediction'] = results['final_prediction'].clip(lower=0)
    
    return results[['ds', 'final_prediction', 'yhat_lower', 'yhat_upper']]

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"message": "Kota AI Forecasting API", "status": "online"}

@app.get("/health")
async def health():
    if supabase:
        # Test connection by checking order_items
        try:
            test = supabase.table("order_items").select("*", count="exact").limit(1).execute()
            db_status = "connected"
            item_count = len(test.data) if test.data else 0
        except Exception as e:
            db_status = f"error: {str(e)}"
            item_count = 0
    else:
        db_status = "not configured"
        item_count = 0
    
    return {
        "status": "healthy", 
        "location": "Klerksdorp",
        "database": db_status,
        "order_items_count": item_count
    }

@app.get("/api/order-stats")
async def get_order_stats():
    """Get statistics about order_items table."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        # Get all order items
        res = supabase.table("order_items").select("*").execute()
        
        if not res.data:
            return {"total_items": 0, "unique_items": []}
        
        df = pd.DataFrame(res.data)
        
        # Group by item_name
        item_stats = df.groupby('item_name').agg({
            'quantity': 'sum',
            'id': 'count',
            'price': lambda x: (x * df.loc[x.index, 'quantity']).sum() / df.loc[x.index, 'quantity'].sum()
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
    """Get demand forecast for a specific item."""
    try:
        data = generate_world_class_forecast(request.item_name, request.days_ahead)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    if data is None:
        # Try to get basic stats if not enough data for forecast
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
        except Exception as e:
            print(f"Error getting order stats: {e}")
        
        raise HTTPException(status_code=404, detail="Insufficient historical data for this item.")
    
    # Format for Frontend
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
    """Get reorder recommendation based on current stock."""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        # Get forecast for next 7 days
        forecast_data = generate_world_class_forecast(request.item_name, 7)
        
        if forecast_data is None:
            # Fallback: estimate based on past sales
            sales_data = get_sales_from_order_items(request.item_name)
            if sales_data.empty:
                avg_daily = 10  # Default if no data
            else:
                avg_daily = sales_data['y'].mean()
            
            weekly_need = avg_daily * 7
        else:
            weekly_need = forecast_data['final_prediction'].sum()
        
        # Calculate days of stock left
        if request.current_stock <= 0:
            days_left = 0
        else:
            days_left = request.current_stock / (weekly_need / 7)
        
        # Calculate recommendation
        safety_factor = 1.5  # 50% safety stock
        recommended_order = max(0, (weekly_need * safety_factor) - request.current_stock)
        
        # Determine urgency
        if days_left < 3:
            urgency = "HIGH"
        elif days_left < 7:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        return {
            "item": request.item_name,
            "current_stock": request.current_stock,
            "predicted_weekly_demand": round(weekly_need, 1),
            "days_of_stock_left": round(days_left, 1),
            "recommended_order": round(recommended_order, 1),
            "urgency": urgency,
            "reorder_threshold": round(weekly_need / 7 * 3, 1)  # 3 days worth
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

# Startup check
@app.on_event("startup")
async def startup_event():
    print("🚀 Kota AI API Starting...")
    if supabase:
        try:
            # Test connection
            res = supabase.table("order_items").select("*", count="exact").limit(5).execute()
            count = len(res.data) if res.data else 0
            print(f"✅ Supabase connected. Found {count} order items")
            
            # List unique items
            if count > 0:
                df = pd.DataFrame(res.data)
                unique_items = df['item_name'].unique()[:5]
                print(f"📋 Sample items: {', '.join(unique_items)}")
        except Exception as e:
            print(f"⚠️ Could not fetch order_items: {str(e)}")
    else:
        print("⚠️ Warning: Supabase not configured. Add SUPABASE_URL and SUPABASE_KEY env vars.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
