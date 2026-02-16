import os
import pandas as pd
import requests
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from prophet import Prophet
from dotenv import load_dotenv

# Load Environment Variables (Ensure these are in your .env file or server config)
load_dotenv()

app = FastAPI(title="Kota AI: Klerksdorp Edition")

# --- SUPABASE CONFIG ---
SUPABASE_URL = os.getenv("https://aweceghpsgqhilydnoun.supabase.co")
SUPABASE_KEY = os.getenv("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9") # Use service role for backend access
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- COORDINATES FOR KLERKSDORP/JOUBERTON ---
KLERKSDORP_LAT = -26.86
KLERKSDORP_LON = 26.63

# --- DATA MODELS ---
class ForecastRequest(BaseModel):
    item_name: str
    days_ahead: int = 7

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
    return [0.9] * days # Default safe multiplier

# --- CORE AI ENGINE ---
def generate_world_class_forecast(item_name: str, days_ahead: int):
    # 1. Fetch Sales History from Supabase
    # Assumes table 'sales_history' with columns: sale_date, quantity, item_name
    res = supabase.table("sales_history").select("sale_date, quantity") \
        .eq("item_name", item_name).execute()
    
    if not res.data or len(res.data) < 5:
        return None # Not enough data to train

    # 2. Prepare Data for Prophet
    df = pd.DataFrame(res.data).rename(columns={"sale_date": "ds", "quantity": "y"})
    df['ds'] = pd.to_datetime(df['ds'])

    # 3. Fetch Events (Spikes like Payday or Soccer Derby)
    # Assumes table 'events' with columns: event_date, impact_score
    event_res = supabase.table("events").select("event_date, impact_score").execute()
    df_events = pd.DataFrame(event_res.data)

    if not df_events.empty:
        df_events['event_date'] = pd.to_datetime(df_events['event_date'])
        df = pd.merge(df, df_events, left_on='ds', right_on='event_date', how='left')
        df['impact_score'] = df['impact_score'].fillna(1.0)
    else:
        df['impact_score'] = 1.0

    # 4. Initialize & Train Prophet
    # 'ZA' adds South African public holidays automatically
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.add_country_holidays(country_name='ZA')
    model.add_regressor('impact_score')
    model.fit(df)

    # 5. Create Future Dataframe
    future = model.make_future_dataframe(periods=days_ahead)
    
    # 6. Apply Future Regressors (Predicting Payday impact)
    # Logic: If date is between 25th and 31st, assume 1.5x impact if not specified
    future['impact_score'] = future['ds'].apply(lambda x: 1.5 if x.day >= 25 else 1.0)
    
    # 7. Predict & Apply Weather Adjustment
    forecast = model.predict(future)
    weather_multipliers = get_klerksdorp_weather(days_ahead)
    
    results = forecast.tail(days_ahead).copy()
    results['final_prediction'] = results['yhat'] * np.array(weather_multipliers)
    
    return results[['ds', 'final_prediction', 'yhat_lower', 'yhat_upper']]

# --- ENDPOINTS ---
@app.get("/health")
def health():
    return {"status": "online", "location": "Klerksdorp"}

@app.post("/api/forecast")
async def get_forecast(request: ForecastRequest):
    data = generate_world_class_forecast(request.item_name, request.days_ahead)
    
    if data is None:
        raise HTTPException(status_code=404, detail="Insufficient historical data for this item.")
    
    # Format for Frontend (Lovable/React)
    formatted_data = [
        {
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted": round(max(0, row['final_prediction']), 1),
            "safety_stock": round(max(0, row['yhat_upper']), 1) # Highest likely demand
        } 
        for _, row in data.iterrows()
    ]
    
    return {
        "item": request.item_name,
        "weekly_total": round(data['final_prediction'].sum(), 1),
        "forecast": formatted_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))    },
    "Beef Kota": {
        "ingredients": {"Bread": 1, "Beef": 200, "Polony": 2, "Atchar": 1, "Onions": 0.5, "Tomatoes": 1, "Chips": 150}
    },
    "Vegetable Kota": {
        "ingredients": {"Bread": 1, "Potatoes": 100, "Carrots": 50, "Onions": 1, "Tomatoes": 1, "Chips": 100}
    }
}

# ============ SAMPLE DATA ============
def get_sample_data():
    np.random.seed(42)
    dates = pd.date_range(start='2025-11-01', end='2026-01-30', freq='D')
    data = []
    ingredients = ['Bread', 'Cheese', 'Polony', 'Atchar', 'Onions', 'Tomatoes', 'Chips', 'Chicken', 'Beef', 'Potatoes', 'Carrots', 'Sausage']
    for date in dates:
        day_of_week = date.dayofweek
        is_weekend = day_of_week >= 5
        base = 1.5 if is_weekend else 1.0
        for ingredient in ingredients:
            qty = np.random.poisson(15 * base) + np.random.randint(-5, 10)
            data.append({'date': date, 'ingredient': ingredient, 'quantity': max(0, qty), 'day_of_week': day_of_week, 'is_weekend': int(is_weekend)})
    return pd.DataFrame(data)

# ============ WEATHER ============
def get_weather_forecast(days_ahead: int = 7):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude=-26.2041&longitude=28.0473&daily=precipitation_probability&forecast_days={days_ahead}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [1.0 if p < 20 else 0.6 if p < 50 else 0.3 for p in data['daily']['precipitation_probability']]
    except:
        pass
    return [0.8] * days_ahead

# ============ TRAIN FUNCTION ============
def train_model(ingredient_name):
    df = get_sample_data()
    ingredient_data = df[df['ingredient'].str.lower() == ingredient_name.lower()].copy()
    if len(ingredient_data) < 7:
        return None
    prophet_df = ingredient_data[['date', 'quantity']].rename(columns={'date': 'ds', 'quantity': 'y'})
    prophet_df['day_of_week'] = prophet_df['ds'].dt.dayofweek
    prophet_df['is_weekend'] = (prophet_df['day_of_week'] >= 5).astype(int)
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True)
    model.add_regressor('day_of_week')
    model.add_regressor('is_weekend')
    model.fit(prophet_df)
    return model

# ============ PREDICT FUNCTION ============
def predict_demand(ingredient_name, days_ahead=7, weather=None):
    model = train_model(ingredient_name)
    if model is None:
        return None
    future = model.make_future_dataframe(periods=days_ahead)
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['is_weekend'] = (future['day_of_week'] >= 5).astype(int)
    if weather is None:
        weather = get_weather_forecast(days_ahead)
    weather_extended = [0.5] * (len(future) - days_ahead) + weather
    future['weather'] = weather_extended
    forecast = model.predict(future)
    return forecast.tail(days_ahead)

# ============ ENDPOINTS ============
@app.get("/")
def root():
    return {"message": "Kota Restaurant AI API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/menu/")
def get_menu():
    return {"status": "success", "menu": KOTA_MENU}

@app.post("/api/forecast/")
def get_forecast(request: ForecastRequest):
    weather = get_weather_forecast(request.days_ahead)
    predictions = predict_demand(request.item_name, request.days_ahead, weather)
    if predictions is None:
        return {"status": "insufficient_data", "item": request.item_name, "predictions": [], "weekly_total": 0}
    return {"status": "success", "item": request.item_name, "predictions": [{"date": str(row['ds'])[:10], "predicted": round(max(0, row['yhat']), 1)} for _, row in predictions.iterrows()], "weekly_total": round(predictions['yhat'].sum(), 1)}

@app.post("/api/recommend/")
def get_recommendation(request: ForecastRequest):
    weather = get_weather_forecast(7)
    predictions = predict_demand(request.item_name, 7, weather)
    if predictions is None:
        return {"status": "error", "item": request.item_name}
    weekly_demand = predictions['yhat'].sum()
    current = 20
    days_left = (current / (weekly_demand / 7)) if weekly_demand > 0 else 999
    recommended = max(0, weekly_demand * 1.2 - current)
    urgency = "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW"
    return {"status": "success", "item": request.item_name, "current_stock": current, "predicted_weekly": round(weekly_demand, 1), "days_left": round(days_left, 1), "recommendation": round(recommended, 1), "urgency": urgency}

@app.post("/api/dashboard/")
def get_dashboard(request: DashboardRequest):
    weather = get_weather_forecast(7)
    results = []
    for item, stock in zip(request.items, request.inventory):
        predictions = predict_demand(item, 7, weather)
        if predictions is None:
            results.append({"item": item, "current_stock": stock, "urgency": "LOW"})
            continue
        weekly_demand = predictions['yhat'].sum()
        days_left = (stock / (weekly_demand / 7)) if weekly_demand > 0 else 999
        recommended = max(0, weekly_demand * 1.2 - stock)
        urgency = "HIGH" if days_left < 3 else "MEDIUM" if days_left < 7 else "LOW"
        results.append({"item": item, "current_stock": stock, "predicted_weekly": round(weekly_demand, 1), "days_left": round(days_left, 1), "recommendation": round(recommended, 1), "urgency": urgency})
    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    results.sort(key=lambda x: urgency_order.get(x.get("urgency", "LOW"), 3))
    return {"status": "success", "summary": {"total_items": len(results), "high_urgency": sum(1 for r in results if r.get("urgency") == "HIGH")}, "items": results}

# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
