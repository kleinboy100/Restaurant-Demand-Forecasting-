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
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))se "MEDIUM" if days_left < 7 else "LOW"
        results.append({"item": item, "current_stock": stock, "predicted_weekly": round(weekly_demand, 1), "days_left": round(days_left, 1), "recommendation": round(recommended, 1), "urgency": urgency})
    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    results.sort(key=lambda x: urgency_order.get(x.get("urgency", "LOW"), 3))
    return {"status": "success", "summary": {"total_items": len(results), "high_urgency": sum(1 for r in results if r.get("urgency") == "HIGH")}, "items": results}

# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
