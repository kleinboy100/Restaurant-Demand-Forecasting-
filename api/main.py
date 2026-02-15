# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet
import os
import requests
import numpy as np

app = FastAPI(title="Kota Restaurant AI Forecasting API")

# Debugging output
print(f"Starting server on port: {os.environ.get('PORT', '8000')}")

# ============ DATA MODELS ============
class ForecastRequest(BaseModel):
    restaurant_id: int
    item_name: str
    days_ahead: int = 7

class DashboardRequest(BaseModel):
    restaurant_id: int
    items: List[str]
    inventory: List[float]

# ============ KOTA MENU ============
KOTA_MENU = {
    "Cheese Kota": {
        "ingredients": {"Bread": 1, "Cheese": 2, "Polony": 2, "Atchar": 1, "Onions": 0.5, "Tomatoes": 1, "Chips": 150}
    },
    "Sausage Kota": {
        "ingredients": {"Bread": 1, "Sausage": 2, "Polony": 2, "Atchar": 1, "Onions": 0.5, "Tomatoes": 1, "Chips": 150}
    },
    "Chicken Kota": {
        "ingredients": {"Bread": 1, "Chicken": 200, "Polony": 2, "Atchar": 1, "Onions": 0.5, "Tomatoes": 1, "Chips": 150}
    },
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
