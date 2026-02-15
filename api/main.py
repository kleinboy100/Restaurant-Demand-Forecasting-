from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pickle
import pandas as pd
from datetime import datetime, timedelta
import os

app = FastAPI(title="Kota Restaurant Forecasting API")

# ============ MENU DATA ============
KOTA_MENU = {
    "Cheese Kota": {
        "ingredients": {
            "Bread": 1,  # 1 loaf
            "Cheese": 2,  # 2 slices
            "Polony": 2,
            "Atchar": 1,
            "Onions": 0.5,
            "Tomatoes": 1,
            "Chips": 150  # grams
        }
    },
    "Sausage Kota": {
        "ingredients": {
            "Bread": 1,
            "Sausage": 2,
            "Polony": 2,
            "Atchar": 1,
            "Onions": 0.5,
            "Tomatoes": 1,
            "Chips": 150
        }
    },
    "Chicken Kota": {
        "ingredients": {
            "Bread": 1,
            "Chicken": 200,  # grams
            "Polony": 2,
            "Atchar": 1,
            "Onions": 0.5,
            "Tomatoes": 1,
            "Chips": 150
        }
    },
    "Beef Kota": {
        "ingredients": {
            "Bread": 1,
            "Beef": 200,  # grams
            "Polony": 2,
            "Atchar": 1,
            "Onions": 0.5,
            "Tomatoes": 1,
            "Chips": 150
        }
    },
    "Vegetable Kota": {
        "ingredients": {
            "Bread": 1,
            "Potatoes": 100,
            "Carrots": 50,
            "Onions": 1,
            "Tomatoes": 1,
            "Chips": 100
        }
    }
}

# ============ LOAD TRAINED MODELS ============
MODELS = {}
MANIFEST = None

try:
    # Load manifest first to know which models to load
    with open('models/manifest.pkl', 'rb') as f:
        MANIFEST = pickle.load(f)
    
    # Load each ingredient model
    for ingredient in MANIFEST['ingredients']:
        model_path = f'models/{ingredient}_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                MODELS[ingredient] = pickle.load(f)
        else:
            print(f"⚠️ Model not found: {model_path}")
except Exception as e:
    print(f"Warning: Could not load models - {e}")

# ============ DATA MODELS ============
class ForecastRequest(BaseModel):
    ingredient: str
    days_ahead: int = 7
    weather_forecast: List[float] = None  # [weather_score for each day]

class InventoryRequest(BaseModel):
    ingredient: str
    current_stock: float

class DashboardRequest(BaseModel):
    ingredients: List[str]
    current_stocks: List[float]

# ============ API ENDPOINTS ============
@app.get("/")
def root():
    return {"message": "Kota Restaurant Forecasting API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/api/forecast/")
def get_forecast(request: ForecastRequest):
    """Get demand forecast for a specific ingredient"""
    if request.ingredient not in MODELS:
        raise HTTPException(status_code=404, detail="Ingredient not found")
    
    model = MODELS[request.ingredient]
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=request.days_ahead)
    
    # Add regressors for future days
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['is_weekend'] = (future['day_of_week'] >= 5).astype(int)
    
    # Weather forecast (default sunny)
    weather_forecast = request.weather_forecast or [1.0] * request.days_ahead
    weather_extended = [0.5] * (len(future) - request.days_ahead) + weather_forecast
    future['weather_score'] = weather_extended
    
    # Make predictions
    forecast = model.predict(future)
    
    # Prepare response
    predictions = forecast.tail(request.days_ahead)
    
    return {
        "status": "success",
        "ingredient": request.ingredient,
        "predictions": [
            {
                "date": row['ds'].strftime('%Y-%m-%d'),
                "predicted": round(row['yhat'], 1),
                "min": round(row['yhat_lower'], 1),
                "max": round(row['yhat_upper'], 1)
            }
            for _, row in predictions.iterrows()
        ],
        "weekly_total": round(predictions['yhat'].sum(), 1)
    }

@app.post("/api/recommend/")
def get_recommendation(request: InventoryRequest):
    """Get reorder recommendation for an ingredient"""
    forecast_req = ForecastRequest(
        ingredient=request.ingredient,
        days_ahead=7,
        weather_forecast=[1.0] * 7  # Default sunny
    )
    forecast = get_forecast(forecast_req)
    
    if forecast.get("status") != "success":
        return forecast
    
    weekly_demand = forecast["weekly_total"]
    current = request.current_stock
    
    # Calculate days of stock left
    days_left = (current / (weekly_demand / 7)) if weekly_demand > 0 else 999
    
    # Recommended order (weekly demand + 20% buffer - current stock)
    recommended = max(0, weekly_demand * 1.2 - current)
    
    # Urgency
    if days_left < 3:
        urgency = "HIGH"
        action = "Order immediately"
    elif days_left < 7:
        urgency = "MEDIUM" 
        action = "Order within 2 days"
    else:
        urgency = "LOW"
        action = "Stock OK"
    
    return {
        "status": "success",
        "ingredient": request.ingredient,
        "current_stock": current,
        "predicted_weekly": weekly_demand,
        "days_left": round(days_left, 1),
        "recommendation": round(recommended, 1),
        "urgency": urgency,
        "action": action
    }

@app.post("/api/dashboard/")
def get_dashboard(request: DashboardRequest):
    """Get all recommendations for dashboard"""
    results = []
    
    for ingredient, stock in zip(request.ingredients, request.current_stocks):
        req = InventoryRequest(
            ingredient=ingredient,
            current_stock=stock
        )
        result = get_recommendation(req)
        if result.get("status") == "success":
            results.append(result)
    
    # Sort by urgency
    urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    results.sort(key=lambda x: urgency_order.get(x.get("urgency", "LOW"), 3))
    
    return {
        "status": "success",
        "summary": {
            "total_items": len(results),
            "high_urgency": sum(1 for r in results if r.get("urgency") == "HIGH"),
            "medium_urgency": sum(1 for r in results if r.get("urgency") == "MEDIUM"),
        },
        "items": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
