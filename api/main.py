import os
import logging
import asyncio
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from prophet import Prophet
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import requests
from uuid import uuid4

# ... (keep existing logging and FastAPI setup) ...

# Add these Pydantic models
class OrderMealRequest(BaseModel):
    order_id: str  # UUID as string
    meal_name: str
    quantity: int

class MealIngredient(BaseModel):
    id: str
    order_id: str
    meal_name: str
    ingredient_name: str
    quantity_used: float
    unit: str
    created_at: str

# Add these endpoints to your FastAPI app

@app.post("/api/process-meal-order")
async def process_meal_order(order: OrderMealRequest):
    """Process a meal order and update ingredient stock with unit handling"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        # Get meal recipe from database
        recipe_result = supabase.table("meal_recipes") \
                              .select("*") \
                              .eq("meal_name", order.meal_name) \
                              .execute()
        
        if not recipe_result.data:
            raise HTTPException(status_code=404, detail=f"Meal {order.meal_name} not found in recipes")
        
        # Process each ingredient in the recipe
        total_depleted = 0
        for recipe in recipe_result.data:
            ingredient = recipe["ingredient_name"]
            quantity_per_meal = recipe["quantity_per_meal"]
            unit = recipe["unit"]
            
            # Calculate total quantity used
            quantity_used = order.quantity * quantity_per_meal
            
            # Get current stock for the ingredient
            stock_result = supabase.table("ingredient_stock") \
                                .select("current_stock, unit") \
                                .eq("ingredient_name", ingredient) \
                                .execute()
            
            if stock_result.data:
                current_stock = stock_result.data[0]["current_stock"]
                stock_unit = stock_result.data[0]["unit"]
                
                # Handle unit conversions if needed
                if unit != stock_unit:
                    # Add unit conversion logic here if needed
                    # For now, assume units match
                    logger.warning(f"Unit mismatch for {ingredient}: recipe {unit} vs stock {stock_unit}")
                
                # Update current_stock
                new_stock = max(0, current_stock - quantity_used)
                
                supabase.table("ingredient_stock") \
                       .update({"current_stock": new_stock}) \
                       .eq("ingredient_name", ingredient) \
                       .execute()
                
                # Record ingredient usage
                meal_ingredient_data = {
                    "id": str(uuid4()),
                    "order_id": order.order_id,
                    "meal_name": order.meal_name,
                    "ingredient_name": ingredient,
                    "quantity_used": quantity_used,
                    "unit": unit,
                    "created_at": datetime.now().isoformat()
                }
                
                supabase.table("meal_ingredients").insert(meal_ingredient_data).execute()
                
                total_depleted += quantity_used
                logger.info(f"Updated {ingredient}: {current_stock} -> {new_stock} (used {quantity_used} {unit})")
            else:
                logger.warning(f"Ingredient {ingredient} not found in stock")
        
        return {
            "status": "success",
            "order_id": order.order_id,
            "meal_name": order.meal_name,
            "quantity": order.quantity,
            "total_depleted": round(total_depleted, 2)
        }
    
    except Exception as e:
        logger.error(f"Error processing meal order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ingredient-usage/{ingredient_name}")
async def get_ingredient_usage(ingredient_name: str, days_back: int = 7):
    """Get usage history for a specific ingredient"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get ingredient usage
        usage_result = supabase.table("meal_ingredients") \
                             .select("*") \
                             .eq("ingredient_name", ingredient_name) \
                             .gte("created_at", start_date.isoformat()) \
                             .execute()
        
        # Group by date
        usage_by_date = {}
        for record in usage_result.data:
            date = record["created_at"].split("T")[0]
            if date not in usage_by_date:
                usage_by_date[date] = 0
            usage_by_date[date] += record["quantity_used"]
        
        return {
            "ingredient_name": ingredient_name,
            "days_back": days_back,
            "usage_by_date": usage_by_date,
            "total_usage": sum(usage_by_date.values())
        }
    
    except Exception as e:
        logger.error(f"Error getting ingredient usage: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/real-time-stock")
async def get_real_time_stock():
    """Get current stock levels for all ingredients"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        stock_result = supabase.table("ingredient_stock").select("*").execute()
        
        # Get usage for today
        today = datetime.now().strftime("%Y-%m-%d")
        today_usage_result = supabase.table("meal_ingredients") \
                                   .select("*") \
                                   .ilike("created_at", f"{today}%") \
                                   .execute()
        
        # Calculate today's usage per ingredient
        today_usage = {}
        for record in today_usage_result.data:
            ingredient = record["ingredient_name"]
            if ingredient not in today_usage:
                today_usage[ingredient] = 0
            today_usage[ingredient] += record["quantity_used"]
        
        # Combine stock and usage data
        stock_data = []
        for item in stock_result.data:
            stock_data.append({
                "ingredient_name": item["ingredient_name"],
                "current_stock": item["current_stock"],
                "unit": item["unit"],
                "min_stock_level": item["min_stock_level"],
                "max_stock_level": item["max_stock_level"],
                "today_usage": today_usage.get(item["ingredient_name"], 0),
                "remaining_stock": max(0, item["current_stock"] - today_usage.get(item["ingredient_name"], 0))
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "stock_data": stock_data
        }
    
    except Exception as e:
        logger.error(f"Error getting real-time stock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Update your dashboard endpoint to include depleted quantity
@app.post("/api/dashboard")
async def dashboard_data(request: DashboardRequest):
    """Main dashboard endpoint with real-time stock and depleted quantity"""
    try:
        items_data = []
        total_recommended = 0
        critical_count = 0
        
        # Get today's usage for all ingredients
        today = datetime.now().strftime("%Y-%m-%d")
        today_usage_result = supabase.table("meal_ingredients") \
                                   .select("*") \
                                   .ilike("created_at", f"{today}%") \
                                   .execute()
        
        today_usage = {}
        for record in today_usage_result.data:
            ingredient = record["ingredient_name"]
            if ingredient not in today_usage:
                today_usage[ingredient] = 0
            today_usage[ingredient] += record["quantity_used"]
        
        for item in request.items:
            try:
                # 1. FETCH CURRENT STOCK from ingredient_stock
                current_stock = item.current_stock
                if current_stock is None and supabase:
                    try:
                        lookup = item.item_name.strip()
                        stock_result = supabase.table("ingredient_stock") \
                            .select("ingredient_name, current_stock, unit") \
                            .ilike("ingredient_name", lookup) \
                            .limit(1) \
                            .execute()
                        
                        if stock_result.data:
                            current_stock = stock_result.data[0]["current_stock"]
                        else:
                            current_stock = 0
                    except Exception as e:
                        logger.warning(f"Error reading ingredient_stock for {item.item_name}: {str(e)}")
                        current_stock = 0
                
                # 2. GET TODAY'S USAGE
                today_usage_amount = today_usage.get(item.item_name.strip(), 0)
                
                # 3. GENERATE FORECAST
                forecast_df = generate_world_class_forecast(item.item_name, 7)
                weekly_demand = forecast_df["final_prediction"].sum() if forecast_df is not None else 5.0
                
                # 4. CALCULATE METRICS
                daily_demand = weekly_demand / 7
                days_left = (current_stock - today_usage_amount) / daily_demand if daily_demand > 0 else 999
                recommended_order = max(0, (weekly_demand * 1.5) - (current_stock - today_usage_amount))
                
                if days_left < 3:
                    urgency, status = "HIGH", "CRITICAL"
                    critical_count += 1
                elif days_left < 7:
                    urgency, status = "MEDIUM", "LOW"
                else:
                    urgency, status = "LOW", "OK"
                
                total_recommended += recommended_order
                
                items_data.append({
                    "item_name": item.item_name,
                    "current_stock": current_stock,
                    "today_usage": round(today_usage_amount, 2),
                    "weekly_demand": round(weekly_demand, 1),
                    "days_left": round(days_left, 1),
                    "recommended_order": round(recommended_order, 1),
                    "urgency": urgency,
                    "status": status,
                    "action": "REORDER NOW" if urgency == "HIGH" else "Monitor stock"
                })
                
            except Exception as e:
                logger.error(f"Error processing {item.item_name}: {str(e)}")
                items_data.append({"item_name": item.item_name, "current_stock": 0, "today_usage": 0, "weekly_demand": 0, "days_left": 0, "recommended_order": 0, "urgency": "HIGH", "status": "ERROR", "action": "Check data"})
        
        # Sort items by urgency
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
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
