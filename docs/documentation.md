# Restaurant Demand Forecasting API Documentation

## Overview

The Restaurant Demand Forecasting API, "Kota AI: Klerksdorp Edition," is a FastAPI-based service designed to provide demand forecasts and reorder recommendations for restaurant inventory items. It integrates with a Supabase PostgreSQL database for historical sales data, stock levels, and event information. It also leverages the Open-Meteo API for weather data to enhance forecast accuracy. The API powers a web-based dashboard, offering real-time insights into inventory needs.

**Deployed Location:** https://restaurant-demand-forecasting-1.onrender.com
**Dashboard Location:** https://kleinboy100.github.io/Dashboard/

## Architecture

The system comprises four main components:

1. **Supabase Database:** A PostgreSQL database hosted on Supabase, storing:
   - `order_items`: Details of individual items sold in each order.
   - `orders`: Main order records with timestamps.
   - `stock`: Current inventory levels for each item.
   - `stock_transactions`: Log of stock changes.
   - `events`: Records of local events (e.g., festivals, promotions) and their estimated impact on demand.

2. **FastAPI Backend (`main.py`):** The core application logic, deployed on Render. It handles:
   - Database interactions (reading sales, stock, events).
   - Weather data fetching (from Open-Meteo).
   - Demand forecasting using Prophet.
   - Generating reorder recommendations.
   - Serving API endpoints.

3. **Prophet Forecasting Model:** Integrated directly into the `main.py` code. For each request, Prophet models are trained on-the-fly using current and historical data from Supabase, ensuring forecasts are always up-to-date with the latest sales.

4. **Frontend Dashboard (`index.html`):** A static HTML/JS application hosted on GitHub Pages. It communicates with the FastAPI backend to display inventory status, demand forecasts, and reorder recommendations to the user.

**Important Note on Models:** The current `main.py` does not use any `.pkl` files from a `models/` directory. All forecasts are generated dynamically by Prophet, which is retrained with fresh data for each request (or a batch of requests for the dashboard endpoint).

## Database Connection & Data Flow

**Supabase Client Initialization**

The API connects to Supabase using environment variables:

- `SUPABASE_URL`: The URL of your Supabase project.
- `SUPABASE_ANON_KEY`: The public API key for your Supabase project.

These variables are loaded when the FastAPI application starts. A `supabase` client object is created using the `supabase-py` library, enabling all database interactions throughout the application. A small connectivity test is performed on startup to verify the connection.

**Data Fetching and Transformation (`get_sales_from_order_items`)**

The `get_sales_from_order_items(item_name: str, days_back: int = 90)` function is central to acquiring data for forecasting:

1. **Raw Order Items:** It queries the `order_items` table for all records associated with a specific `item_name`.
2. **Timestamp Integration:** It then joins these records with the `orders` table (using `order_id`) to retrieve the `created_at` or `order_date` timestamp for each sale.
3. **Recent History Filter:** Data is filtered to include sales only within the last `days_back` (defaulting to 90 days).
4. **Daily Aggregation:** Finally, it aggregates the `quantity` of items sold per `sale_date`, transforming the data into a time series format required by Prophet:
   - `ds`: Date (e.g., '2025-01-01')
   - `y`: Total quantity sold on that date

## Forecasting Model (Prophet)

The `generate_world_class_forecast(item_name: str, days_ahead: int)` function drives the forecasting logic:

1. **Data Acquisition:** It calls `get_sales_from_order_items` to get the (ds, y) time series for the specified `item_name`. A minimum of 5 daily sales records is required for forecasting; otherwise, a fallback estimate is provided.
2. **Event Impact:** It fetches event data (`event_date`, `impact_score`) from the Supabase `events` table. Dates with events have their sales `impact_score` adjusted (e.g., 1.2 for a boost, 0.8 for a dip); dates without events default to 1.0.
3. **Prophet Model Training:** A new Prophet model is initialized and fitted to the historical sales data (ds, y).
   - **Seasonality:** Configured for `weekly_seasonality=True`. `yearly_seasonality` and `daily_seasonality` are `False`.
   - **Holidays:** Incorporates public holidays for South Africa (ZA).
   - **Regressors:** If there are varying `impact_score` values (due to events), `impact_score` is added as an external regressor.

4. **Future Dates & Weather Impact:**
   - `model.make_future_dataframe` generates future dates for the `days_ahead` period.
   - The `get_klerksdorp_weather` function calls the Open-Meteo API to get precipitation probabilities for the forecast horizon. These probabilities are converted into a multiplicative `impact_score` (e.g., 1.0 for dry, 0.7 for medium rain, 0.4 for heavy rain) and applied to the future dates. Thus, the `impact_score` for future predictions accounts for anticipated weather.

5. **Prediction Generation:** The `model.predict()` method generates raw forecasts (`yhat`) for the future dates.
6. **Final Adjustment & Output:**
   - The `yhat` predictions are multiplied by the `impact_score` (incorporating both events and weather) to get `final_prediction`. Predictions are clamped to be non-negative.
   - The function returns a DataFrame with `ds`, `final_prediction`, `yhat_lower`, and `yhat_upper`.

## Recommendation Logic

The `get_recommendation(request: RecommendationRequest)` and the dashboard's internal logic for recommendations are based on:

1. **Current Stock:**
   - If provided in the request payload, it's used directly.
   - Otherwise, the API attempts to fetch it from the Supabase `stock` table for the `item_name`.

2. **Predicted Weekly Demand:**
   - Calls `generate_world_class_forecast` for 7 days ahead. If forecasting is not possible (insufficient data), it falls back to an average daily sale multiplied by 7.

3. **Calculations:**
   - `days_of_stock_left`: `current_stock` / (`weekly_demand` / 7).
   - `recommended_order`: `max(0, (weekly_demand * 1.5) - current_stock)`. A 1.5x buffer is applied.
   - `urgency`: Categorized as "HIGH" (<3 days stock), "MEDIUM" (<7 days stock), or "LOW" (otherwise).

The response includes these calculated metrics, providing actionable insights for reordering.

## Dashboard Integration

The frontend dashboard (`index.html` on GitHub Pages) interacts primarily with the `/api/dashboard` endpoint:

1. **Request:** The dashboard sends a POST request to `/api/dashboard` with a list of `item_name` and optionally their `current_stock`.
2. **Processing:** The API processes each item:
   - Determines `current_stock` (from request or Supabase).
   - Generates a 7-day forecast.
   - Calculates `weekly_demand`, `days_left`, `recommended_order`, `urgency`, and `status`.

3. **Response:** The API returns a JSON response containing a summary (total items, critical items, total recommended) and a sorted `items` list (by urgency), which the dashboard then renders.

## API Endpoints

| Endpoint | Method | Description | Request Body | Response |
|---|---|---|---|---|
| `/` | GET | Root endpoint, returns a simple status message. | - | ```json { "message": "Kota AI Forecasting API", "status": "online" } ``` |
| `/health` | GET | Provides health status of the API and its connection to Supabase. | - | ```json { "status": "healthy", "location": "Klerksdorp", "database": "connected", // or "error: ", "not configured" "order_items_count": 1234, // Example count "stock_table_exists": true, "recommendation": "Stock table ready" // or "Use current_stock param" } ``` |
| `/api/forecast` | POST | Generates a demand forecast for a single item over a specified number of days. | ```json { "item_name": "Bread", "days_ahead": 7 } ``` | ```json { "item": "Bread", "days_ahead": 7, "weekly_total": 150.5, "forecast": [ { "date": "2026-02-18", "predicted": 25.1, "low_estimate": 20.0, "high_estimate": 30.0 }, // ... more days ] } ``` <br> If insufficient data: ```json { "item": "Bread", "status": "insufficient_data", "message": "Need ≥5 sales records for AI forecast.", "total_sold_to_date_estimate": 900.0, "recommendation": "Use manual estimate or add sales data." } ``` |
| `/api/recommend` | POST | Provides reorder recommendations for a single item based on its current stock and demand forecast. | ```json { "item_name": "Bread", "current_stock": 10 } ``` <br> `current_stock` is optional; if omitted, the API attempts to fetch it from the Supabase `stock` table. | ```json { "item": "Bread", "current_stock": 10, "predicted_weekly_demand": 150.5, "days_of_stock_left": 0.5, "recommended_order": 215.8, "urgency": "HIGH", "reorder_now": true, "estimated_restock_days": 7 } ``` |
| `/api/dashboard` | POST | The primary endpoint for the dashboard, fetching recommendations for multiple items in a single request. | ```json { "items": [ { "item_name": "Bread", "current_stock": 10 }, { "item_name": "Cheese" }, { "item_name": "Polony", "current_stock": 50 } ] } ``` | ```json { "summary": { "total_items": 3, "critical_items": 1, "total_recommended": 300.0, "timestamp": "2026-02-18T11:00:00.000000" }, "items": [ { "item_name": "Bread", "current_stock": 10, "weekly_demand": 150.5, "days_left": 0.5, "recommended_order": 215.8, "urgency": "HIGH", "status": "CRITICAL" }, { "item_name": "Cheese", "current_stock": 25, "weekly_demand": 70.0, "days_left": 2.5, "recommended_order": 80.0, "urgency": "MEDIUM", "status": "LOW" }, { "item_name": "Polony", "current_stock": 50, "weekly_demand": 50.0, "days_left": 7.0, "recommended_order": 25.0, "urgency": "LOW", "status": "OK" } ] } ``` |
| `/reorder-recommendations` | POST | An alias for `/api/dashboard`, provided for backward compatibility if older frontend code is still referencing this path. | Same as `/api/dashboard`. | Same as `/api/dashboard`. |

## Limitations

- Prophet needs at least 5 days of data for each item to generate a forecast. Otherwise, a fallback estimate is used.
- The API currently does not handle complex features like promotions, price changes, or competitor actions.

## Future Plans

- Add support for loading and using `.pkl` models trained in Colab.
- Integrate with inventory management systems for automated reordering.
- Explore more advanced forecasting models (e.g., XGBoost, LSTM) for potentially higher accuracy.

## Contact

For questions, bug reports, or feature requests, please open an issue on the GitHub repository.
