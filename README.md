# Gradient Boost Car Rental Model

A machine learning pipeline that predicts car rental daily rates and recommends the best-value vehicles by type, fuel, and make. Trained on the [Cornell Car Rental Dataset](https://www.kaggle.com/datasets/kushleshkumar/cornell-car-rental-dataset) and served through a FastAPI backend for integration with a web app.

## Features

- **Price Prediction** — a tuned `GradientBoostingRegressor` predicts a car's daily rental rate (`rate.daily`) from its specs (fuel type, vehicle type, make, model, year, rating, and renter trip history).
- **Best-Value Recommendations** — ranks vehicle type + fuel type + make combinations by a rating-to-predicted-price "value score," and exposes it as a filterable recommendation endpoint.
- **REST API** — a FastAPI service with `/predict` and `/recommend` endpoints, ready to be called from a Node/Express or any other backend.

## Tech Stack

- **Python 3.11**
- **pandas / numpy** — data loading and preprocessing
- **scikit-learn** — `GradientBoostingRegressor`, `LabelEncoder`, `GridSearchCV`
- **FastAPI + Uvicorn** — API server
- **joblib** — model persistence

## Dataset

The Cornell Car Rental Dataset (~5,850 listings) includes vehicle specs, fuel type, rating, renter trip counts, and daily rental price. For this project, location columns (city, country, state, latitude, longitude), `reviewCount`, and `owner.id` were dropped to keep the model focused on vehicle attributes rather than geography or listing metadata.

## Project Structure

```
├── CarRentalModel.ipynb   # Data cleaning, feature engineering, model training & tuning
├── main.py                # FastAPI app serving /predict and /recommend
├── model.pkl              # Trained GradientBoostingRegressor
├── encoders.pkl           # LabelEncoders for high-cardinality features
├── feature_order.pkl      # Exact feature column order expected by the model
├── df_results.pkl         # Precomputed predictions + value scores for recommendations
├── carrentaldata.csv      # Raw dataset (from Kaggle)
└── README.md
```

## Model Performance

| Metric | Value |
|---|---|
| MAE | ~$30.13 |
| R² | ~0.42 |

Tuned via `GridSearchCV` over `n_estimators`, `max_depth`, and `learning_rate`.

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install pandas numpy scikit-learn joblib fastapi "uvicorn[standard]"
   ```
2. Download `carrentaldata.csv` from [Kaggle](https://www.kaggle.com/datasets/kushleshkumar/cornell-car-rental-dataset) and place it in the project root.
3. Run `CarRentalModel.ipynb` top to bottom to train the model and generate the `.pkl` files.
4. Start the API:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```
5. Open `http://localhost:8000/docs` to test the endpoints interactively.

## API Endpoints

### `POST /predict`
Predicts the daily rental rate for a given car.

**Request body:**
```json
{
  "rating": 4.8,
  "renterTripsTaken": 12,
  "vehicle_make": "Toyota",
  "vehicle_model": "Camry",
  "vehicle_year": 2018,
  "fuelType": "GASOLINE",
  "vehicle_type": "car"
}
```

**Response:**
```json
{ "predicted_daily_rate": 67.42 }
```

### `GET /recommend`
Returns the top-ranked cars matching optional filters, sorted by value score (rating relative to predicted price).

**Query parameters:** `vehicle_type`, `fuel_type`, `max_budget`, `top_n` (default 5)

**Example:**
```
GET /recommend?vehicle_type=suv&fuel_type=ELECTRIC&max_budget=100&top_n=5
```

## Roadmap

- [ ] Persist car listings and prediction/recommendation logs in a database
- [ ] Add `location.state` back as a feature to improve prediction accuracy
- [ ] Wire `/recommend` into the web app's frontend search/filter UI
