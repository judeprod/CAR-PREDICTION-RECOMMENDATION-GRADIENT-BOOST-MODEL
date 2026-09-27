# Gradient Boost Car Rental Model

A machine learning pipeline that predicts car rental daily rates, classifies vehicles as high or low demand, and recommends the best-value cars by type, fuel, and make. Trained on the [Cornell Car Rental Dataset](https://www.kaggle.com/datasets/kushleshkumar/cornell-car-rental-dataset) and served through a FastAPI backend for integration with a web app.

## Features

- **Price Prediction** — a tuned `GradientBoostingRegressor` predicts a car's daily rental rate (`rate.daily`) from its specs (fuel type, vehicle type, make, model, year, rating, and renter trip history).
- **Demand Classification** — a tuned `GradientBoostingClassifier` predicts whether a car (by make, type, fuel type, rating, and value score) is likely to be high or low demand, based on real renter trip history.
- **Best-Value Recommendations** — ranks vehicle type + fuel type + make combinations by a rating-to-predicted-price "value score," and exposes it as a filterable recommendation endpoint.
- **REST API** — a FastAPI service with `/predict`, `/predict-demand`, and `/recommend` endpoints, ready to be called from a Node/Express or any other backend.

## Tech Stack

- **Python 3.11**
- **pandas / numpy** — data loading and preprocessing
- **scikit-learn** — `GradientBoostingRegressor`, `GradientBoostingClassifier`, `LabelEncoder`, `RandomizedSearchCV`
- **FastAPI + Uvicorn** — API server
- **joblib** — model persistence

## Dataset

The Cornell Car Rental Dataset (~5,850 listings) includes vehicle specs, fuel type, rating, renter trip counts, and daily rental price. For this project, location columns (city, country, state, latitude, longitude), `reviewCount`, and `owner.id` were dropped to keep the models focused on vehicle attributes rather than geography or listing metadata.

## Project Structure

```
├── CarRentalModel.ipynb       # Data cleaning, feature engineering, model training & tuning
├── main.py                    # FastAPI app serving /predict, /predict-demand, and /recommend
├── model.pkl                  # Trained GradientBoostingRegressor (price)
├── encoders.pkl               # LabelEncoders for high-cardinality features
├── feature_order.pkl          # Feature column order expected by the price model
├── demand_classifier.pkl      # Trained GradientBoostingClassifier (demand)
├── demand_feature_order.pkl   # Feature column order expected by the demand classifier
├── df_results.pkl             # Precomputed predictions + value scores for recommendations
├── carrentaldata.csv          # Raw dataset (from Kaggle)
└── README.md
```

## Model Performance

**Price Predictor (Regression)**

| Metric | Value |
|---|---|
| MAE | ~$30.13 |
| R² | ~0.42 |

**Demand Classifier**

| Metric | Value |
|---|---|
| Accuracy | ~0.79 |
| Precision (High demand) | 0.84 |
| Recall (High demand) | 0.72 |

Both models tuned via `RandomizedSearchCV` over `n_estimators`, `max_depth`, `learning_rate`, and related parameters.

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install pandas numpy scikit-learn joblib fastapi "uvicorn[standard]"
   ```
2. Download `carrentaldata.csv` from [Kaggle](https://www.kaggle.com/datasets/kushleshkumar/cornell-car-rental-dataset) and place it in the project root.
3. Run `CarRentalModel.ipynb` top to bottom to train both models and generate the `.pkl` files.
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

### `POST /predict-demand`
Predicts whether a car is likely to be high or low demand, using the same request body shape as `/predict`.

**Response:**
```json
{
  "demand": "High",
  "confidence": 0.812,
  "predicted_price": 67.42
}
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
- [ ] Add `location.state` back as a feature to improve price prediction accuracy
- [ ] Explore further demand classifier improvements (currently ~0.79 accuracy)
- [ ] Wire `/recommend` and `/predict-demand` into the web app's frontend UI (badges, sort order, host guidance)
