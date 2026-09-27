from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd

app = FastAPI()

# Load price model artifacts
model = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')
feature_cols = joblib.load('feature_order.pkl')

# Load demand classifier artifacts
clf = joblib.load('demand_classifier.pkl')
feature_cols_clf = joblib.load('demand_feature_order.pkl')

# Load precomputed recommendation data
df_results = pd.read_pickle('df_results.pkl')


class CarInput(BaseModel):
    rating: float
    renterTripsTaken: int
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    fuelType: str
    vehicle_type: str


def build_price_row(car: CarInput):
    row = {
        'rating': car.rating,
        'renterTripsTaken': car.renterTripsTaken,
        'vehicle.make': encoders['vehicle.make'].transform([car.vehicle_make])[0]
            if car.vehicle_make in encoders['vehicle.make'].classes_ else -1,
        'vehicle.model': encoders['vehicle.model'].transform([car.vehicle_model])[0]
            if car.vehicle_model in encoders['vehicle.model'].classes_ else -1,
        'vehicle.year': car.vehicle_year,
    }
    for col in feature_cols:
        if col.startswith('fuel_') or col.startswith('type_'):
            row[col] = 0
    fuel_col = f"fuel_{car.fuelType}"
    type_col = f"type_{car.vehicle_type}"
    if fuel_col in feature_cols:
        row[fuel_col] = 1
    if type_col in feature_cols:
        row[type_col] = 1
    return row


@app.post("/predict")
def predict_price(car: CarInput):
    row = build_price_row(car)
    input_df = pd.DataFrame([row])[feature_cols]
    prediction = float(model.predict(input_df)[0])
    return {"predicted_daily_rate": round(prediction, 2)}


@app.get("/recommend")
def recommend(
    vehicle_type: Optional[str] = None,
    fuel_type: Optional[str] = None,
    max_budget: Optional[float] = None,
    top_n: int = 5
):
    filtered = df_results.copy()

    if vehicle_type:
        filtered = filtered[filtered['vehicle_type_raw'].str.lower() == vehicle_type.lower()]
    if fuel_type:
        filtered = filtered[filtered['fuelType_raw'].str.upper() == fuel_type.upper()]
    if max_budget:
        filtered = filtered[filtered['predicted_price'] <= max_budget]

    if filtered.empty:
        return {"message": "No cars match those criteria — try a higher budget or fewer filters."}

    result = filtered.sort_values('value_score', ascending=False).head(top_n)
    return result[['vehicle_type_raw', 'fuelType_raw', 'vehicle_make_raw', 'predicted_price', 'actual_rating']].to_dict(orient='records')


@app.post("/predict-demand")
def predict_demand(car: CarInput):
    # Step 1: get predicted price from the price model
    price_row = build_price_row(car)
    predicted_price = float(model.predict(pd.DataFrame([price_row])[feature_cols])[0])
    value_score = car.rating / predicted_price

    # Step 2: build input for the demand classifier
    demand_row = {col: 0 for col in feature_cols_clf}
    demand_row['vehicle.make'] = price_row['vehicle.make']
    demand_row['value_score'] = value_score
    demand_row['rating'] = car.rating

    fuel_col = f"fuel_{car.fuelType}"
    type_col = f"type_{car.vehicle_type}"
    if fuel_col in feature_cols_clf:
        demand_row[fuel_col] = 1
    if type_col in feature_cols_clf:
        demand_row[type_col] = 1

    input_df = pd.DataFrame([demand_row])[feature_cols_clf]
    prediction = clf.predict(input_df)[0]
    probability = clf.predict_proba(input_df)[0][1]

    return {
        "demand": "High" if prediction == 1 else "Low",
        "confidence": round(float(probability), 3),
        "predicted_price": round(predicted_price, 2)
    }