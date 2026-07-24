import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/city_day.csv")

# Keep required columns
columns = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI"]
df = df[columns]

# Remove missing values
df = df.dropna()

# Features and target
X = df.drop("AQI", axis=1)
y = df["AQI"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("R2 Score:", r2)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/aqi_model.pkl")

print("Model saved successfully!")