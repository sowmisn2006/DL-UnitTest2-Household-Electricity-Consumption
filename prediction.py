import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the saved model and scaler into memory FIRST
print("Loading model and scaler...")
model = load_model('electricity_lstm_model.keras')
scaler = joblib.load('electricity_scaler.gz')

# 1. The Input: 60 consecutive minute-level power readings (in kW).
raw_input_data = [
    1.45, 1.43, 1.42, 1.45, 1.48, 1.50, 1.51, 1.49, 1.48, 1.45,
    1.44, 1.42, 1.41, 1.39, 1.40, 1.42, 1.45, 1.55, 1.60, 1.75,
    2.10, 2.45, 2.80, 3.10, 3.25, 3.30, 3.28, 3.20, 3.15, 2.90,
    2.50, 2.20, 1.95, 1.80, 1.65, 1.55, 1.50, 1.48, 1.45, 1.44,
    1.42, 1.40, 1.38, 1.35, 1.33, 1.32, 1.30, 1.32, 1.34, 1.35,
    1.38, 1.40, 1.42, 1.45, 1.46, 1.48, 1.50, 1.52, 1.55, 1.58
]

# 2. Convert the python list into a 2D NumPy array (60 rows, 1 column)
recent_60_minutes = np.array(raw_input_data).reshape(-1, 1)

# 3. Scale the input using the loaded scaler
recent_60_scaled = scaler.transform(recent_60_minutes)

# 4. Reshape into 3D for the LSTM: (1 sample, 60 time steps, 1 feature)
X_new_prediction = np.reshape(recent_60_scaled, (1, 60, 1))

# 5. Predict the scaled future value
future_prediction_scaled = model.predict(X_new_prediction)

# 6. Inverse transform to get the actual kW value
future_prediction_kW = scaler.inverse_transform(future_prediction_scaled)

# 7. Print the result
print("-" * 50)
print(f"Predicted power consumption for the NEXT minute: {future_prediction_kW[0][0]:.4f} kW")
print("-" * 50)