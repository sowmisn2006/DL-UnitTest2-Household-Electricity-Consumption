# DL UnitTest2 
# Household Electricity Consumption Forecasting

## Problem Statement
An electricity provider wants to forecast household electricity consumption for energy distribution. This project uses 10,000 consecutive valid minute-level records from the Kaggle/UCI **Household Electric Power Consumption** dataset to predict future minute-by-minute energy usage. 

To solve this sequence-to-prediction problem, the code utilizes a **Long Short-Term Memory (LSTM)** deep learning model, which is highly effective at capturing long-term dependencies and mitigating the vanishing gradient problem inherent in time-series forecasting.

## Code Overview
The provided Python script executes the forecasting pipeline in five distinct tasks:

* **Task 1: Preprocessing & Normalization** 
  Loads the `household_power_consumption.txt` dataset, merges the Date and Time columns into a datetime index, and replaces missing `?` values using forward-fill (`ffill`). It extracts the first 10,000 records of `Global_active_power` and normalizes the data between 0 and 1 using Scikit-Learn's `MinMaxScaler`.
* **Task 2: Sequence Generation** 
  Transforms the linear data into sliding window sequences. It takes 60 minutes of historical data (the "lookback" window) to predict the power consumption of the 61st minute. The data is reshaped into a 3D array `[samples, time steps, features]` required by Keras LSTMs and split into an 80% training set and a 20% testing set.
* **Task 3: Model Training** 
  Builds a Keras `Sequential` model featuring an LSTM layer with 50 internal units, followed by a standard `Dense` output layer. It compiles using the Adam optimizer and Mean Squared Error (MSE) loss, and trains for 10 epochs.
* **Task 4: Evaluation (MAE & RMSE)** 
  Generates predictions on the 20% unseen test data. It uses the `inverse_transform` function of the scaler to convert both the predictions and actual answers back to real-world Kilowatts (kW), then calculates the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* **Task 5: Visualization** 
  Uses Matplotlib to plot the model's predicted consumption (dashed red line) against the actual consumption (solid blue line) for the first 300 minutes of the test set, providing a clear visual representation of the model's tracking accuracy.

## Prerequisites
To run this code, you will need the dataset extracted as `household_power_consumption.txt` in the same directory as the script, along with the following Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow