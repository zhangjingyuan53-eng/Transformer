import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib


# Load the input data
X_data = np.load()
Y_data = np.load()  # label data

# data normalization
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_data = scaler_x.fit_transform(X_data)
y_data_scaled = scaler_y.fit_transform(Y_data)
print(f"X: {X_data.shape}, Y: {Y_data.shape}")

# data partitioning
x_temp, x_test, y_temp, y_test, time_temp, time_test = train_test_split(
    X_data, y_data_scaled, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val, time_train, time_val = train_test_split(
    x_temp, y_temp, time_temp, test_size=0.25)
print(f"train: {x_train.shape}, Validation: {x_val.shape}, test: {x_test.shape}")


# Random forest model training
rf_params = {
    'n_estimators': 300,
    'max_depth': 38,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'n_jobs': -1  # CPU multi-threading acceleration
}

model = RandomForestRegressor(**rf_params)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()
print(f"all_time: {end_time - start_time:.2f} s")

# Model saving
rf_model_path = r'\rf.pkl'
joblib.dump(model, rf_model_path)

# Validation set evaluation
val_preds = model.predict(x_val)
y_val_denorm = scaler_y.inverse_transform(y_val)
val_preds_denorm = scaler_y.inverse_transform(val_preds)

print("\nValidation set evaluation metrics：")
val_rmse_list = []
for i in range(y_val.shape[1]):
    rmse = np.sqrt(mean_squared_error(y_val_denorm[:, i], val_preds_denorm[:, i]))
    val_rmse_list.append(rmse)
    print(f"variable{i+1} RMSE: {rmse:.4f}")

# Test set evaluation
test_preds = model.predict(x_test)
y_test_denorm = scaler_y.inverse_transform(y_test)
test_preds_denorm = scaler_y.inverse_transform(test_preds)

print("\nTest set evaluation metrics：")
test_rmse_list = []
for i in range(y_test.shape[1]):
    rmse = np.sqrt(mean_squared_error(y_test_denorm[:, i], test_preds_denorm[:, i]))
    test_rmse_list.append(rmse)
    print(f"variable{i+1} RMSE: {rmse:.4f}")

