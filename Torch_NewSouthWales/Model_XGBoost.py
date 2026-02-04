import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import xgboost as xgb
from tqdm import tqdm

# ------------------------------
# 1. Load all customers' daily data
# ------------------------------
def load_daily_data(file_path='data_daily_long.csv'):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values(['Customer', 'Consumption Category', 'date']).reset_index(drop=True)
    
    # Check unique values for encoding decisions
    print(f"Unique Customers: {df['Customer'].nunique()}")
    print(f"Unique Postcodes: {df['Postcode'].nunique()}")
    print(f"Unique Categories: {df['Consumption Category'].unique()}")
    
    return df

# ------------------------------
# 2. Prepare data: add time features, lags, encoding, scaling
# ------------------------------
def prepare_data(df, lag_window=7, test_ratio=0.2):  # lag_window in days
    # Add time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # Add lag features for daily_total_kWh (per Customer + Category)
    df['group'] = df['Customer'].astype(str) + '_' + df['Consumption Category']  # Temp group key
    groups = df.groupby('group')
    
    for lag in range(1, lag_window + 1):
        df[f'daily_total_kWh_lag_{lag}'] = groups['daily_total_kWh'].shift(lag)
    
    df = df.drop(columns=['group'])
    df = df.dropna().reset_index(drop=True)  # Drop NaN from shifts
    
    # Encoding categoricals
    # OneHot for Consumption Category (low cardinality)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(df[['Consumption Category']])
    cat_cols = ohe.get_feature_names_out(['Consumption Category'])
    df_cat = pd.DataFrame(cat_encoded, columns=cat_cols)
    
    # LabelEncode for Customer and Postcode (potentially high cardinality)
    le_customer = LabelEncoder()
    le_postcode = LabelEncoder()
    df['Customer_encoded'] = le_customer.fit_transform(df['Customer'])
    df['Postcode_encoded'] = le_postcode.fit_transform(df['Postcode'])
    
    # Features: numeric + encoded + time feats + lags
    feature_cols = [
        'Generator Capacity', 'Customer_encoded', 'Postcode_encoded',
        'year', 'month', 'day', 'weekday', 'is_weekend'
    ] + list(df_cat.columns) + [col for col in df.columns if 'lag_' in col]
    
    X = pd.concat([df[feature_cols[:8]], df_cat, df[[col for col in df.columns if 'lag_' in col]]], axis=1)
    y = df['daily_total_kWh']
    
    # Scaling features and target (StandardScaler for trees optional, but for consistency)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Time-based split (global, assuming dates are sorted)
    train_size = int(len(X) * (1 - test_ratio))
    X_train, X_test = X_scaled.iloc[:train_size], X_scaled.iloc[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    y_test_orig = y.iloc[train_size:]  # For evaluation
    
    return X_train, X_test, y_train, y_test, y_test_orig, scaler_X, scaler_y, feature_cols

# ------------------------------
# 3. Train XGBoost model
# ------------------------------
def train_xgboost(X_train, y_train, params=None):
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42,
            # 'tree_method': 'gpu_hist'  # Uncomment for GPU
        }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=1000)
    
    return model

# ------------------------------
# 4. Rolling forecast & evaluate (with inverse scaling)
# ------------------------------
def rolling_forecast_and_evaluate(model, X_test, y_test, scaler_y, forecast_horizon=1):
    predictions_scaled = []
    n_windows = len(X_test) - forecast_horizon + 1
    
    for i in tqdm(range(0, n_windows, forecast_horizon)):
        steps = min(forecast_horizon, len(X_test) - i)
        dtest = xgb.DMatrix(X_test.iloc[i:i + steps])
        pred = model.predict(dtest).flatten()[:steps]
        predictions_scaled.extend(pred)
    
    # Inverse scale predictions and clip to non-negative
    predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    predictions = np.maximum(predictions, 0)
    
    actuals = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Evaluation (adapted from original)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    #mape = np.mean(np.abs((actuals - predictions) / np.abs(actuals))) * 100
    
    epsilon = np.mean(actuals) * 0.1
    mape_safe = np.mean(np.abs((actuals - predictions) / (np.abs(actuals) + epsilon))) * 100
    
    #print(f"Evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, MAPE_SAFE={mape_safe:.2f}%")
    print(f"Evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE_SAFE={mape_safe:.2f}%")
    
    # Additional stats
    print(f"Actual stats: min={actuals.min():.3f}, max={actuals.max():.3f}, mean={actuals.mean():.3f}")
    print(f"Predicted stats: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")
    
    #return mae, rmse, mape, mape_safe, predictions, actuals
    return mae, rmse, mape_safe, predictions, actuals

# ------------------------------
# Main program
# ------------------------------
if __name__ == "__main__":
    LAG_WINDOW = 7  # Past 7 days lags
    TEST_RATIO = 0.2
    
    # Load
    df = load_daily_data()
    
    # Prepare
    X_train, X_test, y_train, y_test, y_test_orig, scaler_X, scaler_y, feature_cols = prepare_data(df, LAG_WINDOW, TEST_RATIO)
    print("Data preparation success")
    print("Features:", feature_cols)
    
    # Train
    model = train_xgboost(X_train, y_train)
    print("Model training success")
    
    # Evaluate with rolling forecast
    rolling_forecast_and_evaluate(model, X_test, y_test, scaler_y)