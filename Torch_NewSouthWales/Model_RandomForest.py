import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# ------------------------------
# 1. Load all customers' daily data
# ------------------------------
# （与 XGBoost 版本完全相同，可直接复用）
def load_daily_data(file_path='data_daily_long.csv'):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values(['Customer', 'Consumption Category', 'date']).reset_index(drop=True)
    
    print(f"Unique Customers: {df['Customer'].nunique()}")
    print(f"Unique Postcodes: {df['Postcode'].nunique()}")
    print(f"Unique Categories: {df['Consumption Category'].unique()}")
    
    return df

# ------------------------------
# 2. Prepare data: add time features, lags, encoding, scaling
# ------------------------------
# （与 XGBoost 版本完全相同，可直接复用）
def prepare_data(df, lag_window=7, test_ratio=0.2):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    df['group'] = df['Customer'].astype(str) + '_' + df['Consumption Category']
    groups = df.groupby('group')
    
    for lag in range(1, lag_window + 1):
        df[f'daily_total_kWh_lag_{lag}'] = groups['daily_total_kWh'].shift(lag)
    
    df = df.drop(columns=['group'])
    df = df.dropna().reset_index(drop=True)
    
    # Encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(df[['Consumption Category']])
    cat_cols = ohe.get_feature_names_out(['Consumption Category'])
    df_cat = pd.DataFrame(cat_encoded, columns=cat_cols)
    
    le_customer = LabelEncoder()
    le_postcode = LabelEncoder()
    df['Customer_encoded'] = le_customer.fit_transform(df['Customer'])
    df['Postcode_encoded'] = le_postcode.fit_transform(df['Postcode'])
    
    feature_cols = [
        'Generator Capacity', 'Customer_encoded', 'Postcode_encoded',
        'year', 'month', 'day', 'weekday', 'is_weekend'
    ] + list(df_cat.columns) + [col for col in df.columns if 'lag_' in col]
    
    X = pd.concat([df[feature_cols[:8]], df_cat, df[[col for col in df.columns if 'lag_' in col]]], axis=1)
    y = df['daily_total_kWh']
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    train_size = int(len(X) * (1 - test_ratio))
    X_train, X_test = X_scaled.iloc[:train_size], X_scaled.iloc[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    y_test_orig = y.iloc[train_size:]
    
    return X_train, X_test, y_train, y_test, y_test_orig, scaler_X, scaler_y, feature_cols

# ------------------------------
# 3. Train Random Forest model
# ------------------------------
def train_randomforest(X_train, y_train, params=None):
    if params is None:
        params = {
            'n_estimators': 200,           # 树的数量
            'max_depth': 12,               # 最大深度（比XGBoost通常浅一些）
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',        # 或 'log2'、0.8 等
            'random_state': 42,
            'n_jobs': -1                   # 使用所有核心
        }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    return model

# ------------------------------
# 4. Rolling forecast & evaluate (几乎与XGBoost版本相同)
# ------------------------------
def rolling_forecast_and_evaluate(model, X_test, y_test, scaler_y, forecast_horizon=1):
    predictions_scaled = []
    n_windows = len(X_test) - forecast_horizon + 1
    
    # RandomForest 支持批量预测，不需要像XGBoost那样用DMatrix
    for i in tqdm(range(0, n_windows, forecast_horizon)):
        steps = min(forecast_horizon, len(X_test) - i)
        pred = model.predict(X_test.iloc[i:i + steps])
        predictions_scaled.extend(pred)
    
    # 反归一化 & 截断负值
    predictions = scaler_y.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
    predictions = np.maximum(predictions, 0)
    
    actuals = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    epsilon = np.mean(actuals) * 0.1 if np.mean(actuals) > 0 else 1e-6
    mape_safe = np.mean(np.abs((actuals - predictions) / (np.abs(actuals) + epsilon))) * 100
    
    print(f"Evaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE_SAFE={mape_safe:.2f}%")
    
    print(f"Actual stats:   min={actuals.min():.3f}, max={actuals.max():.3f}, mean={actuals.mean():.3f}")
    print(f"Predicted stats: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")
    
    return mae, rmse, mape_safe, predictions, actuals

# ------------------------------
# Main program
# ------------------------------
if __name__ == "__main__":
    LAG_WINDOW = 7
    TEST_RATIO = 0.2
    
    # Load & Prepare
    df = load_daily_data()
    X_train, X_test, y_train, y_test, y_test_orig, scaler_X, scaler_y, feature_cols = prepare_data(df, LAG_WINDOW, TEST_RATIO)
    print("Data preparation success")
    print("Features:", feature_cols)
    
    # Train RandomForest
    print("Start training Random Forest...")
    model = train_randomforest(X_train, y_train)
    print("Model training success")
    
    # Evaluate
    print("\nRolling forecast evaluation:")
    rolling_forecast_and_evaluate(model, X_test, y_test, scaler_y, forecast_horizon=1)