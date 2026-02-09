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
def prepare_data(df, lag_window=7, test_ratio=0.2):
    """
    按每个客户独立切分 train/test，避免数据泄漏
    """
    # 1. 时间特征
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # 2. 滞后特征（按 group 计算）
    df['group'] = df['Customer'].astype(str) + '_' + df['Consumption Category']
    groups = df.groupby('group')
    for lag in range(1, lag_window + 1):
        df[f'daily_total_kWh_lag_{lag}'] = groups['daily_total_kWh'].shift(lag)
    df = df.drop(columns=['group'])

    # 3. 按客户切分
    train_dfs, test_dfs = [], []
    for cust, sub in df.groupby('Customer'):
        sub = sub.sort_values('date').reset_index(drop=True)
        if len(sub) < lag_window + 10:
            train_dfs.append(sub)
            continue
        cut_idx = max(lag_window + 1, int(len(sub) * (1 - test_ratio)))
        train_dfs.append(sub.iloc[:cut_idx].copy())
        test_dfs.append(sub.iloc[cut_idx:].copy())

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test  = pd.concat(test_dfs, ignore_index=True)

    # 4. 删除 lag 导致的 NaN（只在训练集处理）
    df_train = df_train.dropna().reset_index(drop=True)

    # 5. One-hot 编码（推荐：fit 在 train 上，避免微小泄漏）
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(df_train[['Consumption Category']])   # 只 fit train

    train_cat = pd.DataFrame(
        ohe.transform(df_train[['Consumption Category']]),
        columns=ohe.get_feature_names_out(),
        index=df_train.index
    )
    test_cat = pd.DataFrame(
        ohe.transform(df_test[['Consumption Category']]),
        columns=ohe.get_feature_names_out(),
        index=df_test.index
    )

    # 关键：合并回 df_train 和 df_test
    df_train = pd.concat([df_train, train_cat], axis=1)
    df_test  = pd.concat([df_test,  test_cat],  axis=1)

    # 6. Label Encoding（Customer 和 Postcode）
    le_customer = LabelEncoder()
    le_postcode = LabelEncoder()
    # fit on train + test 合并后的数据（树模型常见做法）
    combined = pd.concat([df_train['Customer'], df_test['Customer']])
    le_customer.fit(combined)
    combined_post = pd.concat([df_train['Postcode'], df_test['Postcode']])
    le_postcode.fit(combined_post)

    df_train['Customer_encoded'] = le_customer.transform(df_train['Customer'])
    df_train['Postcode_encoded'] = le_postcode.transform(df_train['Postcode'])
    df_test['Customer_encoded']  = le_customer.transform(df_test['Customer'])
    df_test['Postcode_encoded']  = le_postcode.transform(df_test['Postcode'])

    # 7. 特征列表（现在 df_train 已经有 one-hot 列了）
    cat_cols = ohe.get_feature_names_out().tolist()
    feature_cols = [
        'Generator Capacity', 'Customer_encoded', 'Postcode_encoded',
        'year', 'month', 'day', 'weekday', 'is_weekend'
    ] + cat_cols + [f'daily_total_kWh_lag_{i}' for i in range(1, lag_window+1)]

    # 8. 数值特征缩放（不缩放 one-hot 和 encoded 类别特征）
    num_cols = ['Generator Capacity'] + [f'daily_total_kWh_lag_{i}' for i in range(1, lag_window+1)]

    scaler_X_num = MinMaxScaler()
    scaler_y     = MinMaxScaler()

    X_train_num = scaler_X_num.fit_transform(df_train[num_cols])
    X_test_num  = scaler_X_num.transform(df_test[num_cols])

    # 其他特征（类别 + 时间特征）
    other_cols = [c for c in feature_cols if c not in num_cols]
    X_train_other = df_train[other_cols].values
    X_test_other  = df_test[other_cols].values

    # 合并
    import numpy as np
    X_train = np.hstack([X_train_num, X_train_other])
    X_test  = np.hstack([X_test_num,  X_test_other])

    y_train = df_train['daily_total_kWh'].values
    y_test  = df_test['daily_total_kWh'].values
    y_test_orig = y_test.copy()

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    return (X_train, X_test, y_train_scaled, y_test_scaled, y_test_orig,
            df_test, scaler_X_num, scaler_y, feature_cols)

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
def rolling_forecast_and_evaluate(model, X_test, y_test, y_test_orig, scaler_y,
                                 df_test, feature_cols, lag_window=7):
    """
    真正单步自回归滚动预测：每次预测后更新 lag 特征
    """
    from tqdm import tqdm
    import numpy as np
    import xgboost as xgb

    # 不再需要 model.eval() —— XGBoost Booster 预测时不需要这个

    predictions_scaled = []
    actuals = []

    # 准备初始 lag 缓冲（从测试集第一行开始）
    current_row = X_test[0:1].copy()   # shape (1, n_features)
    current_lags = current_row[:, -lag_window:].flatten()  # 最后 lag_window 列

    # 记录 Generator Capacity 用于 clip
    gen_caps = df_test['Generator Capacity'].values

    for i in tqdm(range(len(X_test)), desc="Rolling forecast (autoregressive)"):
        dtest = xgb.DMatrix(current_row)
        pred_scaled = model.predict(dtest)[0]
        predictions_scaled.append(pred_scaled)

        actuals.append(y_test_orig[i])

        if i == len(X_test) - 1:
            break

        # 准备下一行
        next_row = X_test[i+1:i+2].copy()

        # 更新 lag（在 scaled 空间）
        new_lags = np.roll(current_lags, 1)
        new_lags[0] = pred_scaled

        lag_start_idx = len(feature_cols) - lag_window
        next_row[0, lag_start_idx:] = new_lags

        current_row = next_row
        current_lags = new_lags

    # 后续还原尺度、clip、评估部分保持不变
    predictions = scaler_y.inverse_transform(
        np.array(predictions_scaled).reshape(-1, 1)
    ).flatten()
    predictions = np.maximum(predictions, 0)

    max_possible = gen_caps[:len(predictions)] * 8.0
    predictions = np.minimum(predictions, max_possible)

    # 评估指标计算...
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    epsilon = max(0.01, np.mean(actuals) * 0.05)
    mape_safe = np.mean(np.abs((np.array(actuals) - predictions) / 
                              (np.array(actuals) + epsilon))) * 100

    print(f"\nEvaluation (autoregressive rolling):")
    print(f"MAE  = {mae:.4f} kWh")
    print(f"RMSE = {rmse:.4f} kWh")
    print(f"MAPE = {mape_safe:.2f}% (safe)")
    print(f"Actual  stats: min={np.min(actuals):.3f}, mean={np.mean(actuals):.3f}, max={np.max(actuals):.3f}")
    print(f"Predict stats: min={np.min(predictions):.3f}, mean={np.mean(predictions):.3f}, max={np.max(predictions):.3f}")

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
    X_train, X_test, y_train, y_test, y_test_orig, df_test, scaler_X, scaler_y, feature_cols = prepare_data(df, LAG_WINDOW, TEST_RATIO)
    print("Data preparation success")
    print("Features:", feature_cols)
    
    # Train
    model = train_xgboost(X_train, y_train)
    print("Model training success")

    # Evaluate with rolling forecast
    mae, rmse, mape_safe, preds, acts = rolling_forecast_and_evaluate(
        model, X_test, y_test, y_test_orig, scaler_y,
        df_test, feature_cols, lag_window=LAG_WINDOW
    )