import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")  # 抑制收敛警告

# ────────────────────────────────────────────────
# 1. 載入資料（與 XGBoost 一致）
# ────────────────────────────────────────────────
def load_daily_data(file_path='data_daily_long.csv'):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values(['Customer', 'Consumption Category', 'date']).reset_index(drop=True)
    return df

# ────────────────────────────────────────────────
# 2. 特徵工程（參考 XGBoost 風格）
# ────────────────────────────────────────────────
def prepare_data(df, lag_window=7, test_ratio=0.2):
    # 時間特徵
    df['year']   = df['date'].dt.year
    df['month']  = df['date'].dt.month
    df['day']    = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # 滯後特徵（每個 Customer + Consumption Category 獨立計算）
    df['group'] = df['Customer'].astype(str) + '_' + df['Consumption Category']
    groups = df.groupby('group')
    for lag in range(1, lag_window + 1):
        df[f'daily_total_kWh_lag_{lag}'] = groups['daily_total_kWh'].shift(lag)
    df = df.drop(columns=['group'])
    df = df.dropna().reset_index(drop=True)

    # 類別編碼
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(df[['Consumption Category']])
    cat_cols = ohe.get_feature_names_out(['Consumption Category']).tolist()

    # ★ 這裡合併回 df
    df_cat = pd.DataFrame(cat_encoded, columns=cat_cols, index=df.index)
    df = pd.concat([df, df_cat], axis=1)

    le_customer = LabelEncoder()
    le_postcode = LabelEncoder()
    df['Customer_encoded'] = le_customer.fit_transform(df['Customer'])
    df['Postcode_encoded'] = le_postcode.fit_transform(df['Postcode'])

    # 特徵欄位組合
    feature_cols = [
        'Generator Capacity', 'Customer_encoded', 'Postcode_encoded',
        'year', 'month', 'day', 'weekday', 'is_weekend'
    ] + cat_cols + [f'daily_total_kWh_lag_{i}' for i in range(1, lag_window+1)]

    target_col = 'daily_total_kWh'

    # 縮放
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = df[feature_cols].copy()
    y = df[[target_col]]

    X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=feature_cols)
    y_scaled = scaler_y.fit_transform(y).flatten()

    # 時間順序分割（全局）
    train_size = int(len(X) * (1 - test_ratio))
    X_train = X_scaled.iloc[:train_size]
    X_test  = X_scaled.iloc[train_size:]
    y_train = y_scaled[:train_size]
    y_test  = y_scaled[train_size:]
    y_test_orig = y.iloc[train_size:].values.flatten()   # 用於後續評估原始尺度

    df_test = df.iloc[train_size:].reset_index(drop=True)  # 保留原始 df_test 供滾動預測使用

    return (X_train, X_test, y_train, y_test, y_test_orig,
            df_test, scaler_X, scaler_y, feature_cols)

# ────────────────────────────────────────────────
# 3. 建立時間序列資料（每日尺度）
# ────────────────────────────────────────────────
def create_sequences(X, y, seq_length=14, pred_length=1):
    Xs, ys = [], []
    for i in range(len(X) - seq_length - pred_length + 1):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[(i + seq_length):(i + seq_length + pred_length)])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# ────────────────────────────────────────────────
# 4. 訓練函數（基本保留）
# ────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=80, lr=0.001, patience=15,
                grad_clip=1.0, device = torch.device("cpu")):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * Xb.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

# ────────────────────────────────────────────────
# 5. 滾動預測與評估（適配每日尺度）
# ────────────────────────────────────────────────
@torch.no_grad()
def rolling_forecast_and_evaluate(model, df_test, feature_cols, scaler_y,
                                  seq_length=14, forecast_horizon=1, device=torch.device("cpu")):
    model.eval()

    # 為了滾動更新，我們需要依序取出特徵（但這裡簡化為一次性預測）
    # 若要嚴格更新 lag，需在迴圈內重建輸入（較複雜，此處先用近似方式）
    X_test_np = df_test[feature_cols].values.astype(np.float32)
    y_test_orig = df_test['daily_total_kWh'].values

    predictions = []
    actuals = []

    n = len(X_test_np)
    for i in tqdm(range(seq_length, n - forecast_horizon + 1, forecast_horizon), desc="Rolling forecast"):
        steps = min(forecast_horizon, n - i)
        input_seq = X_test_np[i-seq_length:i].reshape(1, seq_length, -1)
        input_tensor = torch.from_numpy(input_seq).to(device)

        pred_scaled = model(input_tensor).cpu().numpy().flatten()[:steps]
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        pred = np.maximum(pred, 0)
        pred = np.minimum(pred, df_test['daily_total_kWh'].max() * 1.5)

        true = y_test_orig[i:i+steps]

        predictions.extend(pred)
        actuals.extend(true)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    epsilon = 1e-3 if np.mean(actuals) < 1 else np.mean(actuals) * 0.05
    mape_safe = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / (np.array(actuals) + epsilon))) * 100

    print(f"\nEvaluation: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE_SAFE={mape_safe:.2f}%")
    print(f"Actual  stats: min={np.min(actuals):.3f}, mean={np.mean(actuals):.3f}, max={np.max(actuals):.3f}")
    print(f"Predict stats: min={np.min(predictions):.3f}, mean={np.mean(predictions):.3f}, max={np.max(predictions):.3f}")

    return mae, rmse, mape_safe, predictions, actuals