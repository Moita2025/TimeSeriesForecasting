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
    """
    按每个客户独立切分 train/test，避免跨客户/未来数据泄漏
    返回值与之前基本兼容，但 df_test 会保留完整信息供滚动预测使用
    """
    # 1. 添加时间特征
    df['year']   = df['date'].dt.year
    df['month']  = df['date'].dt.month
    df['day']    = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # 2. 滞后特征（每个 Customer + Consumption Category 独立）
    df['group'] = df['Customer'].astype(str) + '_' + df['Consumption Category']
    groups = df.groupby('group')
    for lag in range(1, lag_window + 1):
        df[f'daily_total_kWh_lag_{lag}'] = groups['daily_total_kWh'].shift(lag)
    df = df.drop(columns=['group'])

    # 3. 按客户切分 train/test
    train_dfs = []
    test_dfs = []

    for cust, sub in df.groupby('Customer'):
        sub = sub.sort_values('date').reset_index(drop=True)
        if len(sub) < lag_window + 10:  # 样本太少 → 全放训练
            train_dfs.append(sub)
            continue

        cut_idx = max(lag_window + 1, int(len(sub) * (1 - test_ratio)))
        train_part = sub.iloc[:cut_idx].copy()
        test_part  = sub.iloc[cut_idx:].copy()

        train_dfs.append(train_part)
        test_dfs.append(test_part)

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test  = pd.concat(test_dfs, ignore_index=True)

    # 4. 删除训练集中的 lag NaN
    df_train = df_train.dropna().reset_index(drop=True)

    # 5. 类别编码（推荐只在 train 上 fit）
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(df_train[['Consumption Category']])

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

    # 合并 one-hot 列回 df
    df_train = pd.concat([df_train, train_cat], axis=1)
    df_test  = pd.concat([df_test, test_cat], axis=1)

    # Label Encoding for Customer & Postcode
    le_customer = LabelEncoder()
    le_postcode = LabelEncoder()

    # fit on train + test 合并数据（常见做法，避免 unseen label）
    le_customer.fit(pd.concat([df_train['Customer'], df_test['Customer']]))
    le_postcode.fit(pd.concat([df_train['Postcode'], df_test['Postcode']]))

    df_train['Customer_encoded'] = le_customer.transform(df_train['Customer'])
    df_train['Postcode_encoded'] = le_postcode.transform(df_train['Postcode'])
    df_test['Customer_encoded']  = le_customer.transform(df_test['Customer'])
    df_test['Postcode_encoded']  = le_postcode.transform(df_test['Postcode'])

    # 6. 特征列表
    cat_cols = ohe.get_feature_names_out().tolist()
    feature_cols = [
        'Generator Capacity', 'Customer_encoded', 'Postcode_encoded',
        'year', 'month', 'day', 'weekday', 'is_weekend'
    ] + cat_cols + [f'daily_total_kWh_lag_{i}' for i in range(1, lag_window+1)]

    # 7. 缩放（只缩放连续型数值特征）
    num_cols = ['Generator Capacity'] + [f'daily_total_kWh_lag_{i}' for i in range(1, lag_window+1)]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_num = scaler_X.fit_transform(df_train[num_cols])
    X_test_num  = scaler_X.transform(df_test[num_cols])

    other_cols = [c for c in feature_cols if c not in num_cols]
    X_train_other = df_train[other_cols].values
    X_test_other  = df_test[other_cols].values

    import numpy as np
    X_train = np.hstack([X_train_num, X_train_other])
    X_test  = np.hstack([X_test_num,  X_test_other])

    y_train = df_train['daily_total_kWh'].values
    y_test  = df_test['daily_total_kWh'].values
    y_test_orig = y_test.copy()

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    return (X_train, X_test, y_train_scaled, y_test_scaled, y_test_orig,
            df_test, scaler_X, scaler_y, feature_cols)

# ────────────────────────────────────────────────
# 3. 建立時間序列資料（每日尺度）
# ────────────────────────────────────────────────
def create_sequences(X, y, seq_length=14, pred_length=1):
    if isinstance(X, pd.DataFrame):
        X = X.values.astype(np.float32)
    if isinstance(y, pd.Series):
        y = y.values
        
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
                                  seq_length=14, forecast_horizon=1, device=torch.device("cpu"), lag_window=7):
    """
    真正自回归单步滚动预测：每次只预测一步，并用预测值更新序列
    """
    model.eval()

    predictions = []
    actuals = []

    # 初始序列：从测试集开头取 seq_length 步（假设测试集足够长）
    # 如果测试集长度 < seq_length + forecast_horizon，会自动跳过或处理
    X_test_np = df_test[feature_cols].values.astype(np.float32)
    y_test_orig = df_test['daily_total_kWh'].values

    if len(X_test_np) < seq_length + forecast_horizon:
        print("测试集太短，无法进行滚动预测")
        return None, None, None, [], []

    # 初始输入序列（使用真实的过去值作为起点）
    current_seq = X_test_np[:seq_length].copy()   # shape: (seq_length, n_features)
    current_seq = current_seq.reshape(1, seq_length, -1)  # (1, seq_length, n_features)

    gen_caps = df_test['Generator Capacity'].values

    for i in tqdm(range(seq_length, len(X_test_np), forecast_horizon),
                  desc="Autoregressive rolling forecast"):

        input_tensor = torch.from_numpy(current_seq).float().to(device)

        pred_scaled = model(input_tensor).cpu().numpy().flatten()  # 假设输出 shape (1,1)

        # 还原尺度 & clip
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        pred = np.maximum(pred, 0)
        # 物理上限：容量 × 合理日照小时（例如 8 小时）
        max_possible = gen_caps[i] * 8.0 if i < len(gen_caps) else 50.0
        pred = min(pred, max_possible)

        true = y_test_orig[i]

        predictions.append(pred)
        actuals.append(true)

        # ------------------ 自回归更新序列 ------------------
        # 准备下一行的特征（从 df_test 取其他特征 + 新预测的 lag 值）
        if i + 1 >= len(X_test_np):
            break

        next_features = X_test_np[i+1].copy()  # 下一天的其他特征（时间、容量、编码等）

        # 更新 lag 部分：把预测值放入 lag_1，其余 lag 后移
        lag_start = len(feature_cols) - lag_window  # 假设最后 lag_window 列是 lag
        # 当前序列的最后一个时间步的 lag 部分
        last_timestep = current_seq[0, -1, :].copy()

        # 后移 lag
        new_lags = np.roll(last_timestep[lag_start:], 1)
        new_lags[0] = pred_scaled[0]  # 新预测值（scaled）

        # 构建新的 timestep 特征
        new_timestep = next_features.copy()
        new_timestep[lag_start:] = new_lags

        # 序列向前滚动：丢弃最旧，加入新 timestep
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = new_timestep

    # 评估
    if not predictions:
        return None, None, None, [], []

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    epsilon = max(1e-3, np.mean(actuals) * 0.05)
    mape_safe = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100

    print(f"\nEvaluation (true autoregressive rolling):")
    print(f"MAE  = {mae:.4f} kWh")
    print(f"RMSE = {rmse:.4f} kWh")
    print(f"MAPE = {mape_safe:.2f}% (safe)")
    print(f"Actual  stats: min={actuals.min():.3f}, mean={actuals.mean():.3f}, max={actuals.max():.3f}")
    print(f"Predict stats: min={predictions.min():.3f}, mean={predictions.mean():.3f}, max={predictions.max():.3f}")

    return mae, rmse, mape_safe, predictions.tolist(), actuals.tolist()