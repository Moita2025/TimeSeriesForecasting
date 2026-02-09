import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .base_task import BaseForecastTask
from utils import create_sequences
from typing import Tuple, Dict, Any

class NextDay48HalfHoursRRP(BaseForecastTask):
    """预测下一天的48个半小时 TOTALDEMAND 和 RRP"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 覆盖默认目标
        self.target_cols = ["TOTALDEMAND", "RRP"]   # 两个目标
        self.target_dim = len(self.target_cols)

    def prepare_data(self) -> Tuple:
        df = self.load_data()
        df = self.prepare_features(df)

        # 多目标
        targets = df[self.target_cols].values.astype(np.float32)   # shape: (n_timesteps, 2)

        # 特征列（排除目标和非特征列）
        exclude_cols = self.target_cols + ["SETTLEMENTDATE", "date"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.feature_cols = feature_cols

        X = df[feature_cols].values.astype(np.float32)

        # 时间顺序切分（更建议按自然日切分，见后续建议）
        train_size = int(len(X) * self.train_ratio)
        X_train_raw = X[:train_size]
        X_test_raw  = X[train_size:]
        y_train_raw = targets[:train_size]
        y_test_raw  = targets[train_size:]

        # 缩放 X（特征）
        self.scaler_X.fit(X_train_raw)
        X_train = self.scaler_X.transform(X_train_raw)
        X_test  = self.scaler_X.transform(X_test_raw)

        # 分别缩放两个目标（推荐做法，避免量纲差异太大）
        self.scaler_demand = MinMaxScaler()
        self.scaler_price  = MinMaxScaler()

        y_train_demand = self.scaler_demand.fit_transform(
            y_train_raw[:, 0].reshape(-1, 1)
        ).ravel()
        y_train_price = self.scaler_price.fit_transform(
            y_train_raw[:, 1].reshape(-1, 1)
        ).ravel()

        y_test_demand = self.scaler_demand.transform(
            y_test_raw[:, 0].reshape(-1, 1)
        ).ravel()
        y_test_price = self.scaler_price.transform(
            y_test_raw[:, 1].reshape(-1, 1)
        ).ravel()

        # 合并成 (n, 2)
        y_train = np.column_stack([y_train_demand, y_train_price])
        y_test  = np.column_stack([y_test_demand,  y_test_price])

        extra = {
            "df_test": df.iloc[train_size:].reset_index(drop=True),
            "feature_cols": feature_cols,
            "scaler_X": self.scaler_X,
            "scaler_demand": self.scaler_demand,
            "scaler_price": self.scaler_price,
        }

        #return X_train, y_train, X_test, y_test, extra

        print("正在创建滑动窗口序列...")

        X_train_seq, y_train_seq = create_sequences(
            X_train, y_train,
            seq_length=self.seq_len,      # 336
            pred_length=self.horizon      # 48
        )

        X_test_seq, y_test_seq = create_sequences(
            X_test, y_test,
            seq_length=self.seq_len,
            pred_length=self.horizon
        )

        print(f"训练序列数量: {len(X_train_seq)}, 测试序列数量: {len(X_test_seq)}")
        print(f"输入形状示例: {X_train_seq.shape}")   # 应为 (n_seq, 336, 9)

        extra["X_test_seq"] = X_test_seq   # 如果后续滚动预测需要

        return X_train_seq, y_train_seq, X_test_seq, y_test_seq, extra, X_test

    def rolling_forecast_and_evaluate(self, model, X_test, y_test, extra: Dict, device="cpu"):
        """
        注意：這裡的 X_test 應該是原始的 2D 陣列 (n_timesteps, n_features)
        而不是滑窗後的 3D 陣列
        """
        from tqdm import tqdm
        import torch
        import numpy as np

        df_test = extra["df_test"]
        scaler_d = extra["scaler_demand"]
        scaler_p = extra["scaler_price"]

        predictions = []
        actuals = []

        # ── 關鍵：確保 X_test 是 2D ───────────────────────────────
        if X_test.ndim != 2:
            raise ValueError(f"預期 X_test 是 2D 陣列 (n_timesteps, features)，實際得到 {X_test.shape}")

        if len(X_test) < self.seq_len:
            raise ValueError("測試資料太短，無法產生至少一個完整序列")

        # 初始化第一個輸入序列
        current_seq = X_test[:self.seq_len].copy()          # shape: (336, 9)
        print("初始化 current_seq shape:", current_seq.shape)   # debug 用，可之後刪除

        # 迴圈從第 seq_len 個時間點開始預測
        total_steps = min(len(X_test) - self.seq_len, len(y_test))
        pbar = tqdm(total=total_steps, desc="Rolling forecast")

        for step in range(total_steps):
            # 轉 tensor + 加 batch 維度
            inp = torch.from_numpy(current_seq).float().unsqueeze(0).to(device)  # (1, 336, 9)

            with torch.no_grad():
                pred_scaled = model(inp).cpu().numpy()[0]          # (2,)

            # 反歸一化預測值
            pred_demand = scaler_d.inverse_transform([[pred_scaled[0]]])[0, 0]
            pred_price  = scaler_p.inverse_transform([[pred_scaled[1]]])[0, 0]

            # 真實值：對應當前預測的時間點（第 step + seq_len 個時間點）
            # 但因為 y_test 是滑窗後的 (n_seq, 48, 2)，我們用 step 索引
            true_demand_scaled = y_test[step, 0, 0]   # 第 step 個窗口的第一步 demand
            true_price_scaled  = y_test[step, 0, 1]

            true_demand = scaler_d.inverse_transform([[true_demand_scaled]])[0, 0]
            true_price  = scaler_p.inverse_transform([[true_price_scaled]])[0, 0]

            predictions.append([pred_demand, pred_price])
            actuals.append([true_demand, true_price])

            # 更新 current_seq 到下一個時間窗
            next_idx = self.seq_len + step
            if next_idx >= len(X_test):
                break

            next_features = X_test[next_idx]   # (9,)

            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1] = next_features

            pbar.update(1)

        pbar.close()

        # ── 後續評估指標計算 ────────────────────────────────────────
        predictions = np.array(predictions)  # shape: (n_steps, 2)
        actuals     = np.array(actuals)      # shape: (n_steps, 2)

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # 分開計算 Demand 和 Price
        demand_actual = actuals[:, 0]
        demand_pred   = predictions[:, 0]
        price_actual  = actuals[:, 1]
        price_pred    = predictions[:, 1]

        # Demand 指標
        mae_d   = mean_absolute_error(demand_actual, demand_pred)
        rmse_d  = np.sqrt(mean_squared_error(demand_actual, demand_pred))
        epsilon_d = max(1e-3, np.mean(demand_actual) * 0.05)
        mape_d  = np.mean(np.abs((demand_actual - demand_pred) / (demand_actual + epsilon_d))) * 100

        # Price 指標
        mae_p   = mean_absolute_error(price_actual, price_pred)
        rmse_p  = np.sqrt(mean_squared_error(price_actual, price_pred))
        epsilon_p = max(1e-3, np.mean(price_actual) * 0.05)
        mape_p  = np.mean(np.abs((price_actual - price_pred) / (price_actual + epsilon_p))) * 100

        print("\nEvaluation (true autoregressive rolling - one-step ahead):")
        print("─" * 60)
        print(f"Demand:")
        print(f"  MAE  = {mae_d:8.4f} MW")
        print(f"  RMSE = {rmse_d:8.4f} MW")
        print(f"  MAPE = {mape_d:6.2f}% (safe)")
        print(f"  Actual  stats: min={demand_actual.min():8.3f}, mean={demand_actual.mean():8.3f}, max={demand_actual.max():8.3f}")
        print(f"  Predict stats: min={demand_pred.min():8.3f},   mean={demand_pred.mean():8.3f},   max={demand_pred.max():8.3f}")
        print()
        print(f"Price (RRP):")
        print(f"  MAE  = {mae_p:8.4f} $/MWh")
        print(f"  RMSE = {rmse_p:8.4f} $/MWh")
        print(f"  MAPE = {mape_p:6.2f}% (safe)")
        print(f"  Actual  stats: min={price_actual.min():8.3f}, mean={price_actual.mean():8.3f}, max={price_actual.max():8.3f}")
        print(f"  Predict stats: min={price_pred.min():8.3f},   mean={price_pred.mean():8.3f},   max={price_pred.max():8.3f}")
        print("─" * 60)

        # 如果你之後想回傳多個指標，可以這樣包裝
        metrics = {
            "demand": {"mae": mae_d, "rmse": rmse_d, "mape": mape_d},
            "price":  {"mae": mae_p, "rmse": rmse_p, "mape": mape_p}
        }

        return metrics, predictions.tolist(), actuals.tolist()