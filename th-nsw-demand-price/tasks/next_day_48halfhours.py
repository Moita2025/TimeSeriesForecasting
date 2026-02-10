import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .base_task import BaseForecastTask
from utils import create_sequences
from typing import Tuple, Dict, Any
import torch

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
        现在使用批量前向预测的方式评估所有测试窗口
        不需要逐步滚动了（因为模型已经学会一次性输出48步）
        """
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from tqdm import tqdm

        # 取出测试集的滑动窗口数据（prepare_data 里已经准备好了）
        X_test_seq = extra["X_test_seq"]          # (n_windows, 336, n_features)
        y_test_seq = y_test                       # (n_windows, 48, 2) scaled

        scaler_d = extra["scaler_demand"]
        scaler_p = extra["scaler_price"]

        model.eval()
        all_pred = []
        all_true = []

        batch_size = 128  # 可根据显存调整
        n_samples = len(X_test_seq)

        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="Batch inference"):
                end = min(i + batch_size, n_samples)
                Xb = torch.from_numpy(X_test_seq[i:end]).float().to(device)
                
                pred_scaled = model(Xb).cpu().numpy()          # (batch_this, 48, 2)

                # 反归一化
                pred_d = scaler_d.inverse_transform(
                    pred_scaled[..., 0].reshape(-1, 1)
                ).reshape(-1, 48)
                
                pred_p = scaler_p.inverse_transform(
                    pred_scaled[..., 1].reshape(-1, 1)
                ).reshape(-1, 48)

                true_d = scaler_d.inverse_transform(
                    y_test_seq[i:end, :, 0].reshape(-1, 1)
                ).reshape(-1, 48)
                
                true_p = scaler_p.inverse_transform(
                    y_test_seq[i:end, :, 1].reshape(-1, 1)
                ).reshape(-1, 48)

                all_pred.append(np.stack([pred_d, pred_p], axis=-1))   # (batch_this, 48, 2)
                all_true.append(np.stack([true_d, true_p], axis=-1))

        # 合并所有窗口
        predictions = np.concatenate(all_pred, axis=0)   # (total_windows, 48, 2)
        actuals     = np.concatenate(all_true, axis=0)

        # ── 计算指标 ────────────────────────────────────────────────
        # 这里可以选择不同粒度评估：
        # 1. 所有步平均
        # 2. 只看第1步（短期）
        # 3. 只看第48步（24小时后）
        # 4. 每一步单独算，再平均

        # 目前用最全面的：所有步展平后算
        demand_true = actuals[..., 0].ravel()
        demand_pred = predictions[..., 0].ravel()
        price_true  = actuals[..., 1].ravel()
        price_pred  = predictions[..., 1].ravel()

        # Demand
        mae_d  = mean_absolute_error(demand_true, demand_pred)
        rmse_d = np.sqrt(mean_squared_error(demand_true, demand_pred))
        mape_d = np.mean(np.abs((demand_true - demand_pred) / (demand_true + 1e-3))) * 100

        # Price
        mae_p  = mean_absolute_error(price_true, price_pred)
        rmse_p = np.sqrt(mean_squared_error(price_true, price_pred))
        mape_p = np.mean(np.abs((price_true - price_pred) / (price_true + 1e-3))) * 100

        print("\nEvaluation (full 48-step direct forecasting):")
        print("─" * 60)
        print("Demand:")
        print(f"  MAE  = {mae_d:8.2f} MW")
        print(f"  RMSE = {rmse_d:8.2f} MW")
        print(f"  MAPE = {mape_d:6.2f}%")
        print(f"  Actual mean: {demand_true.mean():8.2f}")
        print(f"  Pred   mean: {demand_pred.mean():8.2f}")
        print()
        print("Price (RRP):")
        print(f"  MAE  = {mae_p:8.2f} $/MWh")
        print(f"  RMSE = {rmse_p:8.2f} $/MWh")
        print(f"  MAPE = {mape_p:6.2f}%")
        print(f"  Actual mean: {price_true.mean():8.2f}")
        print(f"  Pred   mean: {price_pred.mean():8.2f}")
        print("─" * 60)

        metrics = {
            "demand": {"mae": mae_d, "rmse": rmse_d, "mape": mape_d},
            "price":  {"mae": mae_p, "rmse": rmse_p, "mape": mape_p}
        }

        return metrics, predictions.tolist(), actuals.tolist()