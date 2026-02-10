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
        self.scaler_demand = MinMaxScaler()
        self.scaler_price  = MinMaxScaler()

    def prepare_data(self) -> Tuple:
        df = self.load_data()
        df = self.prepare_features(df)

        targets = df[self.target_cols].values.astype(np.float32)   # (n, 2)
        exclude_cols = self.target_cols + ["SETTLEMENTDATE", "date"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.feature_cols = feature_cols
        X = df[feature_cols].values.astype(np.float32)

        # ── 关键改动：严格按原始时间序列切分 ───────────────────────────────
        n = len(X)
        train_end = int(n * 0.7)          # 建议先用70%训练，后面可调
        val_end   = int(n * 0.85)         # 15%验证
        # test: val_end → end

        X_train_raw = X[:train_end]
        y_train_raw = targets[:train_end]

        X_val_raw   = X[train_end:val_end]
        y_val_raw   = targets[train_end:val_end]

        X_test_raw  = X[val_end:]
        y_test_raw  = targets[val_end:]

        # 分别 fit scaler（只用训练集）
        self.scaler_X.fit(X_train_raw)
        X_train = self.scaler_X.transform(X_train_raw)
        X_val   = self.scaler_X.transform(X_val_raw)
        X_test  = self.scaler_X.transform(X_test_raw)

        # 分别缩放两个目标（只fit训练集）
        self.scaler_demand.fit(y_train_raw[:, 0].reshape(-1, 1))
        self.scaler_price.fit( y_train_raw[:, 1].reshape(-1, 1))

        y_train = np.column_stack([
            self.scaler_demand.transform(y_train_raw[:, 0].reshape(-1, 1)).ravel(),
            self.scaler_price.transform( y_train_raw[:, 1].reshape(-1, 1)).ravel()
        ])

        y_val = np.column_stack([
            self.scaler_demand.transform(y_val_raw[:, 0].reshape(-1, 1)).ravel(),
            self.scaler_price.transform( y_val_raw[:, 1].reshape(-1, 1)).ravel()
        ])

        y_test = np.column_stack([
            self.scaler_demand.transform(y_test_raw[:, 0].reshape(-1, 1)).ravel(),
            self.scaler_price.transform( y_test_raw[:, 1].reshape(-1, 1)).ravel()
        ])

        print(f"序列切分：train:{len(X_train):5d}  val:{len(X_val):5d}  test:{len(X_test):5d}")

        # ── 创建序列 ────────────────────────────────────────────────────────
        X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, self.seq_len, self.horizon)
        X_val_seq, y_val_seq = create_sequences(X_val,   y_val,   self.seq_len, self.horizon)
        X_ts_seq, y_ts_seq = create_sequences(X_test,   y_test,   self.seq_len, self.horizon)

        extra = {
            "df_test": df.iloc[val_end:].reset_index(drop=True),  # 改成测试段
            "feature_cols": feature_cols,
            "scaler_X": self.scaler_X,
            "scaler_demand": self.scaler_demand,
            "scaler_price": self.scaler_price,
            "X_test_seq": X_ts_seq,
            "y_test_seq": y_ts_seq,          # 新增，便于评估
        }

        print("Demand 训练集范围:", y_train_raw[:,0].min(), "→", y_train_raw[:,0].max())
        print("Price  训练集范围:", y_train_raw[:,1].min(), "→", y_train_raw[:,1].max())

        # 传给 config（main.py 会用）
        self.config.update({
            'demand_min': float(y_train_raw[:,0].min()),
            'demand_max': float(y_train_raw[:,0].max()),
            'price_min':  float(y_train_raw[:,1].min()),
            'price_max':  float(y_train_raw[:,1].max()),
        })

        return X_tr_seq, y_tr_seq, X_ts_seq, y_ts_seq, extra, X_test_raw

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
        
        # Demand MAPE
        mask_d = np.abs(demand_true) > 1000   # 负荷低于1000MW视为异常/不计算
        mape_d = np.mean(np.abs((demand_true - demand_pred)[mask_d] / demand_true[mask_d])) * 100 if mask_d.sum() > 0 else np.nan

        # Price
        mae_p  = mean_absolute_error(price_true, price_pred)
        rmse_p = np.sqrt(mean_squared_error(price_true, price_pred))

        # Price MAPE（电力价格可能接近0甚至负）
        mask_p = np.abs(price_true) > 5.0
        mape_p = np.mean(np.abs((price_true - price_pred)[mask_p] / price_true[mask_p])) * 100 if mask_p.sum() > 0 else np.nan

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