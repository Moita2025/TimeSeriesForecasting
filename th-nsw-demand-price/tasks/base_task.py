import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any

class BaseForecastTask:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_file = config.get("data_file")
        self.target_col = config.get("target_col", "RRP")
        self.seq_len = config.get("seq_len", 336)          # 默认过去7天
        self.horizon = config.get("horizon", 48)           # 默认预测下一天48点
        self.train_ratio = config.get("train_ratio", 0.8)

        self.df = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_cols = None

    def load_data(self) -> pd.DataFrame:
        """子类可覆盖"""
        df = pd.read_csv(self.data_file, parse_dates=["SETTLEMENTDATE"])
        df = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)
        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加通用时间特征，子类可扩展"""
        df = df.copy()
        dt = df["SETTLEMENTDATE"]
        df["year"]   = dt.dt.year
        df["month"]  = dt.dt.month
        df["day"]    = dt.dt.day
        df["hour"]   = dt.dt.hour
        df["weekday"] = dt.dt.weekday
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)
        df["dayofyear"] = dt.dt.dayofyear
        df["hourofyear"] = dt.dt.dayofyear * 24 + dt.dt.hour
        return df
    
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
                """
                pred = model(Xb)
                loss = criterion(pred, yb)
                """
                # 在 train_model 函数里，计算 loss 前加一行
                pred = model(Xb)
                yb_last = yb[:, -1, :]          # 只看第 48 步（或改成 [:, 0, :] 只看第一步）
                loss = criterion(pred, yb_last)

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
                    """
                    pred = model(Xb)
                    loss = criterion(pred, yb)
                    """
                    # 在 train_model 函数里，计算 loss 前加一行
                    pred = model(Xb)
                    yb_last = yb[:, -1, :]          # 只看第 48 步（或改成 [:, 0, :] 只看第一步）

                    loss = criterion(pred, yb_last)
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

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        返回: X_train, y_train, X_test, y_test, extra_info
        extra_info 可以放 scaler、df_test 等
        """
        raise NotImplementedError("请在子类中实现 prepare_data")

    def rolling_forecast_and_evaluate(self, model, X_test, y_test, **kwargs):
        """
        滚动预测 & 评估
        不同任务的滚动方式差异较大，因此留给子类实现
        """
        raise NotImplementedError("请在子类中实现 rolling_forecast_and_evaluate")