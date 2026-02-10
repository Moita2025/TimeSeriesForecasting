import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, List

from utils import create_sequences, TimeSeriesDataset

class NSWHalfHourDataModule:
    """
    NSW 半小时电力负荷 & 价格预测 数据模块
    
    主要职责：
    - 数据读取与清洗
    - 时间特征工程
    - train/val/test 切分
    - 特征与目标的独立归一化
    - 滑动窗口序列生成
    - 提供 DataLoader
    - 提供反归一化接口
    """

    def __init__(
        self,
        data_path: str,
        target_cols: List[str] = ["TOTALDEMAND", "RRP"],
        seq_len: int = 336,
        horizon: int = 48,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 64,
        shuffle_train: bool = True,
        num_workers: int = 0,
    ):
        self.data_path = data_path
        self.target_cols = target_cols
        self.seq_len = seq_len
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers

        # 将被初始化的对象
        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: Optional[List[str]] = None
        
        # 归一化器
        self.scaler_X = MinMaxScaler()
        self.scaler_demand = MinMaxScaler()
        self.scaler_price = MinMaxScaler()

        # 切分后的原始数据（用于反归一化参考）
        self.X_train_raw = None
        self.y_train_raw = None

        # 序列数据
        self.X_train_seq = None
        self.y_train_seq = None
        self.X_val_seq = None
        self.y_val_seq = None
        self.X_test_seq = None
        self.y_test_seq = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def prepare_data(self):
        """完整的数据准备流程"""
        self._load_and_clean_data()
        self._add_time_features()
        self._split_and_scale()
        self._create_sequences()
        self._create_dataloaders()

    def _load_and_clean_data(self):
        """读取数据并做基础清洗"""
        df = pd.read_csv(self.data_path, parse_dates=["SETTLEMENTDATE"])
        df = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)
        
        # 去掉无用列（根据你原来的逻辑）
        if "REGION" in df.columns and df["REGION"].nunique() == 1:
            df.drop(columns=["REGION"], inplace=True)
        if "PERIODTYPE" in df.columns and df["PERIODTYPE"].nunique() == 1:
            df.drop(columns=["PERIODTYPE"], inplace=True)

        # 可以在这里加入更多清洗逻辑（负值处理、异常值等）
        # 例如：
        # df.loc[df["RRP"] < -100, "RRP"] = np.nan
        # df["RRP"] = df["RRP"].ffill()

        self.df = df

    def _add_time_features(self):
        """添加通用时间特征"""
        dt = self.df["SETTLEMENTDATE"]
        self.df["year"] = dt.dt.year
        self.df["month"] = dt.dt.month
        self.df["day"] = dt.dt.day
        self.df["hour"] = dt.dt.hour
        self.df["weekday"] = dt.dt.weekday
        self.df["is_weekend"] = (self.df["weekday"] >= 5).astype(int)
        # 可选添加更多特征，如 dayofyear, sin/cos 周期编码等

    def _split_and_scale(self):
        """按时间顺序切分，并对特征和目标分别归一化"""
        n = len(self.df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # 原始数据
        df_train = self.df.iloc[:train_end]
        df_val   = self.df.iloc[train_end:val_end]
        df_test  = self.df.iloc[val_end:]

        # 特征列
        exclude = self.target_cols + ["SETTLEMENTDATE", "date"] if "date" in self.df else self.target_cols + ["SETTLEMENTDATE"]
        self.feature_cols = [c for c in self.df.columns if c not in exclude]

        # 分离 X 和 y
        X_all = self.df[self.feature_cols].values.astype(np.float32)
        y_all = self.df[self.target_cols].values.astype(np.float32)

        self.X_train_raw = X_all[:train_end]
        self.y_train_raw = y_all[:train_end]

        # fit scaler（只用训练集）
        self.scaler_X.fit(self.X_train_raw)
        self.scaler_demand.fit(self.y_train_raw[:, 0].reshape(-1, 1))
        self.scaler_price.fit( self.y_train_raw[:, 1].reshape(-1, 1))

        # transform 所有数据
        X_train = self.scaler_X.transform(self.X_train_raw)
        X_val   = self.scaler_X.transform(X_all[train_end:val_end])
        X_test  = self.scaler_X.transform(X_all[val_end:])

        y_train = np.column_stack([
            self.scaler_demand.transform(self.y_train_raw[:, 0].reshape(-1, 1)).ravel(),
            self.scaler_price.transform( self.y_train_raw[:, 1].reshape(-1, 1)).ravel()
        ])
        y_val = np.column_stack([
            self.scaler_demand.transform(y_all[train_end:val_end, 0].reshape(-1, 1)).ravel(),
            self.scaler_price.transform( y_all[train_end:val_end, 1].reshape(-1, 1)).ravel()
        ])
        y_test = np.column_stack([
            self.scaler_demand.transform(y_all[val_end:, 0].reshape(-1, 1)).ravel(),
            self.scaler_price.transform( y_all[val_end:, 1].reshape(-1, 1)).ravel()
        ])

        print(f"数据切分：train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")
        print(f"Demand 训练范围: {self.y_train_raw[:,0].min():.2f} → {self.y_train_raw[:,0].max():.2f}")
        print(f"Price  训练范围: {self.y_train_raw[:,1].min():.2f} → {self.y_train_raw[:,1].max():.2f}")

        self.X_train_scaled = X_train
        self.y_train_scaled = y_train
        self.X_val_scaled   = X_val
        self.y_val_scaled   = y_val
        self.X_test_scaled  = X_test
        self.y_test_scaled  = y_test

    def _create_sequences(self):
        """生成滑动窗口序列"""
        self.X_train_seq, self.y_train_seq = create_sequences(
            self.X_train_scaled, self.y_train_scaled, self.seq_len, self.horizon
        )
        self.X_val_seq, self.y_val_seq = create_sequences(
            self.X_val_scaled, self.y_val_scaled, self.seq_len, self.horizon
        )
        self.X_test_seq, self.y_test_seq = create_sequences(
            self.X_test_scaled, self.y_test_scaled, self.seq_len, self.horizon
        )

        print(f"序列数量：train={len(self.X_train_seq)}  val={len(self.X_val_seq)}  test={len(self.X_test_seq)}")

    def _create_dataloaders(self):
        """创建 DataLoader"""
        self.train_loader = DataLoader(
            TimeSeriesDataset(self.X_train_seq, self.y_train_seq),
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            drop_last=True
        )
        self.val_loader = DataLoader(
            TimeSeriesDataset(self.X_val_seq, self.y_val_seq),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        self.test_loader = DataLoader(
            TimeSeriesDataset(self.X_test_seq, self.y_test_seq),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    # ------------------ 对外提供的重要接口 ------------------

    def inverse_transform_demand(self, y_scaled: np.ndarray) -> np.ndarray:
        """将归一化的 demand 还原"""
        return self.scaler_demand.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def inverse_transform_price(self, y_scaled: np.ndarray) -> np.ndarray:
        """将归一化的 price 还原"""
        return self.scaler_price.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """还原完整的 (batch, horizon, 2) 或 (n, 2) 的目标"""
        shape = y_scaled.shape
        y_flat = y_scaled.reshape(-1, 2)
        demand = self.inverse_transform_demand(y_flat[:, 0])
        price  = self.inverse_transform_price(y_flat[:, 1])
        return np.column_stack([demand, price]).reshape(shape)

    @property
    def input_size(self) -> int:
        """模型需要的输入特征维度"""
        return len(self.feature_cols) if self.feature_cols else 0


# 使用示例（测试用）
if __name__ == "__main__":
    dm = NSWHalfHourDataModule(
        data_path="data_processed/nsw_halfhour_201207_201306.csv",
        seq_len=336,
        horizon=48,
        batch_size=64
    )
    dm.prepare_data()
    print("DataModule 准备完成")
    print("输入特征数:", dm.input_size)