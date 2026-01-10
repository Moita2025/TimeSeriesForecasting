import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from ..utils.time_series_utils import scale_data

class MultiVarTimeSeriesModel:
    """多变量时间序列模型训练器 - 支持多输入多输出"""

    def __init__(self, name="MultiVarModel", scaler_type='minmax', seq_length=96, pred_length=96):
        self.name = name
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.model = None
        self.history = None
        self.scaler_X = None
        self.scaler_y = None

    def prepare_data(self, data: pd.DataFrame, feature_cols, target_cols):
        """
        准备数据：假设data是DataFrame，feature_cols是输入列，target_cols是输出列
        返回X (samples, seq_length, n_features), y (samples, pred_length, n_targets)
        """
        # 假设用户已处理好归一化，这里简单应用
        features = data[feature_cols].values
        targets = data[target_cols].values

        features_scaled, self.scaler_X = scale_data(data[feature_cols], is_dict=False)
        targets_scaled, self.scaler_y = scale_data(data[target_cols], is_dict=False)
        
        X, y = [], []
        for i in range(len(features) - self.seq_length - self.pred_length + 1):
            X.append(features_scaled[i:i + self.seq_length])
            y.append(targets_scaled[i + self.seq_length:i + self.seq_length + self.pred_length])
        
        return np.array(X), np.array(y)

    def train(self, data: pd.DataFrame, feature_cols, target_cols, validation_split=0.15, epochs=1, patience=18, batch_size=32):
        X, y = self.prepare_data(data, feature_cols, target_cols)
        
        # 简单划分训练/验证
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self.history

    def predict(self, data: pd.DataFrame, feature_cols, target_cols):
        X, _ = self.prepare_data(data, feature_cols, target_cols)  # 只用最后seq_length做预测
        last_X = X[-1:]  # 取最后一个窗口
        pred_scaled = self.model.predict(last_X)
        pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, pred_scaled.shape[-1]))
        return pred.reshape(self.pred_length, len(target_cols))

    def evaluate(self, test_data: pd.DataFrame, 
                 feature_cols, target_cols,
                 horizon_points=None, verbose=True):
        '''
        X_test, y_test = self.prepare_data(test_data, feature_cols, target_cols)
        loss, mae = self.model.evaluate(X_test, y_test)
        return {'loss': loss, 'mae': mae}
        '''

        if self.model is None:
            raise ValueError("模型尚未初始化，请先调用 build_model() 并训练")

        if horizon_points is None:
            horizon_points = self.pred_length

        # 准备完整序列用于滚动预测
        features = test_data[feature_cols].values
        targets = test_data[target_cols].values

        # 注意：这里使用已有的 scaler（假设已在 prepare_data/train 中拟合好）
        features_scaled = self.scaler_X.transform(test_data[feature_cols])
        targets_scaled = self.scaler_y.transform(test_data[target_cols])  # 用于对比

        predictions = []
        actuals = []

        # 滚动窗口数量
        n_windows = len(test_data) - self.seq_length - horizon_points + 1

        if n_windows <= 0:
            raise ValueError(
                f"测试集长度不足以进行滚动评估。\n"
                f"需要至少 {self.seq_length + horizon_points} 个时间点，"
                f"当前只有 {len(test_data)}"
            )

        for i in tqdm(range(n_windows), desc="滚动评估进度"):
            # 准备当前输入窗口
            input_seq = features_scaled[i:i + self.seq_length]
            input_seq = input_seq.reshape(1, self.seq_length, len(feature_cols))

            # 模型预测（scaled）
            pred_scaled = self.model.predict(input_seq, verbose=0)

            # 反归一化 → 真实尺度
            pred = self.scaler_y.inverse_transform(
                pred_scaled.reshape(-1, len(target_cols))
            )  # shape: (horizon_points, n_targets)

            # 真实未来值
            true_future = targets[i + self.seq_length:i + self.seq_length + horizon_points]

            predictions.append(pred)
            actuals.append(true_future)

        # 转换为 numpy 数组
        predictions = np.array(predictions)   # (n_windows, horizon_points, n_targets)
        actuals = np.array(actuals)           # (n_windows, horizon_points, n_targets)

        # 计算各类指标（所有窗口、所有步长、所有目标整体计算）
        mae = mean_absolute_error(actuals.ravel(), predictions.ravel())
        rmse = np.sqrt(mean_squared_error(actuals.ravel(), predictions.ravel()))

        # MAPE 需要小心处理分母为0的情况
        mask = np.abs(actuals.ravel()) > 1e-6  # 避免除以接近0的值
        if np.any(mask):
            mape = np.mean(np.abs(
                (actuals.ravel()[mask] - predictions.ravel()[mask]) / 
                actuals.ravel()[mask]
            )) * 100
        else:
            mape = np.nan
            print("警告：实际值几乎全为0，无法计算有意义的 MAPE")

        # 可选：按目标分别统计（更细粒度）
        metrics_per_target = {}
        for idx, col in enumerate(target_cols):
            mae_t = mean_absolute_error(actuals[..., idx], predictions[..., idx])
            rmse_t = np.sqrt(mean_squared_error(actuals[..., idx], predictions[..., idx]))
            metrics_per_target[f"{col}_MAE"] = mae_t
            metrics_per_target[f"{col}_RMSE"] = rmse_t

        if verbose:
            print("\n" + "="*50)
            print("多变量时间序列 滚动预测性能（整体）")
            print(f"预测窗口数: {n_windows}")
            print(f"预测步长:   {horizon_points}")
            print(f"目标变量数: {len(target_cols)}")
            print("-"*50)
            print(f"整体 MAE:  {mae:,.4f}")
            print(f"整体 RMSE: {rmse:,.4f}")
            print(f"整体 MAPE: {mape:.2f}% (若出现 nan 则数据接近零)")
            print("-"*50)
            print("各目标单独指标：")
            for name, value in metrics_per_target.items():
                print(f"  {name:12}: {value:,.4f}")
            print("="*50)

        result = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            **metrics_per_target,           # 加入每个目标的单独指标
            'n_windows': n_windows,
            'horizon': horizon_points
        }

        return result