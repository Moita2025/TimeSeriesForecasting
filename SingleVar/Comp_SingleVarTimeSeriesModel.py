import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import os
import re
import glob

class SingleVarTimeSeriesModel():

    """单变量时间序列模型的训练/预测/评估流程组件"""

    def __init__(self,
                 name="",
                 seq_length=96,
                 pred_length=96,
                 scaler=None):
        self.name = name
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        self.scaler = scaler if scaler is not None else MinMaxScaler(feature_range=(0, 1))
        self.model = None           # ← 外部要注入模型
        self.history = None

    def prepare_data(self, series: pd.Series):
        """时间序列 → 监督学习格式"""
        data = series.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled) - self.seq_length - self.pred_length + 1):
            X.append(scaled[i:i + self.seq_length])
            y.append(scaled[i + self.seq_length:i + self.seq_length + self.pred_length])
            
        X = np.array(X)   # (samples, seq_length, n_features)
        y = np.array(y).squeeze(-1)  # (samples, pred_length)
        
        return X, y

    def train(self, series: pd.Series,
              validation_split=0.15,
              batch_size=64,
              epochs=100,
              patience=15,
              save_steps=10,               # 每多少个epoch保存一次
              checkpoint_dir="checkpoints",
              verbose=1):
        
        if self.model is None:
            raise ValueError("必须先设置 .model 属性！请先调用模型的 build_model()")

        os.makedirs(checkpoint_dir, exist_ok=True)

        # 文件名模板：LSTM_0123.h5 这种风格
        file_pattern = os.path.join(checkpoint_dir, f"{self.name}_*.h5")
        checkpoint_prefix = os.path.join(checkpoint_dir, f"{self.name}_")

        # 寻找已存在的检查点，尝试找最大的 epoch
        existing_files = glob.glob(file_pattern)
        initial_epoch = 0
        latest_path = None

        #print(existing_files)

        if existing_files:
            epoch_list = []
            for f in existing_files:
                match = re.search(rf"{re.escape(self.name)}__(\d+)\.h5$", os.path.basename(f))
                if match:
                    epoch_list.append((int(match.group(1)), f))
                    #print("re match")

            #print(epoch_list)
            
            if epoch_list:
                epoch_list.sort(key=lambda x: x[0], reverse=True)
                max_epoch, latest_path = epoch_list[0]
                try:
                    self.model.load_weights(latest_path)
                    print(f"成功加载已有检查点：{os.path.basename(latest_path)} (epoch {max_epoch})")
                    initial_epoch = max_epoch
                except Exception as e:
                    print(f"加载权重失败：{e}")
                    print("→ 将从 epoch 0 开始全新训练")

        X, y = self.prepare_data(series)
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 自定义定期保存回调（Keras 2.10 兼容且可靠）
        class PeriodicCheckpoint(Callback):
            def __init__(self, prefix, period):
                super().__init__()
                self.prefix = prefix
                self.period = period

            def on_epoch_end(self, epoch, logs=None):
                # epoch 从0开始，所以 +1 更直观
                if (epoch + 1) % self.period == 0:
                    save_path = f"{self.prefix}_{(epoch + 1):04d}.h5"
                    self.model.save_weights(save_path, overwrite=True)
                    print(f"  已保存检查点: {os.path.basename(save_path)}")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=verbose
            )
        ]

        if save_steps > 0:
            periodic_saver = PeriodicCheckpoint(
                prefix=checkpoint_prefix,
                period=save_steps
            )
            callbacks.append(periodic_saver)

        print(f"\n训练 {self.name}")
        print(f"  • 从 epoch {initial_epoch + 1} 开始")
        print(f"  • 训练样本:{len(X_train)} | 验证样本:{len(X_val)}")
        if save_steps > 0:
            print(f"  • 每 {save_steps} 个 epoch 保存权重到 {checkpoint_dir}")
        if latest_path:
            print(f"  • 续训自: {os.path.basename(latest_path)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        final_epoch = initial_epoch + len(self.history.history['loss'])
        final_path = f"{checkpoint_prefix}_{final_epoch:04d}_final.h5"
        self.model.save_weights(final_path)
        print(f"训练完成，已保存最终权重：{os.path.basename(final_path)}")
        
        return self.history

    def predict(self, series: pd.Series, steps=None):

        if self.model is None:
            raise ValueError("模型尚未初始化")

        """使用最后一段序列进行预测"""
        if steps is None:
            steps = self.pred_length
            
        data = series[-self.seq_length:].values.reshape(-1, 1)
        scaled = self.scaler.transform(data)
        scaled = scaled.reshape(1, self.seq_length, 1)
        
        pred_scaled = self.model.predict(scaled, verbose=0)
        pred = self.scaler.inverse_transform(pred_scaled)
        
        return pred.flatten()[:steps]

    def evaluate(self, test_series: pd.Series, horizon_points=None):

        if self.model is None:
            raise ValueError("模型尚未初始化")

        """滚动窗口评估整个测试集"""
        if horizon_points is None:
            horizon_points = self.pred_length
            
        predictions = []
        actuals = []
        
        for i in tqdm(
            range(len(test_series) - self.seq_length - horizon_points + 1),
            desc="滚动评估进度"
        ):
            input_seq = test_series.iloc[i:i + self.seq_length]
            true_future = test_series.iloc[i + self.seq_length:i + self.seq_length + horizon_points]
            
            pred = self.predict(input_seq, steps=horizon_points)
            
            predictions.append(pred)
            actuals.append(true_future.values)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        print("\n=== 测试集滚动预测性能 ===")
        print(f"MAE:  {mae:,.4f}")
        print(f"RMSE: {rmse:,.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
    