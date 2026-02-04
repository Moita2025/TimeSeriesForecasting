import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from Comp_MultiVarTimeSeriesModel import MultiVarTimeSeriesModel
from ..utils.time_series_utils import get_csv

class BaseBiLSTM:
    """最基础的BiLSTM模型（不含注意力） - 供继承使用，便于与AM、TPA结合"""

    def __init__(self, seq_length=96, pred_length=96, lstm_units=128, dropout_rate=0.15):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

    def build_base(self, inputs, n_features=1, return_sequences=False):
        """构建BiLSTM主干，可被子类复用，便于添加AM或TPA"""
        x = Bidirectional(
            LSTM(
                units=self.lstm_units,
                return_sequences=return_sequences,
                recurrent_dropout=0.1 if tf.test.is_gpu_available() else 0.0
            )
        )(inputs)

        if not return_sequences:
            x = Dropout(self.dropout_rate)(x)

        return x

class BiLSTMForecaster(BaseBiLSTM):
    """
    标准双向LSTM负荷预测模型（单层BiLSTM → Dense），支持多输入多输出
    继承自 BaseBiLSTM，便于后续扩展（加Attention、TPA、残差、堆叠等）
    """
    
    def __init__(self, 
                 seq_length=96,
                 pred_length=96,
                 lstm_units=128,
                 dropout_rate=0.15,
                 learning_rate=0.001,
                 name="BiLSTM"):
        super().__init__(
            seq_length=seq_length,
            pred_length=pred_length,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate
        )
        
        self.learning_rate = learning_rate
        self.name = name

        self.trainer = MultiVarTimeSeriesModel(
            name="BiLSTM",
            seq_length=seq_length,
            pred_length=pred_length
        )
        
        self.model = None
        self.history = None

    def build_model(self, n_features=1, n_targets=1):
        """构建完整模型（可被子类覆写）"""
        inputs = Input(shape=(self.seq_length, n_features), name='input')
        
        # 使用父类的BiLSTM主干
        x = self.build_base(inputs, return_sequences=False)
        
        # 直接映射到多步多输出预测：输出扁平化后 (pred_length * n_targets)
        outputs = Dense(
            self.pred_length * n_targets, 
            activation='linear', 
            name='output_dense'
        )(x)
        
        # Reshape 到 (pred_length, n_targets)
        outputs = tf.keras.layers.Reshape((self.pred_length, n_targets))(outputs)
        
        model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, 
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        model.summary()
        self.model = model
        self.trainer.model = model
        self.trainer.history = None

        return model

    def train(self, data, feature_cols, target_cols, **kwargs):
        return self.trainer.train(data, feature_cols, target_cols, **kwargs)

    def predict(self, data: pd.DataFrame, feature_cols, target_cols, **kwargs):
        return self.trainer.predict(data, feature_cols, target_cols, **kwargs)

    def evaluate(self, test_data: pd.DataFrame, feature_cols, target_cols, **kwargs):
        return self.trainer.evaluate(test_data, feature_cols, target_cols, **kwargs)


# ── 主程序示例 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = get_csv("../synthetic_load_15min_7200.csv", "15min")
    
    # 划分训练/测试（最后20%做测试）
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # 定义列
    feature_cols = ['Load_AreaA', 'Load_AreaB', 'Load_AreaC', 'Temperature']  # 多自变量
    target_cols = ['Load_Main']    # 多因变量（假设目标也是这些，或自定义）
    
    # 创建并训练模型
    model = BiLSTMForecaster(
        seq_length=96,          # 过去1天 (96*15min=24h)
        pred_length=96,         # 预测未来1天
        lstm_units=128,
        dropout_rate=0.15,
        learning_rate=0.001
    )
    
    model.build_model(n_features=len(feature_cols), n_targets=len(target_cols))
    
    # 训练（使用训练集）
    # model.train(train_data, feature_cols, target_cols, validation_split=0.15, epochs=120, patience=18)
    model.train(train_data, feature_cols, target_cols, validation_split=0.15, epochs=1, patience=18)
    
    model.evaluate(test_data, feature_cols, target_cols)

    # 示例预测
    pred = model.predict(test_data, feature_cols, target_cols)
    print("Sample prediction shape:", pred.shape)