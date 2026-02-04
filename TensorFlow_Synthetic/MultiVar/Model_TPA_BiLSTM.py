import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Reshape
from tensorflow.keras.models import Model

# 导入我们自己的模块
from Model_BiLSTM import BaseBiLSTM
from ..utils.Comp_TPA import TemporalPatternAttention
from Comp_MultiVarTimeSeriesModel import MultiVarTimeSeriesModel  # 多变量 trainer
from ..utils.time_series_utils import get_csv

class TPA_BiLSTMForecaster(BaseBiLSTM):
    """
    结合 Temporal Pattern Attention 的 BiLSTM 预测模型 - 多变量多目标版本
    TPA 在多变量场景下优势更明显（能捕捉不同特征的局部模式）
    """
    def __init__(self,
                 seq_length=96,
                 pred_length=96,
                 lstm_units=128,
                 tpa_filters=12,
                 tpa_kernel=3,
                 tpa_attention_len=32,
                 dropout_rate=0.15,
                 learning_rate=0.001,
                 name="TPA-BiLSTM-MultiVar"):
        super().__init__(seq_length, pred_length, lstm_units, dropout_rate)
        
        self.tpa_filters = tpa_filters
        self.tpa_kernel = tpa_kernel
        self.tpa_attention_len = tpa_attention_len
        self.learning_rate = learning_rate
        self.name = name

        # 使用多变量 trainer
        self.trainer = MultiVarTimeSeriesModel(
            name="TPA-BiLSTM",
            seq_length=seq_length,
            pred_length=pred_length
        )

        self.model = None
        self.history = None

    def build_model(self, n_features=1, n_targets=1):
        inputs = Input(shape=(self.seq_length, n_features), name='input')

        # BiLSTM 需要 return_sequences 以供 TPA 使用
        lstm_out = super().build_base(
            inputs,
            n_features=n_features,
            return_sequences=True
        )  # (batch, seq_len, lstm_units*2)

        # 最后一个隐藏状态作为 TPA 的 query
        h_last = lstm_out[:, -1, :]  # (batch, lstm_units*2)

        # TPA 注意力层
        context, attention_weights = TemporalPatternAttention(
            n_filters=self.tpa_filters,
            kernel_size=self.tpa_kernel,
            attention_len=self.tpa_attention_len,
            dense_units=self.lstm_units * 2  # 与 BiLSTM 输出维度匹配
        )(lstm_out, h_last)  # context: (batch, tpa_filters)

        # 融合 BiLSTM 最后状态 + TPA 提取的模式特征
        x = Concatenate()([h_last, context])
        x = Dropout(self.dropout_rate)(x)

        # 输出层：支持多目标多步预测
        outputs = Dense(
            self.pred_length * n_targets,
            activation='linear',
            name='output_dense'
        )(x)

        # 恢复为 (pred_length, n_targets) 形状
        outputs = Reshape((self.pred_length, n_targets))(outputs)

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

    # 接口统一为多变量风格
    def train(self, data, feature_cols, target_cols, **kwargs):
        return self.trainer.train(data, feature_cols, target_cols, **kwargs)

    def predict(self, data: pd.DataFrame, feature_cols, target_cols, **kwargs):
        return self.trainer.predict(data, feature_cols, target_cols, **kwargs)

    def evaluate(self, test_data: pd.DataFrame, feature_cols, target_cols, **kwargs):
        return self.trainer.evaluate(test_data, feature_cols, target_cols, **kwargs)


# ── 主程序示例 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = get_csv("../synthetic_load_15min_7200.csv", "15min")

    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    feature_cols = ['Load_AreaA', 'Load_AreaB', 'Load_AreaC', 'Temperature']
    target_cols = ['Load_Main']                     # 单目标
    # target_cols = ['Load_Main', 'Load_Sub1']     # 多目标示例

    model = TPA_BiLSTMForecaster(
        seq_length=96,
        pred_length=96,
        lstm_units=128,
        tpa_filters=16,
        tpa_kernel=3,
        tpa_attention_len=32,
        dropout_rate=0.15,
        learning_rate=0.001
    )

    model.build_model(n_features=len(feature_cols), n_targets=len(target_cols))

    # 训练（仅演示）
    model.train(train_data, feature_cols, target_cols,
                validation_split=0.15, epochs=1, patience=18)

    model.evaluate(test_data, feature_cols, target_cols)

    pred = model.predict(test_data, feature_cols, target_cols)
    print("Prediction shape:", pred.shape)  # (96, n_targets)