import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape
from tensorflow.keras.models import Model

# 导入我们自己的模块
from Model_BiLSTM import BaseBiLSTM
from ..utils.Comp_AM import BahdanauAttention
from Comp_MultiVarTimeSeriesModel import MultiVarTimeSeriesModel   # ← 改用多变量 trainer
from ..utils.time_series_utils import get_csv

class AM_BiLSTMForecaster(BaseBiLSTM):
    """
    带Bahdanau注意力的 BiLSTM 负荷预测模型 - 多变量多目标版本
    """

    def __init__(self,
                 seq_length=96,
                 pred_length=96,
                 lstm_units=128,
                 attention_units=128,
                 dropout_rate=0.15,
                 learning_rate=0.001,
                 name="AM-BiLSTM-MultiVar"):
        super().__init__(seq_length, pred_length, lstm_units, dropout_rate)
        self.attention_units = attention_units
        self.learning_rate = learning_rate
        self.name = name

        # 改用多变量 trainer
        self.trainer = MultiVarTimeSeriesModel(
            name="AM-BiLSTM",
            seq_length=seq_length,
            pred_length=pred_length
        )

        self.model = None
        self.history = None

    def build_model(self, n_features=1, n_targets=1):
        inputs = Input(shape=(self.seq_length, n_features), name='input')

        # BiLSTM 编码器 - 需要序列输出给注意力机制
        encoder_out = super().build_base(inputs, n_features, return_sequences=True)

        # 注意力层（使用最后一个时间步的隐藏状态作为 query）
        last_hidden = encoder_out[:, -1, :]  # (batch, units*2) 因为是双向
        context, attention_weights = BahdanauAttention(self.attention_units)(
            last_hidden, encoder_out)

        # 融合信息 + dropout
        x = Dropout(self.dropout_rate)(context)

        # 输出层 - 支持多目标多步预测
        outputs = Dense(
            self.pred_length * n_targets,
            activation='linear',
            name='output_dense'
        )(x)

        # 恢复形状为 (batch, pred_length, n_targets)
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

    # 以下方法直接委托给多变量 trainer
    def train(self, data, feature_cols, target_cols, **kwargs):
        return self.trainer.train(data, feature_cols, target_cols, **kwargs)

    def predict(self, data: pd.DataFrame, feature_cols, target_cols, **kwargs):
        return self.trainer.predict(data, feature_cols, target_cols, **kwargs)

    def evaluate(self, test_data: pd.DataFrame, feature_cols, target_cols, **kwargs):
        return self.trainer.evaluate(test_data, feature_cols, target_cols, **kwargs)


# ── 主程序示例 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 假设你已经有 get_csv 函数或其他数据读取方式
    data = get_csv("../synthetic_load_15min_7200.csv", "15min")

    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    feature_cols = ['Load_AreaA', 'Load_AreaB', 'Load_AreaC', 'Temperature']
    target_cols = ['Load_Main']   # 假设预测两个目标

    model = AM_BiLSTMForecaster(
        seq_length=96,
        pred_length=96,
        lstm_units=128,
        attention_units=128,
        dropout_rate=0.15,
        learning_rate=0.001,
        name="AM-BiLSTM-Multi"
    )

    model.build_model(n_features=len(feature_cols), n_targets=len(target_cols))

    # 训练（epochs=1 仅作演示）
    model.train(train_data, feature_cols, target_cols,
                validation_split=0.15, epochs=1, patience=18)

    # 评估 & 预测
    model.evaluate(test_data, feature_cols, target_cols)

    pred = model.predict(test_data, feature_cols, target_cols)
    print("Prediction shape:", pred.shape)