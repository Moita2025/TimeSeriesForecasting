import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from Comp_SingleVarTimeSeriesModel import SingleVarTimeSeriesModel

# 导入基础 BiLSTM 和 TPA 组件
from Model_BiLSTM import BaseBiLSTM
from ..utils.Comp_TPA import TemporalPatternAttention


class TPA_BiLSTMForecaster(BaseBiLSTM):
    """
    结合 Temporal Pattern Attention 的 BiLSTM 负荷预测模型
    目前为单变量实现，后续可轻松扩展到多变量（TPA 优势更明显）
    """
    def __init__(self,
                 seq_length=96,
                 pred_length=96,
                 lstm_units=128,
                 tpa_filters=8,          # TPA 中的 CNN 滤波器数量
                 tpa_kernel=3,
                 tpa_attention_len=16,   # 关注最近多少步的 pattern
                 dropout_rate=0.15,
                 learning_rate=0.001,
                 name="TPA-BiLSTM"):
        super().__init__(seq_length, pred_length, lstm_units, dropout_rate)
        
        self.tpa_filters = tpa_filters
        self.tpa_kernel = tpa_kernel
        self.tpa_attention_len = tpa_attention_len
        self.learning_rate = learning_rate
        self.name = name

        self.trainer = SingleVarTimeSeriesModel(
            name="TPA-BiLSTM",
            seq_length=seq_length,
            pred_length=pred_length
        )

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None

    def build_model(self, n_features=1):
        inputs = Input(shape=(self.seq_length, n_features), name='input')

        # BiLSTM 编码器 - 需要 return_sequences 以供 TPA 使用
        lstm_out = super().build_base(
            inputs, 
            n_features=n_features,
            return_sequences=True
        )  # (batch, seq_len, lstm_units*2)

        # 取最后一个隐藏状态作为 query
        h_last = lstm_out[:, -1, :]

        # TPA 注意力
        context, _ = TemporalPatternAttention(
            n_filters=self.tpa_filters,
            kernel_size=self.tpa_kernel,
            attention_len=self.tpa_attention_len,
            dense_units=self.lstm_units * 2
        )(lstm_out, h_last)

        # 融合 LSTM 最后状态 + TPA context
        x = Concatenate()([h_last, context])
        x = Dropout(self.dropout_rate)(x)

        # 输出层 - 直接多步预测
        outputs = Dense(self.pred_length, activation='linear', name='output')(x)

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

    def prepare_data(self, series: pd.Series):
        return self.trainer.prepare_data(series)

    def train(self, series, **kwargs):
        return self.trainer.train(series, **kwargs)

    def predict(self, series: pd.Series, **kwargs):
        return self.trainer.predict(series, **kwargs)

    def evaluate(self, test_series: pd.Series, **kwargs):
        return self.trainer.evaluate(test_series, **kwargs)

# ── 主程序示例 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    file_path = "../synthetic_load_15min_7200.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index.freq = '15min'

    series = df['Load_Main'].copy()

    train_size = int(len(series) * 0.8)
    train = series[:train_size]
    test  = series[train_size:]

    model = TPA_BiLSTMForecaster(
        seq_length=96,
        pred_length=96,
        lstm_units=128,
        tpa_filters=12,           # 可尝试 8~32
        tpa_kernel=3,             # 3,5,7 都可试
        tpa_attention_len=32,     # 可试 16~64
        dropout_rate=0.15,
        learning_rate=0.001
    )

    model.build_model(n_features=1)
    #model.train(train, epochs=120, patience=18)
    model.train(train, epochs=1, patience=18)
    model.evaluate(test)