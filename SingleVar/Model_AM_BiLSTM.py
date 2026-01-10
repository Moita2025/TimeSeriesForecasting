import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from Comp_SingleVarTimeSeriesModel import SingleVarTimeSeriesModel

# 导入我们自己的模块
from Model_BiLSTM import BaseBiLSTM
from ..utils.Comp_AM import BahdanauAttention


class AM_BiLSTMForecaster(BaseBiLSTM):
    """
    带Bahdanau注意力的 BiLSTM 负荷预测模型
    """

    def __init__(self,
                 seq_length=96,
                 pred_length=96,
                 lstm_units=128,
                 attention_units=128,
                 dropout_rate=0.15,
                 learning_rate=0.001,
                 name="AM-BiLSTM"):
        super().__init__(seq_length, pred_length, lstm_units, dropout_rate)
        self.attention_units = attention_units
        self.learning_rate = learning_rate
        self.name = name

        self.trainer = SingleVarTimeSeriesModel(
            name="AM-BiLSTM",
            seq_length=seq_length,
            pred_length=pred_length
        )

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None

    def build_model(self, n_features=1):
        inputs = Input(shape=(self.seq_length, n_features), name='input')

        # BiLSTM 编码器 - 这次我们需要序列输出
        encoder_out = super().build_base(inputs, n_features, return_sequences=True)

        # 注意力层（使用最后一个时间步的隐藏状态作为query）
        last_hidden = encoder_out[:, -1, :]  # (batch, units*2) 因为是双向
        context, attention_weights = BahdanauAttention(self.attention_units)(
            last_hidden, encoder_out)

        # 融合信息
        x = Dropout(self.dropout_rate)(context)

        # 输出层 - 直接预测未来多步
        outputs = Dense(self.pred_length, activation='linear', name='output')(x)

        model = Model(inputs=inputs, outputs=outputs, name=self.name)

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, clipnorm=1.0)

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


if __name__ == "__main__":
    # 示例运行
    file_path = "../synthetic_load_15min_7200.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index.freq = '15min'

    series = df['Load_Main'].copy()

    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]

    model = AM_BiLSTMForecaster(
        seq_length=96,
        pred_length=96,
        lstm_units=128,
        attention_units=128,
        dropout_rate=0.15,
        learning_rate=0.001
    )

    model.build_model(n_features=1)
    #model.train(train_series, epochs=120, patience=18)
    model.train(train_series, epochs=1, patience=18)
    model.evaluate(test_series)