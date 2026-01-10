import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from Comp_SingleVarTimeSeriesModel import SingleVarTimeSeriesModel

class BaseBiLSTM:
    """最基础的BiLSTM模型（不含注意力） - 供继承使用"""

    def __init__(self, seq_length=96, pred_length=96, lstm_units=128, dropout_rate=0.15):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate

    def build_base(self, inputs, n_features=1, return_sequences=False):
        """构建BiLSTM主干，可被子类复用"""
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
    标准双向LSTM负荷预测模型（单层BiLSTM → Dense）
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

        self.trainer = SingleVarTimeSeriesModel(
            name="BiLSTM",
            seq_length=seq_length,
            pred_length=pred_length
        )
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None

    def build_model(self, n_features=1):
        """构建完整模型（可被子类覆写）"""
        inputs = Input(shape=(self.seq_length, n_features), name='input')
        
        # 使用父类的BiLSTM主干
        x = self.build_base(inputs, return_sequences=False)
        
        # 直接映射到多步预测
        outputs = Dense(
            self.pred_length, 
            activation='linear', 
            name='output_dense'
        )(x)
        
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
    # 读取数据
    file_path = "../synthetic_load_15min_7200.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index.freq = '15T'  # 15分钟
    
    series = df['Load_Main'].copy()
    
    # 划分训练/测试（最后20%做测试）
    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]
    
    # 创建并训练模型
    model = BiLSTMForecaster(
        seq_length=96,          # 过去1天
        pred_length=96,         # 预测未来1天
        lstm_units=128,
        dropout_rate=0.15,
        learning_rate=0.001
    )
    
    model.build_model(n_features=1)
    
    # 训练（使用训练集完整数据）
    #model.train(train_series, validation_split=0.15, epochs=120, patience=18)
    model.train(train_series, validation_split=0.15, epochs=1, patience=18)
    
    # 测试集评估（滚动预测）
    metrics = model.evaluate(test_series)