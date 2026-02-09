import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from ..utils.Model_CNN import BaseCNN
from ..utils.Model_GRU import BaseGRU
from Comp_SingleVarTimeSeriesModel import SingleVarTimeSeriesModel

class CNN_GRUForecaster(BaseCNN, BaseGRU):
    """
    CNN提取局部特征 → GRU捕捉长期依赖 → Dense多步预测
    
    典型结构：
    Input → Conv1D(s) → Pooling → GRU/BiGRU → Dense → Output
    """
    
    def __init__(self,
                 seq_length=96,
                 pred_length=96,
                 cnn_filters=[64, 32],
                 cnn_kernel_sizes=[3, 3],
                 cnn_pool_size=2,
                 gru_units=128,
                 bidirectional_gru=True,
                 dropout_rate=0.15,
                 learning_rate=0.001,
                 name="CNN-GRU"):
        
        # 初始化父类参数
        BaseCNN.__init__(self, 
                        filters=cnn_filters, 
                        kernel_sizes=cnn_kernel_sizes, 
                        pool_size=cnn_pool_size)
        
        BaseGRU.__init__(self,
                        gru_units=gru_units,
                        dropout_rate=dropout_rate,
                        bidirectional=bidirectional_gru)
        
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.learning_rate = learning_rate
        self.name = name

        self.trainer = SingleVarTimeSeriesModel(
            name="CNN-GRU",
            seq_length=seq_length,
            pred_length=pred_length
        )
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None

    def build_model(self, n_features=1):
        inputs = Input(shape=(self.seq_length, n_features), name='input')
        
        # CNN 局部特征提取
        x = self.build_cnn_block(inputs)
        
        # GRU 时序建模
        x = self.build_gru(x, return_sequences=False)
        
        # 直接输出多步预测
        outputs = Dense(
            self.pred_length,
            activation='linear',
            name='output'
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
    
if __name__ == "__main__":
    # 示例运行
    file_path = "../synthetic_load_15min_7200.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index.freq = '15min'

    series = df['Load_Main'].copy()

    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]

    model = CNN_GRUForecaster(
        seq_length=96,
        pred_length=96,
        cnn_filters=[64, 48],
        cnn_kernel_sizes=[3, 3],
        cnn_pool_size=2,
        gru_units=128,
        bidirectional_gru=True,
        dropout_rate=0.18,
        learning_rate=0.0008,
        name="CNN_GRU_128"
    )

    model.build_model(n_features=1)
    #model.train(train_series, epochs=120, patience=18)
    model.train(train_series, epochs=1, patience=18)
    model.evaluate(test_series)