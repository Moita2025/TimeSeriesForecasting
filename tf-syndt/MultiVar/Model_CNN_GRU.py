import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape
from tensorflow.keras.models import Model

# 导入基础模块
from ..utils.Model_CNN import BaseCNN
from ..utils.Model_GRU import BaseGRU

# 导入多变量通用训练器（你提供的文件）
from Comp_MultiVarTimeSeriesModel import MultiVarTimeSeriesModel
from ..utils.time_series_utils import get_csv

class CNN_GRUForecaster(BaseCNN, BaseGRU):
    """
    多变量 CNN-GRU 预测模型（组合式设计）
    
    结构：Input → CNN 局部特征提取 → GRU/BiGRU 时序建模 → Dense 多步多目标预测
    所有数据处理、训练、预测、评估逻辑委托给 MultiVarTimeSeriesModel
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
                 name="CNN_GRU_MultiVar"):
        
        # 初始化父类
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

        # 使用统一的 MultiVarTimeSeriesModel 作为 trainer
        self.trainer = MultiVarTimeSeriesModel(
            name=name,
            seq_length=seq_length,
            pred_length=pred_length
        )

        self.model = None

    def build_model(self, n_features: int, n_targets: int = 1):
        """
        构建 CNN-GRU 模型
        :param n_features: 输入特征数
        :param n_targets:  预测目标数（支持多目标）
        """
        inputs = Input(shape=(self.seq_length, n_features), name='input')
        
        # 1. CNN 提取局部时序模式
        x = self.build_cnn_block(inputs)                    # (batch, reduced_steps, channels)
        
        # 2. GRU 捕捉长期依赖（不返回序列，只取最终隐藏状态）
        x = self.build_gru(x, return_sequences=False)       # (batch, gru_units 或 2*gru_units)
        
        # 3. Dropout 正则化
        x = Dropout(self.dropout_rate)(x)
        
        # 4. 全连接输出层 → 多步多目标
        x = Dense(self.pred_length * n_targets, activation='linear')(x)
        
        # 5. Reshape 为 (pred_length, n_targets)
        outputs = Reshape((self.pred_length, n_targets))(x)

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
        
        # 将构建好的模型注入 trainer
        self.model = model
        self.trainer.model = model
        
        return model

    # ── 统一的多变量接口（直接转发给 trainer） ─────────────────────────────────────
    def train(self, data: pd.DataFrame, feature_cols: list, target_cols: list, **kwargs):
        """训练模型"""
        return self.trainer.train(data, feature_cols, target_cols, **kwargs)

    def predict(self, data: pd.DataFrame, feature_cols: list, target_cols: list, **kwargs):
        """预测未来 pred_length 步（返回 numpy array，shape: pred_length, n_targets）"""
        return self.trainer.predict(data, feature_cols, target_cols, **kwargs)

    def evaluate(self, test_data: pd.DataFrame, feature_cols: list, target_cols: list, **kwargs):
        """滚动评估（支持详细指标输出）"""
        return self.trainer.evaluate(test_data, feature_cols, target_cols, **kwargs)


# ── 主程序示例（与你之前的完全一致） ───────────────────────────────────────────────
if __name__ == "__main__":

    data = get_csv("../synthetic_load_15min_7200.csv", "15min")
    
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    feature_cols = ['Load_AreaA', 'Load_AreaB', 'Load_AreaC', 'Temperature']
    target_cols = ['Load_Main']                     # 单目标
    # target_cols = ['Load_Main', 'Load_Sub1']     # 多目标示例

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
        name="CNN_GRU_Multivariate"
    )

    # 构建模型
    model.build_model(n_features=len(feature_cols), n_targets=len(target_cols))

    # 训练（演示用 epochs=1）
    model.train(train_data, feature_cols, target_cols,
                validation_split=0.15, epochs=1, patience=18)

    # 滚动评估（会输出详细 MAE/RMSE/MAPE）
    model.evaluate(test_data, feature_cols, target_cols)

    # 预测
    pred = model.predict(test_data, feature_cols, target_cols)
    print("Prediction shape:", pred.shape)   # (96, n_targets)