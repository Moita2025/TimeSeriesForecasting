import tensorflow as tf
from tensorflow.keras import layers


class TemporalPatternAttention(tf.keras.layers.Layer):
    """
    Temporal Pattern Attention (TPA) 核心模块
    参考论文: Temporal Pattern Attention for Multivariate Time Series Forecasting (2019)
    
    主要步骤:
    1. 对 LSTM 输出的隐藏序列 h 使用多组 1D-CNN 提取不同尺度的 temporal pattern
    2. 得到 pattern 表征 v (batch, seq_len, n_filters)
    3. 使用当前时间步隐藏状态 h_t 做 attention，关注哪些 pattern 重要
    4. 加权求和得到 context vector
    """
    def __init__(self, 
                 n_filters=8,           # 滤波器数量（模式种类）
                 kernel_size=3,         # 卷积核大小（感受野）
                 attention_len=16,      # attention 考虑多长历史（论文常用16）
                 dense_units=128):
        super(TemporalPatternAttention, self).__init__()
        
        self.n_filters = n_filters
        self.attention_len = attention_len
        self.conv = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='same',          # 保持序列长度
            activation='relu'
        )
        
        # 用于生成 attention score 的全连接
        self.fc_query = layers.Dense(dense_units, activation=None)
        self.fc_key   = layers.Dense(dense_units, activation=None)
        self.fc_out   = layers.Dense(1, activation='sigmoid')  # 生成注意力权重

    def call(self, h, h_last):
        """
        Args:
            h:       LSTM 输出的隐藏序列 (batch, seq_len, units)
            h_last:  最后一个时间步的隐藏状态 (batch, units)  ← 作为 query
            
        Returns:
            context: 加权后的特征 (batch, n_filters)
            weights: attention 权重 (batch, seq_len, 1)
        """
        # 1. 通过 CNN 提取 temporal patterns
        #    shape: (batch, seq_len, n_filters)
        patterns = self.conv(h)
        
        # 取最后 attention_len 个时间步（节省计算，也符合论文做法）
        if self.attention_len < tf.shape(patterns)[1]:
            patterns = patterns[:, -self.attention_len:, :]
        
        # 2. 计算 attention score
        # query: h_last → (batch, 1, units)
        query = tf.expand_dims(h_last, axis=1)
        
        # 简单 dot-product + MLP 风格的 attention
        energy = tf.nn.tanh(
            self.fc_query(query) + self.fc_key(patterns)
        )  # (batch, attention_len, dense_units)
        
        attention_weights = self.fc_out(energy)  # (batch, attention_len, 1)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        
        # 3. 加权求和得到 context
        context = tf.reduce_sum(patterns * attention_weights, axis=1)  # (batch, n_filters)
        
        return context, attention_weights