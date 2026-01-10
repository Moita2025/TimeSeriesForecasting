import tensorflow as tf
from tensorflow.keras import layers


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Bahdanau Attention (Additive Attention)
    用于序列到序列任务，也可用于时间序列多步预测的注意力
    输入：encoder_outputs (batch, seq_len, units)
    返回：context_vector (batch, units), attention_weights (batch, seq_len, 1)
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)           # 对 query
        self.W2 = layers.Dense(units)           # 对 values (encoder hidden states)
        self.V = layers.Dense(1)                # 打分函数 -> scalar

    def call(self, query, values):
        """
        query:   通常是 decoder 的上一步隐藏状态 (batch, 1, units) 或 (batch, units)
        values:  encoder 的所有隐藏状态 (batch, seq_len, units)
        """
        # 扩展 query 维度使其能广播
        query_with_time_axis = tf.expand_dims(query, 1)  # (batch, 1, units)

        # score = tanh(W1*query + W2*value)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)
        ))  # (batch, seq_len, 1)

        # attention weights
        attention_weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len, 1)

        # context vector = sum(attention_weights * values)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, units)

        return context_vector, attention_weights