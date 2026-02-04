import tensorflow as tf

class BaseGRU:
    """最基础的GRU模块 - 供继承使用"""
    
    def __init__(self, gru_units=128, dropout_rate=0.15, bidirectional=True):
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional

    def build_gru(self, inputs, return_sequences=False):
        """可以选择单向或双向GRU"""
        gru = tf.keras.layers.GRU(
            units=self.gru_units,
            return_sequences=return_sequences,
            recurrent_dropout=0.1 if tf.test.is_gpu_available() else 0.0
        )
        
        if self.bidirectional:
            x = tf.keras.layers.Bidirectional(gru)(inputs)
        else:
            x = gru(inputs)
            
        if not return_sequences:
            x = tf.keras.layers.Dropout(self.dropout_rate)(x)
            
        return x