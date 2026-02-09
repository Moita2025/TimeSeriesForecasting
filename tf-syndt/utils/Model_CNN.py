import tensorflow as tf

class BaseCNN:
    """最基础的CNN特征提取模块 - 供继承使用"""
    def __init__(self, filters=[64, 32], kernel_sizes=[3, 3], pool_size=2):
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.pool_size = pool_size

    def build_cnn_block(self, inputs):
        """
        简单两层1D-CNN + MaxPooling
        返回: (batch, timesteps//(pool_size**n), channels)
        """
        x = inputs
        
        for f, k in zip(self.filters, self.kernel_sizes):
            x = tf.keras.layers.Conv1D(
                filters=f,
                kernel_size=k,
                padding='same',
                activation='relu'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)(x)
            
        return x