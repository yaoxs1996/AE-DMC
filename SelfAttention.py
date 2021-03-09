import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer

class SelfAttention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                      shape=(3, input_shape[-1], self.output_dim),
                                      initializer="uniform",
                                      trainable=True)

        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        #print("x.shape", x.shape)
        x = tf.expand_dims(x, axis=1)
        #print("x.shape", x.shape)

        WQ = backend.dot(x, self.kernel[0])
        WK = backend.dot(x, self.kernel[1])
        WV = backend.dot(x, self.kernel[2])

        QK = backend.batch_dot(WQ, backend.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (self.output_dim ** 0.5)
        QK = backend.softmax(QK)

        V = backend.batch_dot(QK, WV)

        return V
