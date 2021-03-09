import tensorflow as tf
from tensorflow.keras import Sequential, Input, layers, Model

from SelfAttention import SelfAttention

tf.random.set_seed(42)

def model(data):
    model = Sequential([
        Input(shape=data.shape[1]),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(26, activation="softmax")
    ])

    return model

def functional_model(data):
    inputs = Input(shape=data.shape[1])
    att = SelfAttention(8)(inputs)
    att = layers.GlobalAveragePooling1D()(att)
    x = layers.concatenate([inputs, att])
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(26, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model