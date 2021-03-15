import numpy as np
from tensorflow.keras import Sequential, layers, Input
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self, input_dim, h1=512, h2=256, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = Sequential([
            Input(shape=input_dim),
            layers.Dense(h1, activation="relu"),
            layers.Dense(h2, activation="relu"),
            layers.Dense(latent_dim, activation="relu")
        ])
        self.decoder = Sequential([
            layers.Dense(h2, activation="relu"),
            layers.Dense(h1, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

# 函数式API创建模型
# 编码器
def build_encoder(data, output_dim=32):
    encoder_input = Input(shape=data.shape[1])
    x = layers.Dense(512)(encoder_input)
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
    x = layers.Dense(output_dim)(x)
    x = layers.BatchNormalization()(x)
    encoder_output = layers.ReLU()(x)

    encoder = Model(encoder_input, encoder_output, name="encoder")
    return encoder

def build_decoder(data, input_dim=32):
    decoder_input = Input(shape=(input_dim,))
    x = layers.Dense(64)(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    decoder_output = layers.Dense(data.shape[1], activation="sigmoid")(x)

    decoder = Model(decoder_input, decoder_output, name="decoder")
    return decoder
    
def build_model(x, y):
    encoder = build_encoder(x)
    decoder = build_decoder(x)

    #print("原始encoder", id(encoder))

    input = Input(shape=x.shape[1], name="original_input")
    encoded = encoder(input)
    decoded = decoder(encoded)

    clf_pred = layers.Dense(len(np.unique(y)), activation="softmax", name="classifier")(encoded)
    model = Model(inputs=input, outputs=[clf_pred, decoded])
    return model, encoder
