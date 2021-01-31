from tensorflow.keras import Sequential, layers, Input
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = Sequential([
            Input(shape=input_dim),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(latent_dim, activation="relu")
        ])
        self.decoder = Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded