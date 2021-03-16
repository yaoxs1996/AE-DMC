import random
import pandas as pd
import numpy as np
from random import sample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import losses

import utils
import AutoEncoder
from DMCluster import DMCluster
from AutoEncoder import Autoencoder

random.seed(42)

def letter_exp():
    x, y = utils.load_data("letter_csv")
    x_train, y_train, x_test, y_test = utils.split_train_test(x, y, 18)
    x_train, y_train, x_val, y_val = utils.split_train_val(x_train, y_train)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # autoencoder = Autoencoder(input_dim=x_train.shape[1], h1=12, h2=8, latent_dim=4)
    # autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())
    # autoencoder.fit(x_train, x_train, epochs=5, shuffle=True, validation_data=(x_test, x_test))

    # x_train = autoencoder.encoder(x_train)
    # x_test = autoencoder.encoder(x_test)

    autoencoder, encoder = AutoEncoder.build_model(x_train, y_train)
    autoencoder.compile(optimizer="adam",
        loss={
            "decoder": losses.MeanSquaredError(),
            "classifier": losses.SparseCategoricalCrossentropy(from_logits=True),
        },
        metrics={
            "classifier": "accuracy"
        })
    autoencoder.fit(x_train, {"classifier": y_train, "decoder": x_train}, epochs=20, batch_size=32)

    x_train = encoder(x_train)
    x_test = encoder(x_test)

    model = DMCluster(nb_mirco_cluster=100, radius_factor=1.5, new_radius=0.9)
    print("开始训练")
    model.fit(x_train, y_train)
    print("开始检测")
    y_pred = model.predict(x_val)
    print(f1_score(y_val, y_pred, average="micro"))
    print(confusion_matrix(y_test, y_pred))

def test():
    x, y = utils.load_data(name="letter_csv")
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    model, encoder = AutoEncoder.build_model(x, y)
    #print("调用encoder", id(encoder))
    model.compile(optimizer="adam",
        loss={
            "decoder": losses.MeanSquaredError(),
            "classifier": losses.SparseCategoricalCrossentropy(from_logits=True),
        },
        metrics={
            "classifier": "accuracy"
        })

    model.fit(x, {"classifier": y, "decoder": x}, epochs=20, batch_size=32)

if __name__ == "__main__":
    letter_exp()
    # test()