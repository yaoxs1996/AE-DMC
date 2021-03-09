import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ResNet

def load_data():
    letter = pd.read_csv("./dataset/letter_csv.csv")
    data = letter.to_numpy()
    y = data[:, -1]
    x = data[:, :-1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    x = x.astype(np.float64)
    y = y.astype(np.int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return (x_train, y_train), (x_test, y_test)

def train(model_type="seq"):
    (x_train, y_train), (x_test, y_test) = load_data()

    if model_type == "seq":
        print("序列式模型")
        model = ResNet.model(x_train)
    else:
        print("函数式模型")
        model = ResNet.functional_model(x_train)
    
    model.compile(optimizer="adam", loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=20, shuffle=True, validation_data=(x_test, y_test))
    #model.fit(x_train, y_train, epochs=20, shuffle=True)
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    # score = tf.nn.softmax(predictions)
    # y_pred = np.argmax(score, axis=1)
    print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    train(model_type="func")
