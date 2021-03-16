import pandas as pd
import numpy as np
from random import sample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(name):
    data = pd.read_csv("./dataset/" + name + ".csv")
    data = data.to_numpy()
    x = data[:, :-1]
    y = data[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    x = x.astype(np.float64)
    y = y.astype(np.int)

    return x, y

def split_train_test(x, y, omega):
    n_classes = np.unique(y).tolist()
    train_set = sample(n_classes, omega)
    test_set = list(set(n_classes) - set(train_set))

    train_index = [i in train_set for i in y]
    test_index = [i in test_set for i in y]

    x_train = x[train_index, :]
    y_train = y[train_index]

    x_test = x[test_index, :]
    y_test = y[test_index]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.4, shuffle=True, stratify=y_train)

    x_test = np.r_[x_test, x_val]
    y_test = np.r_[np.ones(x_test.shape[0])*(-1), np.ones(x_val.shape[0])]

    return x_train, y_train, x_test, y_test

def split_train_val(x, y):
    n_classes = np.unique(y).to_list()
    n_known = int(n_classes / 2 + 0.5)

    train_set = sample(n_classes, n_known)
    val_set = list(set(n_classes) - set(train_set))

    train_index = [i in train_set for i in y]
    val_index = [i in val_set for i in y]

    x_train = x[train_index, :]
    y_train = y[train_index]

    x_val = x[val_index, :]
    y_val = y[val_index]

    x_train, x_extra, y_train, _ = train_test_split(x_train, y_train, test_size=0.4, shuffle=True, stratify=y_train)

    x_val = np.r_[x_extra, x_val, x_extra]
    y_val = np.r_[np.ones(x_extra.shape[0]), np.ones(x_val.shape[0])*(-1), np.ones(x_extra.shape[0])]

    return x_train, y_train, x_val, y_val