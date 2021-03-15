import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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