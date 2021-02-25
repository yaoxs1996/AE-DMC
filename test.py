import random
import pandas as pd
import numpy as np
from random import sample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score

from DMCluster import DMCluster

random.seed(42)

def split_novel(data, labels, n_known, n_novel):
    classes = np.unique(labels)
    train_set = sample(classes.tolist(), n_known)
    remain_set = list(set(classes) - set(train_set))
    test_set = sample(remain_set, n_novel)

    train_index = [i in train_set for i in labels]
    test_index = [i in test_set for i in labels]
    train_ds = data[train_index, :]
    #train_labels = labels[train_index]
    novel_ds = data[test_index, :]
    #novel_labels = labels[test_index]

    train_ds, extra_ds = train_test_split(train_ds, test_size=0.4, random_state=42)
    test_ds = np.r_[extra_ds, novel_ds]
    y_train = np.ones(train_ds.shape[0])
    y_test = np.ones(test_ds.shape[0]) * -1
    #test_labels = np.r_[extra_labels, novel_labels]

    return train_ds, y_train, test_ds, y_test

def letter_exp():
    letter = pd.read_csv("./dataset/letter_csv.csv")
    data = letter.to_numpy()
    labels = data[:, -1]
    data = data[:, :-1]
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    train_ds, train_labels, test_ds, test_labels = split_novel(data, labels, 10, 8)

    model = DMCluster(nb_mirco_cluster=100)
    print("开始训练")
    model.fit(train_ds, train_labels)
    print("开始检测")
    y_pred = model.predict(test_ds)
    print(f1_score(test_labels, y_pred))
    print(confusion_matrix(test_labels, y_pred))

if __name__ == "__main__":
    letter_exp()