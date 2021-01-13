from sklearn.datasets import load_iris
from CKMeans import CKMeans
import numpy as np

x, y = load_iris(return_X_y=True)
ckmeans = CKMeans(n_clusters=20, random_state=42)
ckmeans.fit(x, y)

labels = ckmeans.labels_
print(np.unique(labels, return_counts=True))