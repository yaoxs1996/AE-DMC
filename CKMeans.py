import numpy as np
import sklearn
assert sklearn.__version__ >= "0.24"

from random import random

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster._kmeans import _tolerance, _kmeans_plusplus
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms

class CKMeans:
    def __init__(self, n_clusters=8, init="k-means++", n_init=10,
                 max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _init_centroids(self, x, random_state):
        n_samples = x.shape[0]
        n_clusters = self.n_clusters

        x_squared_norm = row_norms(x, squared=True)

        centers = np.empty((n_clusters, x.shape[1]), dtype=x.dtype)
        indices = None

        if self.init == "k-means++":
            centers, indices = _kmeans_plusplus(x, n_clusters=self.n_clusters, x_squared_norms=x_squared_norm, random_state=random_state)
        elif self.init == "random":
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = x[seeds]
            indices = seeds
        else:
            print("Wrong init")
            exit(0)

        return centers, indices

    def _cloest_clusters(self, datapoint, centers):
        distances = []
        for center in centers:
            d = euclidean_distances(datapoint, center)
            distances.append(d)

        sorted_idx = sorted(range(len(distances)), key=lambda x: distances[x])
        return sorted_idx, distances
        # distances = [euclidean_distances(datapoint, center) for center in centers]
        # return sorted(range(len(distances)), key=lambda x: distances[x]), distances

    def _inertia(self, x, centers, labels):
        n_samples = x.shape[0]

        sq_dist = 0.0
        inertia = 0.0

        for i in range(n_samples):
            j = labels[i]
            sq_dist = euclidean_distances(x[i], centers[j])
            inertia += sq_dist

        return inertia

    def _center_shift(self, centers_old, centers_new):
        center_shift = [0.0] * self.n_clusters
        for j in range(self.n_clusters):
            center_shift[j] = euclidean_distances(centers_new[j], centers_old[j])

        return center_shift

    def _cop_kmeans(self, x, y, random_state):
        tol = _tolerance(x, self.tol)
        centers, indices = self._init_centroids(x, random_state)
        centers_y = y[indices]
        centers_new = np.zeros_like(centers)
        labels = np.full(x.shape[0], -1, dtype=np.int32)
        labels_old = labels.copy()
        center_shift = np.zeros(self.n_clusters, dtype=x.dtype)

        strict_convergence = False

        for _ in range(self.max_iter):
            #labels = np.full(x.shape[0], -1, dtype=np.int32)
            for i, d in enumerate(x):
                center_indices, _ = self._cloest_clusters(d, centers)
                counter = 0
                if labels[i] == -1:
                    found_clusters = False

                    while(not found_clusters) and (counter < len(center_indices)):
                        center_idx = center_indices[counter]
                        if y[i] == centers_y[center_idx]:
                            found_clusters = True
                            labels[i] = center_idx

                        counter += 1

                    if not found_clusters:
                        return None, None, None

            centers, centers_new = centers_new, centers
            center_shift = self._center_shift(centers, centers_new)

            if np.array_equal(labels, labels_old):
                strict_convergence = True
                break
            else:
                center_shift_tot = (center_shift**2).sum()
                if center_shift_tot <= self.tol:
                    break
            
            labels_old[:] = labels

        inertia = self._inertia(x, centers, labels)

        return labels, inertia, centers

    def fit(self, x, y):
        random_state = check_random_state(self.random_state)

        best_centers = None
        best_inertia = None
        best_labels = None

        for _ in range(self.n_init):
            labels, inertia, centers = self._cop_kmeans(x, y, random_state)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia

        return self