import numpy as np

from sklearn.cluster import KMeans
from sklearn.utils import check_random_state, check_array

def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if centers.shape[0] != n_centers:
        raise ValueError(
            f"The shape of the initial centers {centers.shape} does not "
            f"match the number of clusters {n_centers}.")
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            f"The shape of the initial centers {centers.shape} does not "
            f"match the number of features of the data {X.shape[1]}.")

class CKMeans(KMeans):
    def fit(self, X, y, sample_weight=None):
        X = self._validate_data(X, accept_sparse="csr",
                                dtype=[np.float64, np.float32],
                                order="C", copy=self.copy_x,
                                accept_large_sparse=False)   

        self._check_params(X)
        random_state = check_random_state(self.random_state)

        # 验证初始矩阵
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            _validate_center_shape(X, self.n_clusters, init)

        # 