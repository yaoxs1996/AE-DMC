import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import row_norms

from sklearn.cluster._kmeans import _kmeans_single_lloyd, _kmeans_single_elkan
from sklearn.cluster._kmeans import _validate_center_shape

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

        # 减去x的均值以获得更准确的距离计算
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, '__array__'):
                init -= X_mean

        # 预计算数据点的平方范数
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "full":
            kmeans_single = _kmeans_single_lloyd
        else:
            kmeans_single = _kmeans_single_elkan

        best_labels, best_inertia, best_centers = None, None, None

        # 初始化kmeans运行的种子
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self._n_init)

        for seed in seeds:
            # 运行一次k-means
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, sample_weight, self.n_clusters, max_iter=self.max_iter,
                init=init, verbose=self.verbose, tol=self._tol,
                x_squared_norms=x_squared_norms, random_state=seed,
                n_threads=self._n_threads)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning, stacklevel=2)

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self