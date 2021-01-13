import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight

from sklearn.cluster._kmeans import _init_centroids
from sklearn.cluster._k_means_lloyd import lloyd_iter_chunked_sparse, lloyd_iter_chunked_dense
from sklearn.cluster._k_means_fast import _inertia_dense, _inertia_sparse

def _kmeans_single_lloyd(X, sample_weight, n_clusters, max_iter=300,
                         init='k-means++', verbose=False, x_squared_norms=None,
                         random_state=None, tol=1e-4, n_threads=1):
    """k-means lloyd算法的一次运行，假定所有准备已事先完成。

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        聚类的观测值。如果是稀疏矩阵，必须为CSR格式。

    sample_weight : ndarray of shape (n_samples,)
        X中每一个观测值的权重。

    n_clusters : int
        形成的簇的数量，同时也是生成的形心的数量。

    max_iter : int, default=300
        k-means算法运行的最大迭代次数。

    init : {'k-means++', 'random', ndarray, callable}, default='k-means++'
        初始化方法：

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    verbose : bool, default=False
        Verbosity mode

    x_squared_norms : ndarray of shape(n_samples,), default=None
        预计算 x_squared_norms.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        在k-means的最后一次迭代发现的形心。

    label : ndarray of shape (n_samples,)
        label[i] 是第i个观测值最接近的形心的编码或索引。

    inertia : float
        inertia 准则的最终值（训练集中所有的观测值到最近形心的平方距离和）。

    n_iter : int
        迭代运行的次数。
    """
    random_state = check_random_state(random_state)
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms)

    if verbose:
        print("Initialization complete")

    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    strict_convergence = False

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubsciption.
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            lloyd_iter(X, sample_weight, x_squared_norms, centers, centers_new,
                       weight_in_clusters, labels, center_shift, n_threads)

            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels)
                print("Iteration {0}, inertia {1}" .format(i, inertia))

            if np.array_equal(labels, labels_old):
                # First check the labels for strict convergence.
                if verbose:
                    print(f"Converged at iteration {i}: strict convergence.")
                strict_convergence = True
                break
            else:
                # No strict convergence, check for tol based convergence.
                center_shift_tot = (center_shift**2).sum()
                if center_shift_tot <= tol:
                    if verbose:
                        print(f"Converged at iteration {i}: center shift "
                              f"{center_shift_tot} within tolerance {tol}.")
                    break

            centers, centers_new = centers_new, centers
            labels_old[:] = labels

        if not strict_convergence:
            # rerun E-step so that predicted labels match cluster centers
            lloyd_iter(X, sample_weight, x_squared_norms, centers, centers,
                       weight_in_clusters, labels, center_shift, n_threads,
                       update_centers=False)

    inertia = _inertia(X, sample_weight, centers, labels)

    return labels, inertia, centers, i + 1