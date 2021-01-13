import numpy as np

from sklearn.utils.extmath import row_norms

CHUNK_SIZE = 256

def lloyd_iter_chunked_dense(X, sample_weight, x_squared_norms, centers_old,
                             centers_new, weight_in_clusters, labels, center_shift,
                             n_threads, update_centers=True):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_clusters = centers_new.shape[0]

    n_samples_chunk = CHUNK_SIZE if n_samples > CHUNK_SIZE else n_samples
    n_chunks = n_samples // n_samples_chunk
    n_samples_rem = n_samples % n_samples_chunk

    centers_squared_norms = row_norms(centers_old, squared=True)

    # 计算余下的所有的数据块个数
    n_chunks += n_samples != n_chunks * n_samples_chunk

    # 线程数不应大于数据块个数
    n_threads = min(n_threads, n_chunks)

    if update_centers:
        centers_new = np.full(shape=(n_clusters, n_features), fill_value=0, dtype=float)
        weight_in_clusters = np.full(shape=(n_clusters,), fill_value=0, dtype=float)

    return centers_new, weight_in_clusters, labels, center_shift
