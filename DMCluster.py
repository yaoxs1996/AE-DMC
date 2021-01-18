# 动态微簇判别新类
import sys
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array
from scipy.spatial import distance

from CKMeans import CKMeans
from MicroCluster import MicroCluster as model

class DMCluster(BaseEstimator, ClassifierMixin):
    def __init__(self, nb_mirco_cluster=100, micro_clusters=[]):
        self.nb_micro_cluster = nb_mirco_cluster
        self.micro_clusters = micro_clusters
        self.nb_created_clusters = 0

    def fit(self, x, y):
        x = check_array(x, accept_sparse="csr")
        ckmeans = CKMeans(n_clusters=self.nb_micro_cluster, random_state=42)
        micro_cluster_labels = ckmeans.fit_predict(x, y)
        x = np.column_stack((micro_cluster_labels, x))
        initial_clusters = [x[x[:, 0]==l][:, 1:] for l in set(micro_cluster_labels)]
        for cluster in initial_clusters:
            self.create_micro_cluster(cluster)

    def create_micro_cluster(self, cluster):
        n_dim = cluster.shape[1]
        linear_sum = np.zeros(n_dim)
        squared_sum = np.zeros(n_dim)
        self.nb_created_clusters += 1
        new_m_cluster = model(nb_points=0, linear_sum=linear_sum, squared_sum=squared_sum)

        for point in cluster:
            new_m_cluster.insert(point)

        self.micro_clusters.append(new_m_cluster)

    def distance_to_cluster(self, point, cluster):
        return distance.euclidean(point, cluster.get_center())

    def find_closest_cluster(self, point, micro_clusters):
        min_distance = sys.float_info.max
        closest_cluster = None
        for cluster in micro_clusters:
            distance_cluster = self.distance_to_cluster(point, cluster)
            if distance_cluster < min_distance:
                min_distance = distance_cluster
                closest_cluster = cluster

        return closest_cluster

    def check_fit_in_cluster(self, point, cluster):
        if cluster.get_weight() == 1:
            radius = sys.float_info.max