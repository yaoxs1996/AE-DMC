# 动态微簇判别新类
import sys
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from scipy.spatial import distance
from progressbar import ProgressBar

from CKMeans import CKMeans
from MicroCluster import MicroCluster as model

class DMCluster(BaseEstimator, ClassifierMixin):
    def __init__(self, nb_mirco_cluster=100, micro_clusters=[], radius_factor=1.2, new_radius=0.8):
        self.nb_micro_cluster = nb_mirco_cluster
        self.micro_clusters = micro_clusters
        self.nb_created_clusters = 0
        self.radius_factor = radius_factor      # 半径系数
        self.new_radius = new_radius        # 对于只有一个点的微簇的半径系数

    def fit(self, x, y):
        x = check_array(x, accept_sparse="csr")
        # ckmeans = CKMeans(n_clusters=self.nb_micro_cluster, random_state=42)
        # micro_cluster_labels = ckmeans.fit_predict(x, y)
        kmeans = KMeans(n_clusters=self.nb_micro_cluster)
        micro_cluster_labels = kmeans.fit_predict(x)
        x = np.column_stack((micro_cluster_labels, x))
        initial_clusters = [x[x[:, 0]==l][:, 1:] for l in set(micro_cluster_labels)]
        for cluster in initial_clusters:
            self.create_micro_cluster(cluster)

    def create_micro_cluster(self, cluster, mark="old"):
        #print(type(cluster))
        n_dim = cluster.shape[1]
        linear_sum = np.zeros(n_dim)
        squared_sum = np.zeros(n_dim)
        self.nb_created_clusters += 1
        new_m_cluster = model(nb_points=0, linear_sum=linear_sum, squared_sum=squared_sum, mark=mark)

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

    def check_fit_in_cluster(self, point, cluster: model):
        if cluster.get_weight() == 1:
            radius = sys.float_info.max
            micro_clusters = self.micro_clusters.copy()
            micro_clusters.remove(cluster)
            next_cluster = self.find_closest_cluster(point, micro_clusters)
            dist = distance.euclidean(next_cluster.get_center(), cluster.get_center())
            radius = min(self.new_radius * (dist - next_cluster.get_radius()), radius)
        else:
            radius = cluster.get_radius()

        if self.distance_to_cluster(point, cluster) < (self.radius_factor * radius):
            return True
        else:
            return False

    def predict(self, x_test):
        x = check_array(x_test, accept_sparse="csr")
        y_pred = np.empty(shape=(x.shape[0], 1), dtype=np.int8)
        bar = ProgressBar()
        for i in bar(range(x.shape[0])):
        #for i, data in enumerate(x):
            data = x[i, :]
            cluster = self.find_closest_cluster(data, self.micro_clusters)
            if(self.check_fit_in_cluster(data, cluster) and (cluster.mark=="old")):
                y_pred[i] = 1
                cluster.insert(data)
            elif (self.check_fit_in_cluster(data, cluster) and (cluster.mark=="new")):
                y_pred[i] = -1
                cluster.insert(data)
            else:
                self.create_micro_cluster(np.array([data]), mark="new")
                y_pred[i] = -1

        return y_pred
