import math
import numpy as np

class MicroCluster:
    def __init__(self, nb_points=0, linear_sum=None, squared_sum=None, m=100, mark="old"):
        self.nb_points = nb_points
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.m = m

        self.radius_factor = 1.8
        self.epsilon = 0.00005
        self.min_variance = math.pow(1, -5)
        #self.radius = self.get_radius()
        self.mark = mark

    def insert(self, new_point):
        self.nb_points += 1
        for i in range(len(new_point)):
            self.linear_sum[i] += new_point[i]
            self.squared_sum[i] += math.pow(new_point[i], 2)

    def get_center(self):
        center = [self.linear_sum[i] / self.nb_points for i in range(len(self.linear_sum))]
        return center

    def get_weight(self):
        return self.nb_points

    def get_radius(self):
        ls_mean = self.linear_sum / self.nb_points
        ss_mean = self.squared_sum / self.nb_points
        variance = ss_mean - ls_mean**2
        radius = np.sqrt(np.sum(variance))
        return radius

    def set_radius(self, radius):
        self.radius = radius