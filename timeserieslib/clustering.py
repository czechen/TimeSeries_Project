#!/bin/python3

"""
Clustering Module
-------------------


Implemented algorithms:
K-means
DBUS
"""

import numpy as np
import optimalization as opt
from matplotlib import pyplot as plt

np.random.seed(42)


class KMeans(object):
	'''
	K-means clustering
	'''
	def __init__(self,data, num_of_clusters):
		self.num_clust = num_of_clusters
		self.data = data #data matrix

	def fit(self,MAX_ITER,threshold_error):
		#initialization using k++
		self.centroids = [self.data[np.random.choice(len(self.data), size=1, replace=False)]]
		for _ in range(self.num_clust - 1):
			dist_point_to_closest_centroid = []
			for x in self.data:
				distances = [opt.l2_dist(x,self.centroids[i]) for i in range(len(self.centroids))] # distances from point x to centroids
				dist_point_to_closest_centroid.append(distances[np.argmin(distances)]**2)
			dist_point_to_closest_centroid = np.array(dist_point_to_closest_centroid/sum(dist_point_to_closest_centroid))
			new_centroid = self.data[np.random.choice(len(self.data), size=1,p=dist_point_to_closest_centroid , replace=False)]
			self.centroids += [new_centroid]
		previous_centroids = None
		
		for _ in range(MAX_ITER):
			#cluster assignment
			clusters = [[] for i in range(self.num_clust)]
			for x in self.data:
				dist_to_cetroids = [opt.l2_dist(x,self.centroids[i]) for i in range(self.num_clust)]
				closest_centroid = np.argmin(dist_to_cetroids)
				clusters[closest_centroid].append(x)
			#new centroids
			previous_centroids = self.centroids
			self.centroids = [np.mean(cluster) for cluster in clusters]
			for i, centroid in enumerate(self.centroids):
				if np.isnan(centroid).any():
					self.centroids[i] = previous_centroids[i]
			if sum([opt.l2_dist(self.centroids[i],previous_centroids[i]) for i in range(i)]) < threshold_error:
				return self.centroids
		return self.centroids

	def evaluate(self,values):
		clusters = [[] for i in range(self.num_clust)]
		for x in values:
			dist_to_cetroids = [opt.l2_dist(x,self.centroids[i]) for i in range(self.num_clust)]
			closest_centroid = np.argmin(dist_to_cetroids)
			clusters[closest_centroid].append(x)
		return clusters

#Testing
if __name__ == "__main__":
	test_data = np.zeros((40000, 4))
	test_data[0:10000, :] = 30.0
	test_data[10000:20000, :] = 60.0
	test_data[20000:30000, :] = 90.0
	test_data[30000:, :] = 120.0
	clustering = KMeans(test_data,4)
	print(clustering.fit(10,10-4))
	print(clustering.evaluate([10,20,40,100]))
