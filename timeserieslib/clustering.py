#!/bin/python3

"""
Clustering Module
-------------------


Implemented algorithms:
K-means DONE
DBSCAN DONE
OPTICS TODO

Others:
finding optimal epsilon for DBSCAN using k-nearest neighbor graph
https://stackoverflow.com/questions/15050389/estimating-choosing-optimal-hyperparameters-for-dbscan
DBSCAN optimalization using R(*)-tree
"""
import sys,os
import numpy as np
sys.path.append(os.getcwd())
import timeserieslib.optimalization as opt
import timeserieslib.exceptions as exc
from matplotlib import pyplot as plt


np.random.seed(42)


class KMeans(object):
	'''
	K-means clustering

	'''
	def __init__(self,data, num_of_clusters):
		self.num_clust = num_of_clusters
		self.data = data #data matrix
		self.fitted = None

	def fit(self,MAX_ITER,threshold_error):
		#initialization using k++
		self.centroids = [self.data[np.random.choice(len(self.data), size=1, replace=False)]]
		for _ in range(self.num_clust - 1):
			dist_point_to_closest_centroid = []
			for x in self.data:
				distances = [np.linalg.norm(x,self.centroids[i]) for i in range(len(self.centroids))] # distances from point x to centroids
				dist_point_to_closest_centroid.append(distances[np.argmin(distances)]**2)
			dist_point_to_closest_centroid = np.array(dist_point_to_closest_centroid/sum(dist_point_to_closest_centroid))
			new_centroid = self.data[np.random.choice(len(self.data), size=1,p=dist_point_to_closest_centroid , replace=False)]
			self.centroids += [new_centroid]
		previous_centroids = None
		
		for _ in range(MAX_ITER):
			#cluster assignment
			clusters = [[] for i in range(self.num_clust)]
			for x in self.data:
				dist_to_cetroids = [np.linalg.norm(x,self.centroids[i]) for i in range(self.num_clust)]
				closest_centroid = np.argmin(dist_to_cetroids)
				clusters[closest_centroid].append(x)
			#new centroids
			previous_centroids = self.centroids
			self.centroids = [np.mean(cluster) for cluster in clusters]
			for i, centroid in enumerate(self.centroids):
				if np.isnan(centroid).any():
					self.centroids[i] = previous_centroids[i]
			if sum([np.linalg.norm(self.centroids[i],previous_centroids[i]) for i in range(i)]) < threshold_error:
				return self.centroids
		self.fitted = True
		return self.centroids

	def evaluate(self,values):
		'''
		Each elements in values is assigned a clased based on the clustering model trained above
		'''
		if not self.fitted:
			raise exc.ModelNotTrained('Cannot evaluate. Model is yet to be trained.')
		clusters = [[] for i in range(self.num_clust)]
		for x in values:
			dist_to_cetroids = [np.linalg.norm(x,self.centroids[i]) for i in range(self.num_clust)]
			closest_centroid = np.argmin(dist_to_cetroids)
			clusters[closest_centroid].append(x)
		return clusters

class DBSCAN_point(object):
	'''
	Adds multiple attributes to data_point.
	Used for DBSCAN below.
	'''
	def __init__(self,data_point, label = -1, neighbours = None,center = False):
		self.point = data_point
		self.label = label
		self.neighbours = neighbours
		self.center = center


class DBSCAN2(object):
	'''
	DBSCAN clustering 
	
	standard DBSCAN implementation with naive RangeQuery
	'''
	def __init__(self,data_matrix,minPts,epsilon,distance_function = 2):
		self.epsilon = epsilon
		self.minPts = minPts
		self.data = [DBSCAN_point(x) for x in data_matrix]
		self.dist = distance_function
		self.num_of_clusters = 0

	def find_neighbours(self,x):
		neigh = [] #neighbours of x
		for y in self.data:
			if np.linalg.norm(x.point-y.point,ord = self.dist) <= self.epsilon:
				neigh.append(y)
		return neigh


	def fit(self):
		for x in self.data:
			if x.label == -1:
				x.neighbours = self.find_neighbours(x)
				if len(x.neighbours) < self.minPts:
					x.label = 0
					continue
				self.num_of_clusters += 1
				x.label = self.num_of_clusters
				x.center = True
				S = x.neighbours
				for y in S:
					if y.label == 0:
						y.label = self.num_of_clusters
					if y.label == -1:
						y.label = self.num_of_clusters
						y.neighbours = self.find_neighbours(y)
						if len(y.neighbours) >= self.minPts:
							y.center = True
							for z in y.neighbours:
								if z not in S:
									S.append(z)
					else:
						continue
			else:
				continue
		return 1

	def plot(self,DPI):
		#plotting clustering results for 2D data
		X,Y,core_indices = [],[],[]
		for x in self.data:
			X.append(list(x.point))
			Y.append(x.label)
			if x.center:
				core_indices.append(self.data.index(x))
		X,Y = np.array(X),np.array(Y)
		unique_labels = set(Y)

		core_samples_mask = np.zeros_like(Y, dtype=bool)
		core_samples_mask[core_indices] = True
		
		colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
		for k, col in zip(unique_labels, colors):
			if k == 0:
				# Black used for noise.
				col = [0, 0, 0, 1]

			class_member_mask = Y == k

			xy = X[class_member_mask & core_samples_mask]
			plt.plot(
				xy[:, 0],
				xy[:, 1],
				"o",
				markerfacecolor=tuple(col),
				markeredgecolor="k",
				markersize=14,
			)

			xy = X[class_member_mask & ~core_samples_mask]
			plt.plot(
				xy[:, 0],
				xy[:, 1],
				"o",
				markerfacecolor=tuple(col),
				markeredgecolor="k",
				markersize=6,
			)
		plt.title(f"Estimated number of clusters: {self.num_of_clusters}")
		plt.show()
		
	def predict(self,points):
		#not tested
		for p in points:
			p = DBSCAN_point(p,label = 'Noise')
			p.neighbours = self.find_neighbours(p)
			for q in p.neighbours:
				if q.label != 'Noise':
					p.label = q.label
					break
		return points

#Testing
if __name__ == "__main__":
	#Testing k-means
	'''
	test_data = np.zeros((40000, 4))
	test_data[0:10000, :] = 30.0
	test_data[10000:20000, :] = 60.0
	test_data[20000:30000, :] = 90.0
	test_data[30000:, :] = 120.0
	clustering = KMeans(test_data,4)
	print(clustering.fit(10,10-4))
	print(clustering.evaluate([10,20,40,100]))
	'''
	#Testing DBSCAN
	'''
	test_data = np.zeros((4000, 1))
	test_data[0:1000, :] = 30.0
	test_data[1000:2000, :] = 60.0
	test_data[2000:3000, :] = 90.0
	test_data[3000:, :] = 120.0
	clustering = DBSCAN(test_data,4,10)
	clustering.fit()
	clustering.plot(300)
	'''
	from sklearn.datasets import make_blobs
	from sklearn.preprocessing import StandardScaler
	from sklearn.cluster import DBSCAN


	centers = [[1, 1], [-1, -1], [1, -1]]
	X, labels_true = make_blobs(
		n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

	X = StandardScaler().fit_transform(X)
	data = X
	db = DBSCAN2(data_matrix = data,minPts = 10,epsilon = 0.3,distance_function = 2)
	db.fit()
	db.plot(300)
	plt.scatter(X[:, 0], X[:, 1])
	plt.show()

	db = DBSCAN(eps=0.3, min_samples=10).fit(X)
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	print("Estimated number of clusters: %d" % n_clusters_)
	print("Estimated number of noise points: %d" % n_noise_)

	unique_labels = set(labels)
	core_samples_mask = np.zeros_like(labels, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True

	colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = [0, 0, 0, 1]

		class_member_mask = labels == k

		xy = X[class_member_mask & core_samples_mask]
		plt.plot(
			xy[:, 0],
			xy[:, 1],
			"o",
			markerfacecolor=tuple(col),
			markeredgecolor="k",
			markersize=14,
		)

		xy = X[class_member_mask & ~core_samples_mask]
		plt.plot(
			xy[:, 0],
			xy[:, 1],
			"o",
			markerfacecolor=tuple(col),
			markeredgecolor="k",
			markersize=6,
		)

	plt.title(f"Estimated number of clusters: {n_clusters_}")
	plt.show()