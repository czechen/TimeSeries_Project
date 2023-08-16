#!/bin/python3

"""
Optimalization Module
-------------------


Implemented algorithms:
(batch) Gradient descent
Stohcastic Gradient descent
Mean-Square error
Iteratively reweighted least squares
Derivative-free algorithms
"""
import numpy as np

def L2_norm(x):
	return np.linalg.norm(x)

def l2_dist(x,y,**kwargs):
	return np.linalg.norm(x-y)

def standard_scale(data_matrix):
	return (data_matrix - data_matrix.mean())/(data_matrix.std())



if __name__ == "__main__":    
	#testing
	#treshold error for testing
	EPSILON = 10e-6
	a = np.array([1,2,3,4,5])
	print(standard_scale(a))
