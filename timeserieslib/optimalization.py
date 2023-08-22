#!/bin/python3

"""
Optimalization Module
-------------------
nearest neighbor graph (NNG) TODO
"""

import numpy as np


def Standard_scale(data_matrix):
	'''
	Standardization function

	returns a data_matrix where every element was standardized based on the formula:

	x_std = (x-mu)/sigma, where mu is the data_matrix mean and sigma is its standard deviation
	'''
	return (data_matrix - data_matrix.mean())/(data_matrix.std())

if __name__ == "__main__":    
	#testing
	#treshold error for testing
	EPSILON = 10e-6
	a = np.array([1,2,3,4,5])
	print(standard_scale(a))
