#!/bin/python3

"""
Optimalization Module
-------------------
nearest neighbor graph (NNG) TODO
"""

import numpy as np

def Standard_scale(data_matrix):
	return (data_matrix - data_matrix.mean())/(data_matrix.std())

if __name__ == "__main__":    
	#testing
	#treshold error for testing
	EPSILON = 10e-6
	a = np.array([1,2,3,4,5])
	print(standard_scale(a))
