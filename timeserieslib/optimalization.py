#!/bin/python3

"""
Optimalization Module
-------------------


Implemented algorithms:
(batch) Gradient descent
Stohcastic Gradient descent
Mean-Square error
Iteratively reweighted least squares
"""
import numpy as np


#testing constant
DELTA = 10e-6
def euclidean_norm(x):
	return np.linalg.norm(x)

def MSE(x,y,**kwargs):
	return np.linalg.norm(x-y)




if __name__ == "__main__":    
	# MSE test
	print('MSE test:')
	x = np.array([20,30.1,18.8,-12,-10])
	y = np.array([1.7,-2.1,3.10,-4.20,-10])
	if (MSE(x,y) - 40.97633463354184) > DELTA:
		print('Value test false:')
		print(f'Expected output was 40.97633463354184 but {MSE(x,y)} was returned.')
	else:
		print('Value test passed')
