#!/bin/python3

"""
Regression Module
-------------------


Implemented algorithms:
Linear
Ridge

"""
import numpy as np
import optimalization as opt

class Regression(object):
	''' Regression class '''
	
	def linear_predictor(self,data,a,b):
		return np.matmul(a,data)+b

	def link_function(self,x):
		return x

	def fit(self,data):
		pass

	def __init__(self):
		pass

#
class LinearRegression(Regression):
	def __init__(self):
		pass

#Testing
if __name__ == "__main__":
	linear1 = LinearRegression()
	print(linear1.link_function(10))
