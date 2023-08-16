#!/bin/python3

"""
Basic statistics
"""

import numpy as np

def input_type_test(data):
	if type(data) != np.ndarray and type(data) == list:
		data = np.array(data)
	else:
		raise TypeError('Data are not in a list')
	return True

def mean(data,**kwargs):
	if input_type_test(data):
		return np.mean(data)

def var(data,**kwargs):
	if input_type_test(data):
		return np.var(data)

#Testing
if __name__ == "__main__":
    print('testing...')
    print('Exception test:')
    print('running mean("hello")')
    #mean('hello')
    print(var([1,2,3]))

