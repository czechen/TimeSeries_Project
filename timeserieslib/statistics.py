#!/bin/python3

"""
Basic statistics
"""

import numpy as np

def mean(data,**kwargs):
	if type(data) != np.ndarray:
		try: 
			data = np.array(data)
		except:
			print('')
	return np.mean(data)

def var(data,**kwargs):
	if type(data) != np.ndarray:
		data = np.array(data)
	return np.mean(data)


#Testing
if __name__ == "__main__":
    print('test')
