#!/bin/python3

class Model(object):
	def f(self,x):
		return 2*x

	def __init__(self, function = f):
		self.function = f

def square(x):
	return x**2 

test1 = Model()
print(test1.function(10))
test2 = Model(square)
print(test2.function(10))