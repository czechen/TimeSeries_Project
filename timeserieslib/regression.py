#!/bin/python3

"""
Regression Module
-------------------


Implemented algorithms:
Linear - DONE
GLE - TODO
Ridge (L2 regularization) (part of the linear regression) - DONE (partly)
Polynomial - DONE

Other:
Plotting - DONE
Errors for coeffs - TODO

"""
import sys, os
sys.path.append(os.getcwd())
import numpy as np
import timeserieslib.exceptions as exceptions
from matplotlib import pyplot as plt


#printing full np.arrays
#np.set_printoptions(threshold=sys.maxsize)

class LinearRegression(object):
	'''
	Linear Regression with L2 regulaziation

	'''
	def __init__(self,data_matrix,y,intercept = True, L2_coeff = 0):
		self.n = len(y) #number of 'samples'
		self.dim = len(data_matrix[0]) #dimension
		self.intercept = intercept
		self.L2 = L2_coeff
		if intercept:
			ones_matrix = np.ones((self.n,self.dim+1))
			ones_matrix[:,:-1] = data_matrix
			self.data = ones_matrix
			self.k = self.dim + 1 #number of parameters
		else:
			self.data = data_matrix
			self.k = self.dim #number of parameters
		self.y = y
	
	def fit(self):
		#calculates the paramateres of linear regression using the closed form
		self.params = np.dot(np.linalg.inv(np.dot(self.data.T,self.data)+self.L2*np.identity(self.k)),np.dot(self.data.T,self.y)) #(X^TX-lambdaI)^(-1)X^Ty (Ridge)
		SSr = np.linalg.norm(self.y - np.dot(self.data,self.params))**2 #residual sum of squares
		SSt = np.linalg.norm(self.y - np.mean(self.y)*np.ones((self.n,1)))**2 #total sum of squares
		self.R_squared = 1-SSr/SSt
		return self.params,self.R_squared

	def predict(self,values):
		if values.shape[1] != self.dim:
			raise exceptions.IncompatibleDimensions(f'Incorrect number of predictors; expected {self.dim} but {values.shape[1]} were given')
		params = getattr(self, 'params', None)
		if type(params) == type(None):
			self.fit()
			print(self.params)
		val_matrix = np.ones((len(values),self.dim+int(self.intercept)))
		val_matrix[:,0:self.dim] = values
		return np.dot(val_matrix,self.params)
		

	def fit_sgd(self,MAX_STEPS,treshold_error,learning_step):
		if self.L1 != 0:
			raise AttributeError( "Using SGD for Ridge Regression fitting is not recommended" )
		self.params = np.ones([self.k,1])
		for i in range(MAX_STEPS):
			r = self.y - np.dot(self.data,self.params)
			if np.linalg.norm(r) < treshold_error:
				return self.params
			else:
				for j in range(self.n):
					for k in range(self.k):
						self.params[k] = self.params[k] + learning_step*(r[j])*self.data[j][k]
		SSr = np.linalg.norm(self.y - np.dot(self.data,self.params))**2 #residual sum of squares
		SSt = np.linalg.norm(self.y - np.mean(self.y)*np.ones((self.n,1)))**2 #total sum of squares
		self.R_squared = 1-SSr/SSt
		return self.params,self.R_squared 


	def plot(self,DPI):
		params = getattr(self, 'params', None)
		if type(params) == type(None):
			self.fit()
		if self.dim == 1:
			predicted_y =  self.predict(self.data[:,0:self.dim])
			fig, ax = plt.subplots(1, 1,figsize=(6,6),dpi=DPI)
			fig.suptitle(f'Linear regression; R\N{SUPERSCRIPT TWO} = {self.R_squared}')
			ax.scatter(self.data[:,0],self.y,label='Data Points')
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.scatter(self.data[:,0],predicted_y,label='predicted values')
			ax.plot(self.data[:,0],predicted_y,label='model',color='g')
			ax.legend()
			ax.grid(True)
			plt.show()
		elif self.dim == 2:
			x1_range = np.arange(self.data[:,0].min(), self.data[:,0].max())
			x2_range = np.arange(self.data[:,1].min(), self.data[:,1].max())
			X1, X2 = np.meshgrid(x1_range, x2_range)
			Z = np.zeros(X1.shape)
			for r in range(X1.shape[0]):
			    for c in range(X1.shape[1]):
			        Z[r,c] = self.params[0] * X1[r,c] + self.params[1] * X2[r,c] + self.params[2]

			predicted_y =  self.predict(self.data[:,0:self.dim])
			fig = plt.figure()
			fig.suptitle(f'Linear regression; R\N{SUPERSCRIPT TWO} = {self.R_squared}')
			ax = fig.add_subplot(projection='3d')
			ax.scatter(self.data[:,0],self.data[:,1],self.y,label='Data Points')
			ax.set_xlabel('X')
			ax.set_ylabel('Y')
			ax.scatter(self.data[:,0],self.data[:,1],predicted_y,label='Predicted values')
			ax.plot_wireframe(X1, X2, Z, alpha = 0.5,color='g',label='model')
			ax.legend()
			#axs.grid(True)
			plt.savefig('plot.png')
			plt.show()
		else:
			raise exceptions.DimensionError('Cannot plot higher-dimensional data. Human beings are prisoners to 3D space.')




class PolynomialRegression(LinearRegression):
	'''
	Polynomial regression in one dimension

	For numerical stability reasons fitting with Stochastic Gradient descent is not supported
	'''
	def __init__(self,x_values,y,degree,intercept=True):
		self.dim = degree
		self.intercept = intercept
		self.n = len(y) #number of 'samples'
		self.k = degree + int(intercept)
		data_matrix = np.array(x**(int(not intercept)))
		for i in range(1 + int(not intercept),degree+1):
			data_matrix = np.hstack((x_values**i,data_matrix))
		self.data = data_matrix
		self.x_values = x_values
		self.y = y

	def fit_sgd(self,*args):
		raise AttributeError( "Polynomial Regression does not support SGD" )

	def plot(self,DPI):
		params = getattr(self, 'params', None)
		if type(params) == type(None):
			self.fit()
		predicted_y =  self.predict(self.x_values)
		x_range = np.arange(self.x_values.min(), self.x_values.max(),0.5).reshape(-1,1)
		Y = self.predict(x_range)
		fig, ax = plt.subplots(1, 1,figsize=(6,6),dpi=DPI)
		fig.suptitle(f'Polynomial regression (degree: {self.dim}); R\N{SUPERSCRIPT TWO} = {self.R_squared}')
		ax.scatter(self.x_values,self.y,label='Data Points')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.scatter(self.x_values,predicted_y,label='Predicted values')
		ax.plot(x_range,Y,label='model',color='g')
		ax.legend()
		ax.grid(True)
		plt.show()

	def predict(self,values):
		if values.shape[1] != 1:
			raise exceptions.IncompatibleDimensions(f'Incorrect number of predictors; expected {1} but {values.shape[1]} were given')
		value_matrix = np.array(values)
		for i in range(2,self.dim+1):
			value_matrix = np.hstack((values**i,value_matrix))
		return super().predict(value_matrix)

#Testing
if __name__ == "__main__":
	#testing linear regression
	#'''
	x = np.array([[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35],[0,1],[5,1]])
	y = np.array([4, 5, 20, 14, 32, 22, 38, 43,4,5])
	#x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
	#y = np.array([15, 11, 2, 8, 25, 32]).reshape((-1,1))
	#x, y = np.array(x), np.array(y).reshape(-1,1)
	#print('x:',x)
	#print('y:',y)
	model = LinearRegression(x,y,True,10)
	print(model.fit())
	#print(model.fit_sgd(50000,1e-4,0.0001))
	#print(model.predict(np.array([[0,1],[5,1],[15,2]])))
	model.plot(300)
	'''
	x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
	y = np.array([15, 11, 2, 8, 25, 32]).reshape((-1,1))
	model_pol = PolynomialRegression(x,y,3,True)
	print(model_pol.data)
	print(model_pol.fit())
	#print(model_pol.fit_sgd(100000,1e-4,1e-7))
	print(model_pol.predict(np.array([[2]])))
	model_pol.plot(300)
	'''

