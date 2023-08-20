#!/bin/python3

'''
Data processing script

'''

#module loading
import numpy as np
import math
import sys
from timeserieslib import preprocessing as pre
from timeserieslib import regression as reg
from timeserieslib import clustering as clust
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeavePOut


np.random.seed(42)

#printing full np.arrays
#np.set_printoptions(threshold=sys.maxsize)


def get_3Dvariance(patient_data):
    '''
    Calculates the Euclidean distance from the camera of every point and returns its variance
    Returns a 21-dimensional vector of variances
    '''
    Vars = []
    for i in range(21):
        dist = [math.sqrt(patient_data[i][0][j]**2+patient_data[i][1][j]**2+patient_data[i][2][j]**2) for j in range(len(patient_data[i][0]))]
        Vars.append(np.var(dist))
    return Vars

if __name__ == '__main__':
    '''
    For each patient (separate measurement) we compute the variations during the exercise bout for each points separately.
    Thus we get a vector of dimensions (# of patients)x21. We use these vectors to perform further data analysis.
    '''
    '''
    Vars,labels = [],[]
    Faces = pre.DataSet() 
    Faces.load_all(True)
    for patient in Faces.dataset:
        Vars.append(get_3Dvariance(patient.DATA))
        labels.append(patient.label)
    Vars,labels = np.array(Vars),np.array(labels)
    print(labels)
    np.savetxt('VARS.txt',Vars,fmt='%1.4e') #saving for time
    np.savetxt('LABELS.txt',labels,fmt='%1.4e') #ditto
    '''
    
    '''
    Ridge regression:

    We use 
    '''

    Vars = np.loadtxt('VARS.txt',dtype=float)
    labels = np.loadtxt('LABELS.txt',dtype=float)
    train_indices = np.random.choice(len(labels),size = int(len(labels)*0.9),replace = False)
    test_indices = np.delete(np.arange(0,len(labels)),train_indices)
    train_data,train_labels = Vars[train_indices],labels[train_indices]
    test_data,test_labels = Vars[test_indices],labels[test_indices]
    print(f'Is Nan check: {sum(np.isnan(Vars))}')
    print(f'Nan index: {np.argwhere(np.isnan(Vars))}')
    Vars = np.delete(Vars,[15],0)
    labels = np.delete(labels,[15],0)
    print(f'Is Nan check: {sum(np.isnan(Vars))}')
    print(f'Nan index: {np.argwhere(np.isnan(Vars))}')
    print(len(Vars))
    lpo = LeavePOut(p=1)
    clf = Ridge(alpha=1e-7)
    print(clf.fit(test_data,test_labels))
    print(cross_val_score(clf,Vars,labels,scoring='neg_mean_squared_error',cv=10))
    print(f'Score for test data: {clf.score(test_data,test_labels)}')
    cv_scores = cross_val_score(clf,Vars,labels,scoring='neg_mean_squared_error',cv=lpo)
    print(cv_scores.mean(),cv_scores.std())






    
    #ridge = reg.LinearRegression(train_data,train_labels,intercept=True,L2_coeff = 0.01)
    #print(ridge.fit())

    #plotting the regression performance
    predicted_labels = clf.predict(test_data)
    index = np.arange(0,len(test_labels))
    print((np.linalg.norm(test_labels - predicted_labels)**2)/len(predicted_labels))
    fig, axs = plt.subplots(1, 2,figsize=(6,6),dpi=300)
    fig.suptitle(f'True labels vs. Predicted')
    axs[0].scatter(index,test_labels,label='Test labels')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Label')
    axs[0].scatter(index,predicted_labels,label='Predicted values')
    axs[0].legend()
    axs[0].grid(True)
    axs[1].scatter(test_labels,predicted_labels)
    axs[1].set_title('True labels vs predicted labels')
    axs[1].set_ylabel('Predicted label')
    axs[1].set_xlabel('True Label')
    axs[1].grid(True)
    plt.show()
    





