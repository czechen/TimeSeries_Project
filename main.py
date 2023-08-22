#!/bin/python3

'''
Data abalysis script

Due to time and stability consideration we use the standar sklean python library.
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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph


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
    #For each patient (separate measurement) we compute the variations during the exercise bout for each points separately.
    #Thus we get a vector of dimensions (# of patients)x21. We use these vectors to perform further data analysis.
    
    '''
    Vars,labels = [],[]
    Faces = pre.DataSet() 
    Faces.load_all(True)
    for patient in Faces.dataset:
        Vars.append(get_3Dvariance(patient.DATA))
        labels.append(patient.label)
    Vars,labels = np.array(Vars),np.array(labels)
    print(labels)
    np.savetxt('VARS.txt',Vars,fmt='%1.4e') #saving for future usage
    np.savetxt('LABELS.txt',labels,fmt='%1.4e') #ditto
    '''
    
    '''
    Ridge regression:

    We use cross-validation and Leave P out method to assess the quality of the Ridge regresion model.
    '''

    Vars = np.loadtxt('VARS.txt',dtype=float)
    labels = np.loadtxt('LABELS.txt',dtype=float)
    #removing Nan values
    print(f'Is Nan check: {sum(np.isnan(Vars))}')
    print(f'Nan index: {np.argwhere(np.isnan(Vars))}')
    Vars = np.delete(Vars,[15],0)
    labels = np.delete(labels,[15],0)
    #print(f'Is Nan check: {sum(np.isnan(Vars))}')
    #print(f'Nan index: {np.argwhere(np.isnan(Vars))}')
    '''
    clf = Ridge(alpha=1e-3)
    '''
    '''

    #plotting leave one out results
    
    predicted_labels = np.array([])
    for i in range(len(Vars)):
        train_data,train_labels = np.delete(Vars,[i],0), np.delete(labels,[i],0)
        test_data = Vars[i]
        clf.fit(train_data,train_labels)
        predicted_labels = np.append(predicted_labels,clf.predict([test_data]))

    fig, ax = plt.subplots(1, 1,figsize=(7,10),dpi=150)
    index = np.arange(0,len(Vars))
    errors = labels-predicted_labels
    ax.scatter(index,errors,color='r',marker='x',label = 'Leave One out error')
    ax.set_xlabel('Index')
    ax.set_ylabel('Label Error')
    ax.set_title(f'LOO error, MSE = {(np.sum(errors**2))/len(errors)}')
    ax.legend()
    ax.grid(True)
    plt.show()
    '''


    #plotting LPO results based on alpha
    '''
    n_alphas = 15
    alphas = np.logspace(-10, 0, n_alphas)
    print(alphas)
    lpos = [[],[]]
    lpos_std = [[],[]]
    for a in alphas:
        clf = Ridge(alpha = a)
        for p in [1,2]:
            lpo = LeavePOut(p)
            cv_scores = -1*cross_val_score(clf,Vars,labels,scoring='neg_mean_squared_error',cv=lpo)
            lpos[p-1].append(cv_scores.mean())
            lpos_std[p-1].append(cv_scores.std())

    print(lpos[0],lpos_std[0])
    fig, ax = plt.subplots(1, 1,figsize=(7,10),dpi=150)
    ax.errorbar(alphas,lpos[0],yerr=lpos_std[0],uplims=False, lolims=False,color = 'r',label = 'Avg. MSE for Leave One out')
    ax.errorbar(alphas,lpos[1],yerr=lpos_std[1],uplims=False, lolims=False,color = 'g',label = 'Avg. MSE for Leave Two out')
    ax.set_xlabel('Alpha')
    ax.set_xscale("log")
    ax.set_ylabel('Avg. MSE')
    ax.set_title(f'Leave P out performance vs. alpha')
    ax.legend()
    ax.grid(True)
    plt.show()
    #plotting the regression performance

    train_indices = np.random.choice(len(labels),size = int(len(labels)*0.8),replace = False)
    test_indices = np.delete(np.arange(0,len(labels)),train_indices)
    train_data,train_labels = Vars[train_indices],labels[train_indices]
    test_data,test_labels = Vars[test_indices],labels[test_indices]
    
    print(clf.fit(train_data,train_labels))
    print(f'Score for test data: {clf.score(test_data,test_labels)}')
    
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
    '''
    '''
    DBSCAN
    '''
    #finding optimal epsilon values for given minPts parameter using k-nearest neighbor graph 
    '''
    epsilons = []
    distances = []
    A = kneighbors_graph(Vars, 50, mode='distance', include_self=False)
    for i in range(A.shape[0]):
        distances.append(A[i].tocoo().data)
    distances = np.array(distances)
    distances_sorted = []
    mean_distances = []
    for i in range(50):
        distances_sorted.append(sorted(distances[:,i],reverse = True))
        mean_distances.append(distances[:,i].mean())
    print(mean_distances)
    fig, ax = plt.subplots(1, 1,figsize=(7,10),dpi=150)
    index = np.arange(0,len(Vars))
    for i in range(9,50,10):
        ax.plot(index,distances_sorted[i],label = f'{i+1}th nearest neighbour distance')
    ax.set_xlabel('Index')
    ax.set_ylabel('Distance')
    ax.set_title(f'Kth nearest neighbour distance using euclidean metric')
    ax.legend()
    ax.grid(True)
    plt.show()
    '''

    #Plotting DBSCAN performance for multple parameters (epsilon, minPts)
    n_clusters,n_noise = [],[]
    epsilons = np.arange(0.01,0.5,0.005)
    minPts_range = np.arange(2,10,1)
    for eps in epsilons:
        for minPts in minPts_range:
            db = DBSCAN(eps=eps, min_samples=minPts).fit(Vars)
            labels_db = db.labels_
            n_clusters.append(len(set(labels_db)) - (1 if -1 in labels_db else 0))
            n_noise.append(list(labels_db).count(-1))
    
    X,Y = np.meshgrid(epsilons,minPts_range)
    fig = plt.figure(figsize=(10,10),dpi=150)
    fig.suptitle(f'DBSCAN performance for varying parameters')
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X,Y,n_clusters,label='num. of formed clusters')
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('minPts')
    ax.scatter(X,Y,n_noise,label='num. of noise points')
    ax.legend()
    #axs.grid(True)
    plt.savefig('plot.png')
    plt.show()