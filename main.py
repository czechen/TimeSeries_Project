#!/bin/python3

#module loading
import numpy as np
import math
from timeserieslib import preprocessing as pre
from timeserieslib import optimalization as opt

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


DataSet = pre.DataSet() 
#DataSet.load_all()
patient = DataSet.load_patient(1,DataSet.Patients[1][0])
patient_ST = DataSet.load_patient(1,DataSet.Patients[1][0], standardization=True)
print(patient.DATA[0])
print(get_3Dvariance(patient.DATA))
print(np.var(patient.DATA))
print(patient_ST.DATA[0])


