#!/bin/python3

"""
Preprocessing module
------------------------

Module for creating a dataset


"""
import pandas as pd
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

class DataSet:
	CVIKY = ['oboci','mraceni','oci','usmev','zuby','spuleni-rtu','tvare-nafouknuti','oci-zuby','celo-rty']
	points_names = {0:'LefteyeMidbottom',1:'LefteyeMidtop',2:'LefteyeInnercorner',3:'LefteyeOutercorner',
			4:'LefteyebrowInner',5:'LefteyebrowCenter',6:'RighteyeMidbottom',7:'RighteyeMidtop',8:'RighteyeInnercorner',9:'RighteyeOutercorner',
			10:'RighteyebrowInner',11:'RighteyebrowCenter',12:'NoseTip',13:'MouthLowerlipMidbottom',14:'MouthLeftcorner',15:'MouthRightcorner',
			16:'MouthUpperlipMidtop',17:'ChinCenter',18:'ForeheadCenter',19:'LeftcheekCenter',20:'RightcheekCenter'}
	#s = pd.read_csv('/home/czechen/Projects/Diplomka/DATA/len.txt')
	#MEAN = int(s.mean()[0])
	#SD = s.std()[0]

	class PatientEval():
		"""
		Class storing information of one pation evaluation. Evaluations of the same patient at different dates are considered independent.
		Atributes:
		ID	patient id
		date	date of the evaluation
		file_names	names of all the csv files coresponding to the given evaluation
		label	HB value of the pation
		BF	befor/after medical procedure
		"""
		def __init__(self, ID, date):
			self.ID = ID
			self.date = date
			search_value = (11-len(str(self.ID))) * '0' + str(self.ID) + '_' + self.date
			self.file_names = glob(f'/home/czechen/Projects/Diplomka/DATA/csv_points/{search_value}*') #search for all files corresponding to evaluation
			self.label = self.file_names[0].split('/')[7].split('_')[4][-1]
			self.BF = self.file_names[0].split('/')[7].split('_')[5][-1]
			self.load_data()

		def load_data(self):
			self.DATA = []
			for i in range(21):
				t,x,y,z = [],[],[],[]
				files_cvik = [s for s in self.file_names if DataSet.points_names[i] in s]
				files_cvik.sort(key=lambda x: str(x.split('/')[7].split('_')[5]))
				for j in files_cvik:
					point = pd.read_csv(j)
					t = t+list(point.t)
					x = x+list(point.x)
					y = y+list(point.y)
					z = z+list(point.z)
				self.DATA.append(np.array([np.array(t),np.array(x),np.array(y),np.array(z)]))

		def shape(self):
			return self.DATA.shape()

	def __init__(self):
		'''
		self.CVIKY = ['oboci','mraceni','oci','usmev','zuby','spuleni-rtu','tvare-nafouknuti','oci-zuby','celo-rty']
		self.points_names = {0:'LefteyeMidbottom',1:'LefteyeMidtop',2:'LefteyeInnercorner',3:'LefteyeOutercorner',
			4:'LefteyebrowInner',5:'LefteyebrowCenter',6:'RighteyeMidbottom',7:'RighteyeMidtop',8:'RighteyeInnercorner',9:'RighteyeOutercorner',
			10:'RighteyebrowInner',11:'RighteyebrowCenter',12:'NoseTip',13:'MouthLowerlipMidbottom',14:'MouthLeftcorner',15:'MouthRightcorner',
			16:'MouthUpperlipMidtop',17:'ChinCenter',18:'ForeheadCenter',19:'LeftcheekCenter',20:'RightcheekCenter'}
		s = pd.read_csv('/home/czechen/Projects/Diplomka/DATA/len.txt')
		self.MEAN = int(s.mean()[0])
		self.SD = s.std()[0]
		'''
		patients = {}
		for i in glob('/home/czechen/Projects/Diplomka/DATA/csv_points/*'):
			ID = int(i.split('/')[7].split('_')[0])
			DATE = str(i.split('/')[7].split('_')[1])
			if ID not in patients:
				patients.update({ID:{DATE}})
			else:
				patients[ID].add(DATE)
		keys = list(patients.keys())
		keys.sort()
		self.Patients = {i: sorted(list(patients[i])) for i in keys}

	def __len__(self):
		S = 0
		for i in self.Patients.keys():
			S += len(self.Patients[i])
		return S

	def validate(self, ID, date, MEAN, SD):
		search_value = (11-len(str(ID))) * '0' + str(ID) + '_' + date
		file_name = glob(f'/home/czechen/Projects/Diplomka/DATA/csv_points/{search_value}*')
		lenght = len(list(pd.read_csv(file_name[0]).t))
		if  MEAN - SD/2 <= lenght <= MEAN + SD/2 and len(file_name) >= 21*9-1: return True
		else:
			print('Not enought data') 
			return False

	def load_patient(self,ID,date):
		patient = self.PatientEval(ID,date)
		return(patient)	

	def load_all(self):
		data = []
		for i in self.Patients.keys():
			for j in self.Patients[i]:
				data.append(self.load_patient(i,j))
		self.dataset = data

#Testing
if __name__ == "__main__":
	DataSet = DataSet() 
	print(DataSet.Patients)
	patient = DataSet.load_patient(1,DataSet.Patients[1][0])
	print(patient.file_names)
	print(patient.DATA[0])
	print(len(patient.DATA[0]))
	#plt.plot(patient.DATA[0][0],patient.DATA[0][3])
	plt.plot(list(range(len(patient.DATA[0][0]))),patient.DATA[0][3])
	plt.show()
	DataSet.load_all()