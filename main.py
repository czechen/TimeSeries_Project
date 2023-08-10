#!/bin/python3

#module loading
import numpy as np
from timeserieslib import statistics as stat
from timeserieslib import preprocessing as pre
from timeserieslib import optimalization as opt

DataSet = pre.DataSet() 
DataSet.load_all()

