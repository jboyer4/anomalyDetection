# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:10:02 2020

@author: Justin
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.svm
import pandas as pd
import JB_functions_ocIris as ocIris

#USER DEFINED GLOBAL VARIABLES
#Alpha is expected percentage of anomolies in the dataset as a decimal
alpha = .03
#nTrain is percent of nominal points used in the training data as a decimal
nTrain = .75

#Gamma is the kernal width - the array contains the values to test/compare in gridSearch
#gArray = [1.6, 1.4, 1.2, 1, .8, .6, .4, .2, .01, .005]
gArray = [1.6, 1.4, 1.2, 1, .8, .6, .4]

#Nu is the upper bound of rejected target data
#nuArray = [.5, .3, .1, .05, .01, .03, .005, .001]
nuArray = [alpha]
#Import location
irisData = pd.read_csv(r"C:\Users\Justin\OneDrive\Desktop\OSU\419\databases\iris.csv")

#Dataframe labels
sepalCols = ["Sepal length", "Sepal width"]
petalCols = ["Petal length", "Petal width"]
dataCols = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
labelCols = ["Species"]

selectedCols = sepalCols

#Would you like to see a point pair plot?
showPairPlot = False  

###############################################################################
#Workflow#
###############################################################################

##DATA PREP##

#Prepare data and split into training and testing groups
trainData, testData = ocIris.splitData(irisData, alpha, nTrain)
#Normalize data
scaledTrain, scaledTest = ocIris.normalizeData(trainData, testData)

###############################################################################

##CREATE MODEL##
