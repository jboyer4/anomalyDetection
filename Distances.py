# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:16:59 2020

@author: Justin
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.svm
import pandas as pd

from scipy.spatial import distance
import JB_functions_ocIris as ocIris

#Alpha is expected percentage of anomolies in the dataset as a decimal
alpha = .03
#nTrain is percent of nominal points used in the training data as a decimal
nTrain = .5
#Gamma is the kernal width - the array contains the values to test/compare in gridSearch
#gArray = [.6, .5, .4, .3, .2]
gArray = [1.5, 1.2, 1, .8, .5, .2, .01]
#Nu is the upper bound of rejected target data
nVal = alpha

#Import location
irisData = pd.read_csv(r"C:\Users\Justin\OneDrive\Desktop\OSU\419\databases\iris.csv")
#Dataframe labels
predictors = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
target = ["Species"]

def Model(gamma):    
    ##DATA PREP##
    #Prepare data and split into training and testing groups
    trainData, testData = ocIris.splitData(irisData, alpha, nTrain)
    #Normalize data
    scaledTrain, scaledTest = ocIris.normalizeData(trainData, testData, predictors)    
    ############################################################################
    ##CREATE MODEL##
    oc = sk.svm.OneClassSVM(gamma = gamma, nu = nVal)
    #Fit model and get dataframe results
    oc.fit(scaledTrain)
    trainResults = ocIris.getResults(oc, scaledTrain, trainData[target].to_numpy())
    testResults = ocIris.getResults(oc, scaledTest, testData[target].to_numpy())
    return trainResults, testResults



###############################################################################
#Workflow#
###############################################################################
    
##DATA PREP##

#Prepare data and split into training and testing groups
trainData, testData = ocIris.splitData(irisData, alpha, nTrain)
#Normalize data
scaledTrain, scaledTest = ocIris.normalizeData(trainData, testData, predictors)
#Get array of all 4d points to get euclidean dist
allPts = np.concatenate((scaledTrain, scaledTest))

distanceMatrix = np.zeros((150,150))
for i in range(0,150):
    for j in range(i, 150):
        d = distance.euclidean(allPts[i], allPts[j])
        distanceMatrix[i][j] = d
        distanceMatrix[j][i] = d
        
np.savetxt("distMatrix.csv", distanceMatrix, delimiter=",")
#distance matrix populated
#Get max and mins of each 
    