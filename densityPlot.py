# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:10:02 2020

@author: Justin
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection

import pandas as pd
import JB_functions_ocIris as ocIris
import seaborn 

#USER DEFINED GLOBAL VARIABLES

#How many times to fit the model with different splits
splits = 30

#Alpha is expected percentage of anomolies in the dataset as a decimal
alpha = .03
#nTrain is percent of nominal points used in the training data as a decimal
nTrain = .5

#Gamma is the kernal width - the array contains the values to test/compare in gridSearch
gVal = 1
#Nu is the upper bound of rejected target data
nVal = alpha

#Import location
irisData = pd.read_csv(r"C:\Users\Justin\OneDrive\Desktop\OSU\419\databases\iris.csv")
#Dataframe labels
sepalCols = ["Sepal length", "Sepal width"]
petalCols = ["Petal length", "Petal width"]
dataCols = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
labelCols = ["Species"]

selectedCols = dataCols

#Plot decision factor histogram
def PlotDecF(nominal, anomalous):
    #Create plot data
    hData = [nominal["Decision Function"],  anomalous["Decision Function"]]
    colors = ['blue', 'red']
    labels = ['nominal', 'anomalous']

    plt.hist(hData, bins = 20, color = colors, label = labels)
    plt.legend()
    plt.xlabel('Decision Function Value')

###############################################################################
#Workflow#
###############################################################################

#Fit model many times for distribution of possible splits
allResults = pd.DataFrame()
while splits > 0:
    
    ##DATA PREP##
    #Prepare data and split into training and testing groups
    trainData, testData = ocIris.splitData(irisData, alpha, nTrain)
    #Normalize data
    scaledTrain, scaledTest = ocIris.normalizeData(trainData, testData, selectedCols)

   ############################################################################

    ##CREATE MODEL##
    oc = sk.svm.OneClassSVM(gamma = gVal, nu = nVal)
    #Fit model and get dataframe results
    oc.fit(scaledTrain)
    trainResults = ocIris.getResults(oc, scaledTrain, trainData[labelCols].to_numpy())
    testResults = ocIris.getResults(oc, scaledTest, testData[labelCols].to_numpy())
    
    ##RESET LOOP##
    allResults = pd.concat([allResults, testResults, trainResults])
    splits -= 1

    ############################################################################
   
   

##BUILD HISTOGRAM

##Prepare decision function hist data
#Group the decision function values into nominal and anomalous
groups = allResults.groupby("True Value")
nominal = groups.get_group(1)
anomalous = groups.get_group(-1)

#PlotDecF(nominal, anomalous)
seaborn.distplot(nominal["Decision Function"], hist = False)
seaborn.distplot(anomalous["Decision Function"], hist = False)

