# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:22:15 2020

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
scaledTrain, scaledTest = ocIris.normalizeData(trainData, testData, selectedCols)

###############################################################################

##CREATE MODEL##

#Set up grid search
oneClass = sk.model_selection.GridSearchCV(sk.svm.OneClassSVM(kernel='rbf'),
                   param_grid={"nu": nuArray,
                               "gamma":gArray}, scoring="accuracy")

#Fit the model using the labelCol to score
oneClass.fit(scaledTrain, trainData[labelCols])
print("Grid search selected the hyperparameters: ")
print(oneClass.best_params_,"\n")


#Build results dataframes
trainResults = ocIris.getResults(oneClass, scaledTrain, trainData[labelCols].to_numpy())
testResults = ocIris.getResults(oneClass, scaledTest, testData[labelCols].to_numpy())



###############################################################################

##PLOT RESULTS##

#Build pairplot if asked for
if showPairPlot:
    ocIris.plotPairs(testResults, len(selectedCols))

else:
    #If the training data was 2D, draw the training boundrary
    if len(trainResults.columns) == 5:
        xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        Z = oneClass.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    #Plot
    ocIris.plotScatter(trainResults, testResults, 0,1)
    
    #PLOT IDEAS FOR FULL DATASET
    #Sepal len x width
    #ocIris.plotScatter(trainResults, testResults, 0,1)
    #Petal len x width
    #ocIris.plotScatter(trainResults, testResults, 2,3)









