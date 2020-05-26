# -*- coding: utf-8 -*-
"""
This script creates a contoured boundry plot to visualize the decision boundry 
It can makes a 2-D plot and requires the user to choose any two predictors

Created on Mon May 25 18:21:32 2020
@author: Justin
"""
import numpy as np
import JB_functions_ocIris as ocIris
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.svm
import pandas as pd


#The boundrary plot only works with 2 predictors. Choose which two to use.
#The workd flow is otherwise the same as JB_functions_ocIris, check there for 
#more details about variables and functions

s = 15
gamma = (1/s)
nTrain = .75
alpha = .03
irisData = pd.read_csv(r"C:\Users\Justin\OneDrive\Desktop\ML_research\Iris mods\iris.csv")
sepalCols = ["Sepal length", "Sepal width"]
petalCols = ["Petal length", "Petal width"]
dataCols = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
target = ["Species"]

#Select which cols to use for training here
predictors = sepalCols

#Prepare data and split into training and testing groups
trainData, testData = ocIris.splitData(irisData, alpha, nTrain)
#Normalize data
scaledTrain, scaledTest = ocIris.normalizeData(trainData, testData, predictors)    

###############################################################################
##CREATE MODEL##
###############################################################################
oc = sk.svm.OneClassSVM(gamma = gamma, nu = alpha)
#Fit model and get dataframe results
oc.fit(scaledTrain)
trainResults =  ocIris.getResults(oc, scaledTrain, trainData[target].to_numpy())
testResults =  ocIris.getResults(oc, scaledTest, testData[target].to_numpy())

###############################################################################
##PLOT##
###############################################################################

matrixGroups = testResults.groupby("Confusion Matrix")
colorDict = {'true nominal':'blue', 'false nominal':'red',  'true anomaly':'green', 'false anomaly':'orange'}

for name, group in matrixGroups:
    print(name,":",len(group))
    plt.scatter(group[0], group[1], c = colorDict[name], label = name, alpha = .5)

#Overlay training points for reference
plt.scatter(trainResults[0], trainResults[1], c = 'black', alpha = 1, s = 10, label = "training data")
plt.legend(loc="best")

#If the training data was 2D, draw the training boundrary oc is the name of the model made in JB_functions_ocIris
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = oc.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.plot()
