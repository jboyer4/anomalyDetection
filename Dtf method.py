# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:46:19 2020

@author: Justin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:57:58 2020

@author: Justin
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.svm
import pandas as pd
import JB_functions_ocIris as ocIris

import math

#USER DEFINED GLOBAL VARIABLES

#How many times to fit the model with different splits

splits = 100
#Alpha is expected percentage of anomolies in the dataset as a decimal
alpha = .03
#nTrain is percent of nominal points used in the training data as a decimal
nTrain = .5

#sigmas to test
sArray = [2.6, 2.4, 2.2, 2, 1.8, 1.6, 1.4, 1.2, 1.0, .8, .6, .4, .2, .09]
 
#Gamma is the kernal width - the array contains the values to test/compare in gridSearch
#gArray = [.6, .5, .4, .3, .2]
#gArray = [1.5, 1.2, 1, .8, .5, .2, .01]

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


def plotROC(data, plot):
    fpr, tpr, threshold = sk.metrics.roc_curve(testResults['True Value'], testResults['Decision Function'])
    label = str(sigma) + " sigma"
    roc.plot(fpr,tpr, label = label)
    

    

###############################################################################
#Workflow#
###############################################################################

#Plot ROC for each sigma in array
fig, roc = plt.subplots() 
for sigma in sArray:
    gamma = 1/(2*math.pow(sigma, 2))           
    trainResults, testResults = Model(gamma)
    plotROC(testResults, roc)
    aucScore = sk.metrics.roc_auc_score(testResults['True Value'], testResults['Decision Function'])
    print(sigma, " AUC: ", aucScore)
    
#plot y=x reference line
roc.plot([0,1],[0,1], color='black', alpha = .5, linestyle = "dashed")
roc.set_xlabel("False Positive Rate")
roc.set_ylabel("True Positive Rate")
roc.set_title("Sample ROCs")
roc.legend()
roc.plot()


#loo
fig, auc = plt.subplots() 
auc_averages = []
gArray = []
for sigma in sArray:
    gamma = 1/(2*math.pow(sigma, 2))
    gArray.append(gamma)
    counter = splits
    total = 0
    while counter > 0:
        trainResults, testResults = Model(gamma)
        aucScore = sk.metrics.roc_auc_score(testResults['True Value'], testResults['Decision Function'])
        total += aucScore
        auc.scatter(sigma, aucScore, alpha = .2, color='blue')
        counter -= 1
    auc_averages.append(total/splits)
    
    

print(sArray)
print(auc_averages)

auc.set_xlabel("sigma")
auc.set_ylabel("auc")
auc.plot()
print("gamma vals: ")
print(gArray)

