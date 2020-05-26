# -*- coding: utf-8 -*-
"""
OCSVM hyperparameter tuning algorithm for gamma. DFN (Xiao 2013) uses the nearest
and furthest neighbor for each point to optimize 's'or the denominator in 
gamma = 1/(2*sigma^2) 

Created on Tue Apr 21 17:16:59 2020
@author: Justin
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.svm
import pandas as pd
import math
import sympy as sy
from scipy.spatial import distance
from scipy.optimize import minimize
import JB_functions_ocIris as ocIris

alpha = .03
nTrain = .5
irisData = pd.read_csv(r"C:\Users\Justin\OneDrive\Desktop\ML_research\Iris mods\iris.csv")
predictors = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
target = ["Species"]

def dfn(x, nearPts, farPts):
    n = sy.Symbol('n')
    expression1 = 0
    expression2 = 0
    for i in range(len(irisData)):
        expression1 = expression1 + sy.exp(-allNearest[i]/n)
        expression2 = expression2 + sy.exp(-allFurthest[i]/n)
    dfn = -(2/len(irisData))*(((expression1)-(expression2)))
    return  dfn.subs(n, x)

###############################################################################
##DATA PREP##
###############################################################################
    
#Prepare data and split into training and testing groups and normalize the data
#to create a euclidean distance area logging the 4-D distance between each point
trainData, testData = ocIris.splitData(irisData, alpha, nTrain)
scaledTrain, scaledTest = ocIris.normalizeData(trainData, testData, predictors)
allPts = np.concatenate((scaledTrain, scaledTest))
distanceMatrix = np.zeros((150,150))

###############################################################################
##CONSTRUCT DISTANCE MATRIX##
###############################################################################

#Save the distance to the nearest and furthest point for each value. The near 
#and far distances are used to create "pseudo-outliers" in the dfn method (Xiau 2013)
allFurthest = []
allNearest = []
optimized = []
for i in range(0,150):
    #Get the max and min of each column to sum in the DFN equation Xiau 2013: 
    far = -(math.inf)
    near = math.inf

    for j in range(0, 150):
        d = pow(distance.euclidean(allPts[i], allPts[j]),2)
        #d = distance.euclidean(allPts[i], allPts[j])
        if d == 0: distanceMatrix[i][j] = None
        else: 
            distanceMatrix[i][j] = d
            if d > far: far = d
            if d < near: near = d
        
    allNearest.append(near)    
    allFurthest.append(far)

###############################################################################
##GET OPTIMIZED S USING DFN##
###############################################################################    

points = (allNearest, allFurthest) 
convergence = minimize(dfn,
        x0=10,
        args = points,
        method = 'Nelder-Mead'
        )

gamma = (1/(convergence.x[0]))


###############################################################################
##Plot optimized value over distance histogram##
###############################################################################
fig, plots = plt.subplots(1,2,figsize=(15, 5))

        
plots[0].axvline(x = (convergence.x[0]), c= 'red')
plots[0].hist(distanceMatrix.flatten(), bins = 300, density = True)
   
###########################################################################
#Plot ROC ##
###########################################################################
#Plot ROC for optimized s value
oc = sk.svm.OneClassSVM(gamma = gamma, nu = alpha)
#Fit model and get dataframe results
oc.fit(scaledTrain)
trainResults = ocIris.getResults(oc, scaledTrain, trainData[target].to_numpy())
testResults = ocIris.getResults(oc, scaledTest, testData[target].to_numpy())


fpr, tpr, threshold = sk.metrics.roc_curve(testResults['True Value'], testResults['Decision Function'])
#label = str(sigma) + " sigma"
plots[1].plot(fpr,tpr,label = "AUC: " + str(sk.metrics.roc_auc_score(testResults['True Value'], testResults['Decision Function'])))
plots[1].plot([0,1],[0,1], color='black', alpha = .5, linestyle = "dashed")
plots[1].legend()
plots[1].plot
print("AUC: " + str(sk.metrics.roc_auc_score(testResults['True Value'], testResults['Decision Function'])))    
