# -*- coding: utf-8 -*-
"""
Creates a Euclidean distance plot for the Iris data set
It plots a histogram of all distances and exports the matrix as a csv

Created on Tue Apr 21 17:16:59 2020
@author: Justin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.spatial import distance
import JB_functions_ocIris as ocIris

alpha = .03
nTrain = .5
irisData = pd.read_csv(r"C:\Users\Justin\OneDrive\Desktop\ML_research\Iris mods\iris.csv")
predictors = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
target = ["Species"]

###############################################################################
#Workflow#
###############################################################################
    
##DATA PREP##
#Prepare data and split into training and testing groups and normalize the data
#to create a euclidean distance area logging the 4-D distance between each point
trainData, testData = ocIris.splitData(irisData, alpha, nTrain)
scaledTrain, scaledTest = ocIris.normalizeData(trainData, testData, predictors)
allPts = np.concatenate((scaledTrain, scaledTest))
distanceMatrix = np.zeros((150,150))

##CONSTRUCT DISTANCE MATRIX##
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

        
np.savetxt("distMatrix.csv", distanceMatrix, delimiter=",")
plt.hist(distanceMatrix.flatten(), bins = 300, density = True)
plt.plot()
