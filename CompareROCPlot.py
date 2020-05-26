# -*- coding: utf-8 -*-
"""
This script allows you to compare ROC plots of many different s values

Created on Tue Apr 14 11:57:58 2020
@author: Justin
"""

import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.svm
import JB_functions_ocIris as ocIris


#How many times to fit the model with different splits
#s values to test (s = 2*(sigma^2))
sArray = [25, 15, 12, 9, 7, 5]

def plotROC(data, plot, aucScore):
    fpr, tpr, threshold = sk.metrics.roc_curve(testResults['True Value'], testResults['Decision Function'])
    label = str(s) + " AUC: " + str(aucScore)
    roc.plot(fpr,tpr, label = label)

###############################################################################
#Workflow#
###############################################################################

#Plot ROC for each s (sigma^2) in array
fig, roc = plt.subplots()

aucArray = [] 
for s in sArray:
    gamma = 1/(s)           
    trainResults, testResults = ocIris.Model(gamma)
    aucScore = sk.metrics.roc_auc_score(testResults['True Value'], testResults['Decision Function'])
    plotROC(testResults, roc, aucScore)
    aucArray.append(aucScore)
    print(s, " AUC: ", aucScore)
    
#plot y=x reference line
roc.plot([0,1],[0,1], color='black', alpha = .5, linestyle = "dashed")
roc.set_xlabel("False Positive Rate")
roc.set_ylabel("True Positive Rate")
roc.set_title("Sample ROCs")

#Legend works well for small sArrays only
roc.legend()
roc.plot()

