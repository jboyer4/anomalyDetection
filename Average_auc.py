# -*- coding: utf-8 -*-
"""
This script finds the average AUC score for each s value in an array by doing a
specified number of random train/test splits

Created on Tue Apr 14 11:57:58 2020
@author: Justin
"""

import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.svm
import pandas as pd
import JB_functions_ocIris as ocIris

#How many times to fit the model with different splits
#s values to test (s = 2*(sigma^2))
sArray = [60,55,50,45,40,35,30, 25, 20,17, 15, 13, 11, 9, 8,7, 5, 3, 1, .5, .1]

#How many times to fit the model with different splits. Useful for determining 
#averages over many possible training/test splits. O(n^2) complexity
splits = 25

###############################################################################
#Workflow#
###############################################################################

auc_averages = []
gArray = []
for s in sArray:
    gamma = (1/s)
    gArray.append(gamma)
    counter = splits
    total = 0
    while counter > 0:
        trainResults, testResults = ocIris.Model(gamma)
        aucScore = sk.metrics.roc_auc_score(testResults['True Value'], testResults['Decision Function'])
        total += aucScore
        plt.scatter(s, aucScore, alpha = .2, color='blue')
        counter -= 1
    auc_averages.append(total/splits)
    
plt.xlabel("s")
plt.ylabel("auc")
plt.plot()

d = {'s value': sArray, 'g value': gArray, 'average AUC':auc_averages}
averagesTable = pd.DataFrame(data = d)
print(averagesTable)