# -*- coding: utf-8 -*-
"""
Created on Sat May 16 20:12:38 2020

@author: Justin
"""

sArray = [.1, 2,4,6,8,10,12,14,16,18,20, 22, 24, 26,28,30,32,34,26,38,40]
#sArray = [2,4,6,8,10,12,14,16,18,20]
aucArray = []
for s in sArray:
    gamma = (1/s)
    
    oc = sk.svm.OneClassSVM(gamma = gamma, nu = nVal)
    #Fit model and get dataframe results
    oc.fit(scaledTrain)
    trainResults = ocIris.getResults(oc, scaledTrain, trainData[target].to_numpy())
    testResults = ocIris.getResults(oc, scaledTest, testData[target].to_numpy())
    
    
    fpr, tpr, threshold = sk.metrics.roc_curve(testResults['True Value'], testResults['Decision Function'])
    #label = str(sigma) + " sigma"
    plt.plot(fpr,tpr)
    auc = sk.metrics.roc_auc_score(testResults['True Value'], testResults['Decision Function'])
    aucArray.append(auc)
    print(str(s) + " AUC: " + str(auc)) 
plt.plot([0,1],[0,1], color='black', alpha = .5, linestyle = "dashed")
plt.legend()
plt.plot


plt.plot(sArray, aucArray)