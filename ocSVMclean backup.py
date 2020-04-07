# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:22:15 2020

@author: Justin
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import math
# Seaborn visualization library
import seaborn as sns

#Step 1) Import data and Choose Presets:
#Alpha is expected percentage of anomolies in the dataset
alpha = .03
#nTrain is percent of nominal points used in the training data 
nTrain = .5

#Gamma is the kernal width - the array contains the values to test/compare in gridSearch
#gArray = [1.6, 1.4, 1.2, 1, .8, .6, .4, .2, .01, .005]
gArray = [1.6, 1.4, 1.2, 1, .8]

#Nu is the upper bound of rejected target data
nuArray = [.5, .3, .1, .05, .01, .03, .005, .001]

#Import location
irisData = pd.read_csv(r"C:\Users\Justin\OneDrive\Desktop\OSU\419\databases\iris.csv")
speciesGroup = irisData.groupby("Species")

#Dataframe labels
sepalCols = ["Sepal length", "Sepal width"]
dataCols = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
labelCols = ["Species"]

selectedCols = sepalCols




#Step 2)Prepare data and organize into training and testing groups
(verTrain, verTest) = sk.model_selection.train_test_split(speciesGroup.get_group("Iris-versicolor"), train_size = nTrain)
(virTrain, virTest) = sk.model_selection.train_test_split(speciesGroup.get_group("Iris-virginica"), train_size = nTrain)

#anomaly test size will be determined as a percent of the test data (nu)
target_aCount = math.ceil((len(verTrain) + len(virTrain))*alpha)
aTrain = target_aCount/(len(speciesGroup.get_group("Iris-setosa")))
(setTrain, setTest) = sk.model_selection.train_test_split(speciesGroup.get_group("Iris-setosa"), train_size = aTrain)

#Combine corrisponding test and training sets for each species
trainData = pd.concat([verTrain, virTrain, setTrain])
testData = pd.concat([verTest, virTest, setTest])

#Convert Species (3 class) to nominal:1 (vir, ver) and anomaly:-1 (set)
trainData = trainData.replace("Iris-versicolor", 1).replace("Iris-virginica", 1).replace("Iris-setosa", -1)
testData = testData.replace("Iris-versicolor", 1).replace("Iris-virginica", 1).replace("Iris-setosa", -1)





#Step 3: Normalize from training
#Seperate into data and species label
#Use Standard scaler to adjust data - fit to training only - transform to test
#This step also seperated the species column for proper scoring
scaler = sk.preprocessing.StandardScaler().fit(trainData[selectedCols])
#??? Do I apply the scaler to the training data and the test data???
scaledTrain = scaler.transform(trainData[selectedCols])
scaledTest = scaler.transform(testData[selectedCols])





#Step 4: Fit Model
#Set up grid search
oneClass = sk.model_selection.GridSearchCV(sk.svm.OneClassSVM(kernel='rbf'),
                   param_grid={"nu": nuArray,
                               "gamma":gArray}, scoring="accuracy")

#Fit the model using the labelCol to score
oneClass.fit(scaledTrain, trainData[labelCols])
print(oneClass.best_params_)
prediction = oneClass.predict(scaledTest)

#Add predicted value and true value columns back on
result = np.hstack((scaledTest, np.atleast_2d(prediction).T))
result = np.hstack((result, testData[labelCols]))

#
cMatrix = []


#Build Plot and Confusion matrix
trueVal = -1
predVal = -2
for sample in result:
    if sample[trueVal] ==  1 and sample[predVal] == 1:
        cMatrix.append('tPos')
    elif sample[trueVal] ==  1 and sample[predVal] == -1:
         cMatrix.append('fNeg')
    elif sample[trueVal] ==  -1 and sample[predVal] == -1:
         cMatrix.append('tNeg')        
    elif sample[trueVal] ==  -1 and sample[predVal] == 1:
        cMatrix.append('fPos')
    else:
        print("error:", sample)

      
#result = pd.DataFrame(data = result, columns = ["Sepal length", "Sepal width", "Petal length", "Petal width", "Predicted Value", "True Value"])        
result = pd.DataFrame(data = result, columns = ["Sepal length", "Sepal width", "Predicted Value", "True Value"])
#sns.pairplot(result)

result["Confusion Matrix"] = cMatrix
matrixGroups = result.groupby("Confusion Matrix")
tPos = matrixGroups.get_group("tPos")
fPos = matrixGroups.get_group("fPos")
tNeg = matrixGroups.get_group("tNeg")
fNeg = matrixGroups.get_group("fNeg")
plt.scatter(tPos["Sepal length"], tPos["Sepal width"], c = 'g', label = "True Pos")
plt.scatter(fPos["Sepal length"], fPos["Sepal width"], c = 'r', label = "False Pos")
plt.scatter(tNeg["Sepal length"], tNeg["Sepal width"], c = 'b', label = "True Neg")
plt.scatter(fNeg["Sepal length"], fNeg["Sepal width"], c = 'y', label = "False Neg")
plt.legend()
plt.show()
#Plots first 2 cols (2d plot)
#plt.scatter(result[:,0],result[:,1], c='y')








