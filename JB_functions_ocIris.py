# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:20:31 2020

@author: Justin
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.svm
import pandas as pd
import math
import seaborn as sns

################################################################################
#Functions#
###############################################################################


#splitData(data: Iris data as pandas data frame, alpha: % of training data to be anomaly as decimal, nTrain: % of nominal points to use for training as decimal)
#Because the iris dataset is three class not one class, we need to convert it to a one class dataset in order to use it as sample data for anomaly detection with ocsvm
#Iris setosa is considered anomlaous (-1) with versicolor and virginica considered nominal (1).
#
#returns: Two pandas dataframes - one to be used for training the data and the other for use as a testing set
def splitData(data, alpha, nTrain):
    speciesGroup = data.groupby("Species")
    
    #Split nominal points
    (verTrain, verTest) = sk.model_selection.train_test_split(speciesGroup.get_group("Iris-versicolor"), train_size = nTrain)
    (virTrain, virTest) = sk.model_selection.train_test_split(speciesGroup.get_group("Iris-virginica"), train_size = nTrain)
    
    #anomaly test size will be determined as a percent of the test data (nu)
    target_aCount = math.ceil((len(verTrain) + len(virTrain))*alpha)
    aTrain = target_aCount/(len(speciesGroup.get_group("Iris-setosa")))
    
    #Split anomalous points
    (setTrain, setTest) = sk.model_selection.train_test_split(speciesGroup.get_group("Iris-setosa"), train_size = aTrain)

    #Combine corrisponding test and training sets for each species
    trainData = pd.concat([verTrain, virTrain, setTrain])
    testData = pd.concat([verTest, virTest, setTest])
    
    #Convert Species (3 class) to nominal:1 (vir, ver) and anomaly:-1 (set)
    trainData = trainData.replace("Iris-versicolor", 1).replace("Iris-virginica", 1).replace("Iris-setosa", -1)
    testData = testData.replace("Iris-versicolor", 1).replace("Iris-virginica", 1).replace("Iris-setosa", -1)
    
    return(trainData, testData)

#buildMatrix(predictions: a numpy array containing the class predicted for each data point, labels: a numpy array containing the "true class" for each data point)
#By comparing the model prediction with the true value this function decides if the point was correctly classed and categorizes each as:
#true nominal: correctly labeled as nomial point 
#false nominal: incorrectly labeled as nomial point
#true anomaly: correctly labeled as anomaly
#false anamaly: incorrectly labeled as anomaly
#
#returns: numpy array containing one of the above classifications for each point
def buildMatrix(predictions, labels):
    if(len(predictions) != len(labels)):
        print("error: predictions and true values do not match up")
    cMatrix = []
    for x in range(0,len(predictions)):
        if predictions[x] ==  1 and labels[x] == 1:
            cMatrix.append('true nominal')
        elif predictions[x] ==  1 and labels[x] == -1:
            cMatrix.append('false nominal')
        elif predictions[x] ==  -1 and labels[x] == -1:
            cMatrix.append('true anomaly')        
        elif predictions[x] ==  -1 and labels[x] == 1:
            cMatrix.append('false anomaly')
        else:
            print("error: invalid value in array")
    return cMatrix

#getResults(model: fitted sklearn model to predict data class (1 or -1), scaledData: numpy array or normalized data, trueClass: numpy array containing the "true class"corresponding to each point in the scaledData
#This function uses the constructed model to pridict the class for each point. It then compares this to the "true class" to classify each point in a confusion matrix
#
#returns: pandas dataframe with the the normailzed data columns followed by the predicted class, the "true class", and a confusion matrix classification
def getResults(model, scaledData, trueClass):
    prediction = model.predict(scaledData)
    df = pd.DataFrame(data = scaledData)        
    df["Prediciton"] = np.atleast_2d(prediction).T
    df["True Value"] = trueClass
    df["Confusion Matrix"] = buildMatrix(prediction, trueClass)
    return df


#normalizeData(train: pandas dataframe of the training split, test:pandas dataframe of the testing split, cols: array containing the column labels of the data to use
#Use Standard scaler to adjust data - fit to training only - transform to test
#This step also seperated the species column for proper scoring (Don't want 1, -1 normalized)
#
#returns: numpy arrays of normalized train and test data
def normalizeData(train, test, cols):
    scaler = sk.preprocessing.StandardScaler()
    scaledTrain = scaler.fit_transform(train[cols])
    scaledTest = scaler.transform(test[cols])
    return(scaledTrain, scaledTest)


#plotScatter(train: pandas df of training results, test: pandas df of test results, var1: int of column to be used for the scatter x values, var2: int of column to be used for the scatter y values )
#Group data by the confusion matrix category category
#plot training data by confusion matrix category category
#plot training points in black
#
#returns: Nothing - prints plot to screen
def plotScatter(train, test, var1, var2):
    matrixGroups = test.groupby("Confusion Matrix")
    colorDict = {'true nominal':'blue', 'false nominal':'red',  'true anomaly':'green', 'false anomaly':'orange'}

    for name, group in matrixGroups:
        print(name,":",len(group))
        plt.scatter(group[var1], group[var2], c = colorDict[name], label = name, alpha = .5)

    #Overlay training points for reference
    plt.scatter(train[var1], train[var2], c = 'black', alpha = 1, s = 10, label = "training data")

    plt.legend(loc="best")
    plt.plot()


#plotPairs(data: pandas df to plot, cols: int number of columns to plot)
#
#Returns: nothing - prints seaborn pair plot to the screen
def plotPairs(data, cols):
    ppCols = []
    for x in range(0, cols):
        ppCols.append(x)
    sns.pairplot(data, vars = ppCols, hue = "Confusion Matrix")