# -*- coding: utf-8 -*-
"""
This script creates a density plot of the decision function values returned by 
the ocsvm model.  It runs 'split' times in order to get an avarage from many 
different possible train/test splits

Created on Tue Apr  7 16:10:02 2020
@author: Justin
"""

import matplotlib.pyplot as plt
import pandas as pd
import JB_functions_ocIris as ocIris
import seaborn 

splits = 100
s = 20
gamma = (1/s)

#Fit model many 'split' number of times to get a good average using many possible
#random 
values = pd.DataFrame()
while splits > 0:   
    trainResults,testResults =  ocIris.Model(gamma)
    values = pd.concat([values, testResults, trainResults])
    splits -= 1

  
###############################################################################
##PLOT##
###############################################################################    

##Prepare decision function hist data
#Group the decision function values into nominal and anomalous
groups = values.groupby("True Value")
nominal = groups.get_group(1)
anomalous = groups.get_group(-1)

plt.axvline(0, c = 'black', linestyle = 'dashed')
plt.ylabel('Density')
plt.xlabel('Decision Function Value')
seaborn.distplot(nominal["Decision Function"], hist = False, label = 'nominal')
seaborn.distplot(anomalous["Decision Function"], hist = False, label = 'anomalous', color = 'r')
plt.legend()


###############################################################################
#Alternative style (histogram)
def PlotDecF(nominal, anomalous):
    #Create plot data
    hData = [nominal["Decision Function"],  anomalous["Decision Function"]]
    colors = ['blue', 'red']
    labels = ['nominal', 'anomalous']

    plt.hist(hData, bins = 20, color = colors, label = labels)
    plt.legend()
    plt.xlabel('Decision Function Value')
    
#PlotDecF(nominal, anomalous)