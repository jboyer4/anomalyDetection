# anomalyDetection
Machine learning research project focused on one class SVM. The purpose of this project is to explore how diferent factors - especially hyperparameters - affect ocsvm models. 
The goal is to determine how to systemically determine the hyperparameters that will result in the best model.

ocSVM.py is a practice pipeline for one class svm hyperparameter optimization using the iris practice dataset (included as csv). Since the Iris dataset
has three classes, in order to use it as a sample dataset for one-class models it is seperated into nominal (I. Versicolor, I. Virginica) and anomalous
(I. Setosa).

To get started, load the file JB_functions_ocIris. This file will create a pandas dataframe containing the results from the model that can then be used in other
scripts. Check the other scripts for more details on their specific function.