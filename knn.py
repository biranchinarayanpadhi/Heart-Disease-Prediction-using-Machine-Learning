# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 17:03:44 2021

@author: vxp190034
"""
import numpy as np
import pandas as pd
import math as math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Hamming distance, Manhattan distance and Minkowski distance.
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

def manhattan_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)):
		distance += abs(row1[i] - row2[i])
	return distance

def minkowski_distance(row1, row2, p):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += abs(row1[i] - row2[i])**p
    return distance**(1.0/p)

# Locate the most similar neighbors
def get_neighbors(dataset, test_row, num_neighbors, p):
    distances = list()
    for train_row in dataset:
        dist = minkowski_distance(test_row, train_row, p)
        #dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a classification prediction with neighbors
def predict_classification(dataset, test_row, num_neighbors, p):
    neighbors = get_neighbors(dataset, test_row, num_neighbors, p)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """
    n = len(y_true)
    err = [y_true[i] != y_pred[i] for i in range(n)]
    return sum(err) / n
    raise Exception('Function not yet implemented!')
    
def confusionMatrixCalculation(p_labels,t_labels):
    pred_labels = np.asarray(p_labels)
    true_labels = np.asarray(t_labels)
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    ##print ('TP: %i, FP: %i, TN: %i, FN: %i' % (TP,FP,TN,FN))
    return [[TP, FN] , [FP, TN]]                      #Mainting the order as per scikit learns implementation

if __name__ == '__main__':
    dataset = np.genfromtxt('heart2.csv', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    y = dataset[1:, 11]
    x = dataset[1:, 1:11]
    min_k=[]
    
    #scikit's train test split for model training and testing
    Xtrn, Xtst, Ytrn, Ytst = train_test_split(x, y)

    mini = Xtrn.min(axis=0)
    maxi = Xtrn.max(axis=0)
    
    
    #applying mix max scaler/normalization
    Xtrn = (Xtrn-Xtrn.min(axis=0))/(Xtrn.max(axis=0)-Xtrn.min(axis=0))
    
    mini = Xtst.min(axis=0)
    maxi = Xtst.max(axis=0)

    Xtst = (Xtst-Xtst.min(axis=0))/(Xtst.max(axis=0)-Xtst.min(axis=0))

    Y_train = Ytrn.reshape(len(Ytrn), 1)
    dataset = np.concatenate((Xtrn,Y_train),axis=1)
    Y_test = Ytst.reshape(len(Ytst), 1)
    tdataset = np.concatenate((Xtst,Y_test),axis=1)

    #p=1 for manhattan distance, p=2 for euclidean distance ,p=3 for Minkowski distance 
    error_rate=[]
    k_values=[]
    for j in range(1,31):
        knn = KNeighborsClassifier(n_neighbors=j, p = 2)
                                  # metric='minkowski')
        knn.fit(Xtrn,Ytrn)
        predicted= knn.predict(Xtst) 
        tst_err2 = compute_error(Ytst, predicted)        
        #conf = confusionMatrixCalculation(predicted,Ytst)
       

        
        print("neibhors=",j)
        print('Scikit Test Error = {0:4.2f}%.'.format(tst_err2 * 100))

        prediction= list()
        k=10
        for i in range(len(tdataset)):
            prediction.append(predict_classification(dataset, tdataset[i], j, p=2))
        
        tst_err = compute_error(Ytst, prediction)       
       

        print('My Test Error = {0:4.2f}%.'.format(tst_err * 100))
        print("---------------------------")
        #accuracy.append(1-tst_err)
        error_rate.append(tst_err*100)
        k_values.append(j)

    min_tst_err = min(error_rate)
    mean=[]
    for index in range(len(error_rate)):
        if error_rate[index] == min_tst_err:
            mean.append(k_values[index])
    
    min_k.append(sum(mean)/len(mean))

print("Mean Value of K with same minimum Test Error is {}".format(sum(min_k)/len(min_k)))
