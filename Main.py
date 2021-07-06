# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:20:36 2021

@author: 1996b
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from NaiveBayes import NaiveBayes
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as cf


            
def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy   
      

def confusion_matrix(ytest,ypred):
    y_pred = np.asarray(ypred)
    y_test = np.asarray(ytest)
    
     
    #finding the True Positive
    true_positive = np.sum(np.logical_and(y_pred == 1 , y_test == 1))
    
     #finding the True Negative
    true_negative = np.sum(np.logical_and(y_pred == 0, y_test == 0))
    
    #finding the false positive
    false_positive = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    
     #finding the false negative
    false_negative = np.sum(np.logical_and(y_pred == 0, y_test == 1))
    
    return [[true_positive,false_positive],[false_negative,true_negative]]

    
def bagging(x,y,num_models,bag_size):
    alpha=1
    models=[]
    size = bag_size
    
    for _ in range(num_models):
         random_indexes_from_dataset=np.random.choice(len(x),size,replace=True,p=None)
         NB = NaiveBayes()
         NB.fit(x[random_indexes_from_dataset], y[random_indexes_from_dataset])
         models.append((alpha,NB))
         
    return models

def plot_roc_(false_positive_rate,true_positive_rate,roc_auc):
    plt.figure(figsize=(5,5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    dataset = pd.read_csv("heart2.csv")
    
    #print(dataset.corr(method="spearman"))
    
    #renaming the column output as target
    dataset = dataset.rename({"output":"Target"},axis = 1)
    dataset = dataset.rename({"target":"Target"},axis = 1)
    #discarding Target/y column from the dataset for Feature set
    dataset = dataset.drop(['age',"fasting blood sugar"],axis = 1)
    #dataset["cholesterol"].replace(to_replace=0,value =dataset["cholesterol"].mean(),inplace=True)
    x= dataset.drop('Target', axis=1)
    
    #assigning the Target Values to Y.
    y = dataset['Target']
    
    
    
    X_train , X_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform( X_train )
    X_test = scaler.transform( X_test )
    
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Scikit Learn's Library")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy is: {0:4.2f}%".format(accuracy(y_test,y_pred)*100))
    print("----------------------")
    
    
    X_train , X_test , y_train , y_test = np.asarray(X_train) , np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)
    
    XTST=X_test
    NB = NaiveBayes()
    NB.fit(X_train,y_train)
    
    
    y_pred = NB.predict(X_test)
    #print(classification_report(y_test,y_pred))
    #print(cf(y_test,y_pred))
    
    print("Naive Bayes Algo:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy is: {0:4.2f}%".format(accuracy(y_test,y_pred)*100))
  
    
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,y_pred)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # plot_roc_(false_positive_rate,true_positive_rate,roc_auc)

    print("----------------------")
    
    
  
    models = bagging(X_train,y_train,8,100)
    predicted_values=[]
    
    predicted_values = [model[1].predict(X_test) for model in models]
    final_pred =[]
    for index in range(len(predicted_values[0])):
        predicted=[]
        for num in range(len(predicted_values)):
            predicted.append(predicted_values[num][index])
            
        final_pred.append(Counter(predicted).most_common(1)[0][0])
        
    # for x in X_test:
    #     predicted=[]
    #     for index in range(len(models)):
    #         model=models[index][1]
    #         predicted.append(model.predict([x]))
        
    #     predicted_values.append(Counter(predicted).most_common(1)[0][0])
    
    print("Our Bagging Algorithm: ")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy(y_test,y_pred)*100)
    #print(classification_report(y_test,y_pred))
    print("----------------------")
    
    print("Scikit's Bagging with Naive Bayes:")
    bag_model=BaggingClassifier(base_estimator=GaussianNB(), n_estimators=8, bootstrap=True,max_samples=100)
    bag_model=bag_model.fit(X_train,y_train)
            
    y_pred=bag_model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy is: {0:4.2f}%".format(accuracy(y_test,y_pred)*100))
    #print(classification_report(y_test,y_pred))
    print("----------------------")
    
    
    x=np.asarray(x)
    y=np.asarray(y)
    indices = np.asarray(list(range(0,918)))
    random_indexes_from_dataset=np.random.choice(len(x),int(0.8*len(x)),replace=True,p=None)
    testing_indices = np.asarray(list(set(indices).difference(set(random_indexes_from_dataset))))
    
    random_testing_indices=np.random.choice(len(testing_indices),int(0.2*len(x)),replace=True,p=None)
    
    random_testing_indices = testing_indices[random_testing_indices]
    
    
    #print(set(random_indexes_from_dataset).intersection(set(random_testing_indices)))
    
    X_train =x[random_indexes_from_dataset]
    X_test = x[random_testing_indices]
    y_train = y[random_indexes_from_dataset]
    y_test = y[random_testing_indices]
    
 
    NB = NaiveBayes()
    NB.fit(X_train,y_train)
    y_pred = NB.predict(X_test)
    
    
    print("Naive Bayes with Manual Train_test_split")
    print(confusion_matrix(y_test, y_pred))
    #print(set(y_test).intersection(set(y_pred)))
    print("Accuracy is: {0:4.2f}%".format(accuracy(y_test,y_pred)*100))
  
    
  
    
    
    
    
    
    
    
    
    
    
    
         
    
    
    
   
    
        

    
    
    
    
    
         
        

        
    
    
    
   