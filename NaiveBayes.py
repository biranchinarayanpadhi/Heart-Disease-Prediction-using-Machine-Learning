# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 23:43:24 2021

@author: 1996b
"""
import numpy as np

class NaiveBayes:
    
    def __init__(self):
        pass
    
    def fit(self,X,Y):
        
        #finding the number of samples and the number of feature in X
        n_samples , n_features = X.shape
        
        #finding the unique values in target
        self.classes = np.unique(Y)
        no_of_classes = len(self.classes)
        
        #initial mean , variance and priors probability of classes
        self.mean =  np.zeros((no_of_classes,n_features),dtype = np.float64)
        
        self.variance = np.zeros((no_of_classes,n_features),dtype = np.float64)
        
        self.prior_probability = np.zeros(no_of_classes,dtype = np.float64)
        
        
        #depending on the number of class 
        for index,classs in enumerate(self.classes):
            X_class = X[Y == classs]
            
            #finding the mean
            self.mean[index,:] = X_class.mean(axis = 0)
            
            #finding the variance
            self.variance[index,:] = X_class.var(axis=0)
            
            self.prior_probability[index] = X_class.shape[0] / float(n_samples)
            
            
    def predict(self,X):
        y_pred = [self.predict_helper(x) for x in X]
        return y_pred
    
    def predict_one(self,x):
        y_pred = self.predict_helper(x) 
        return y_pred
    
    
    def predict_helper(self,x):
        post_probability = []
        
        for index , classs in enumerate(self.classes):
            prior_probability = np.log(self.prior_probability[index])
            class_conditional = np.sum(np.log(self.PDF(index, x)))
            
            posterior = prior_probability + class_conditional
            post_probability.append(posterior)
            
        
        return self.classes[np.argmax(post_probability)]
            
    #Probability density function
    def PDF(self,index,x):
        
        mean = self.mean[index]
        variance = self.variance[index]
        
        numerator = np.exp(-(x-mean)**2/(2*variance))
        denominator = np.sqrt(2*np.pi*variance)
        
        #print(denominator == 0,numerator == 0)
        return numerator/denominator
    
    def pred_proba(self,X):
        ans=[]
        for x in X:
            post_probability = []
            
            for index , classs in enumerate(self.classes):
                prior_probability = np.log(self.prior_probability[index])
                class_conditional = np.sum(np.log(self.PDF(index, x)))
                
                posterior = prior_probability + class_conditional
                post_probability.append(posterior)
                
        
            ans.append(post_probability[1])    
            
        return ans