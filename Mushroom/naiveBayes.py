'''
Created on 22 Dec 2017

@author: zhi liang
'''

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
from Utils import convertToDummy
from sklearn.metrics.classification import classification_report,\
    confusion_matrix
from _functools import reduce
import statistics
import matplotlib.pyplot as plt
import numpy as np
'''
naive bayes uses individual conditional probability to estimate the likelyhood of a classification given a certain
feature. The final probability is simply the product of each conditional probability (due to independence of each feature).
Subtlety of this method involves the distribution of a feature - normal distribution, bernoulli distribution, or multi normal distribution
'''

def naiveBayes(csv_String, result_Name):
    
    naiveBayes_data = convertToDummy(csv_String, result_Name, drop = True)
    
    class_col = naiveBayes_data['class']
    del naiveBayes_data['class']
    
    correlation_matrix = naiveBayes_data.corr()

    #Remove a subset of correlated variables
    correlation_Threshold = 0.75
    correlation_matrix.loc[:,:] =  np.tril(correlation_matrix, k=-1) 

    already_in = set()
    result = []
    for col in correlation_matrix:
        perfect_corr = correlation_matrix[col][correlation_matrix[col] > correlation_Threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    for correlated_subset in  result:
        for features in correlated_subset[1:]:
            del naiveBayes_data[features]
     
    naiveBayes_data.insert(0,'class',class_col)
    #print(logReg_data)

    Y = naiveBayes_data.values[:,0]
    X = naiveBayes_data.values[:, 1:118] 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.9, random_state = 100)
    
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    #print(accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred, digits=10))

    
