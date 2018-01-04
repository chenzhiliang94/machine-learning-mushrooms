'''
Created on 1 Jan 2018

@author: zhi liang
'''


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from Utils import convertToDummy
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.decomposition import PCA

'''
logistic regression: using value based dependent variables map ->  multi-categorical result 
e.g age, height -> gender;
    gender, eye colour, skin colour -> low class, middle class, high class
'''
def logReg(csv_String, result_Name): 
    
    logReg_data = convertToDummy(csv_String, result_Name) 
    del logReg_data['class']
    correlation_matrix = logReg_data.corr()
    #correlation_matrix.to_csv('corr_out.csv')
    print(correlation_matrix)
    #correlation_matrix.to_csv('correlation_matrix')
    plt.matshow(correlation_matrix)
    plt.show()
    
    ''''Y = logReg_data.values[:,0]
    X = logReg_data.values[:, 1:118] 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.9, random_state = 100)
    
    LogReg = LogisticRegression()
    LogReg.fit(X_train, y_train)
    
    #print(LogReg.predict_proba(X_test)) #individual result probability of test X data 
    
    y_pred = LogReg.predict(X_test) #obtain Y prediction for test X data
    confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
    print(confusion_matrix)
    print(classification_report(y_test, y_pred, digits=10)) #compare Y prediction with Y actual result
'''