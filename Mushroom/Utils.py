'''
Created on 2 Jan 2018

@author: zhi liang
'''
import pandas as pd
from sklearn.decomposition.pca import PCA
from numpy import around

def convertToDummy(csv_String, result_Name, drop):
    if (drop == False):
        data = pd.read_csv(csv_String) #read data
        result = data[result_Name] #identify result column
        
        del data[result_Name] #remove result column
        data = pd.get_dummies(data) #convert all dependent variables to dummy variables
        #del data['odor_n']
        data.insert(loc=0, column = result_Name, value=result) #insert back result column
        return data
    else:
        data = pd.read_csv(csv_String) #read data
        result = data[result_Name] #identify result column
        
        del data[result_Name] #remove result column
        data = pd.get_dummies(data, drop_first = True) #convert all dependent variables to dummy variables
        data.insert(loc=0, column = result_Name, value=result) #insert back result column
        return data
    

def valueOf(index, anotherArray):
    return anotherArray[index]

#subject a training set to PCA and convert the test set
def PComponent_(train_Set, test_Set, var_Threshold = None, components = None):
    if (var_Threshold == None and components == None):
        print("please give a threshold for PComponent - either var threshold or components")
        quit()
    if (var_Threshold != None and components != None):
        print("give only one threshold")
        quit()
    if (var_Threshold != None):
        pca = PCA()
        pca.fit(train_Set)
        
        #variance ratio in percentage
        explain_Variance = around(pca.explained_variance_ratio_, decimals = 4)
        explain_Variance = explain_Variance.tolist()
        explain_Variance = [x * 100 for x in explain_Variance]
    
        #cumulative variance
        temp=0
        for x in range(len(explain_Variance)):
            explain_Variance[x] = temp + explain_Variance[x]
            temp = explain_Variance[x]
        explain_Variance = [x for x in explain_Variance if x < var_Threshold]
        n_components = len(explain_Variance)
        pca = PCA(n_components=n_components)
        return(pca.fit_transform(train_Set), pca.transform(test_Set))
    else:
        pca = PCA(n_components=components)
        return(pca.fit_transform(train_Set), pca.transform(test_Set))

