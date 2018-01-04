'''
Created on 2 Jan 2018

@author: zhi liang
'''
import pandas as pd

def convertToDummy(csv_String, result_Name):
    data = pd.read_csv(csv_String) #read data
    result = data[result_Name] #identify result column
    
    del data[result_Name] #remove result column
    del data['veil-type']
    data = pd.get_dummies(data) #convert all dependent variables to dummy variables
    #del data['odor_n']
    data.insert(loc=0, column = result_Name, value=result) #insert back result column
    return data
    

def valueOf(index, anotherArray):
    return anotherArray[index]