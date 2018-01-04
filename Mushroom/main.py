'''
Created on 15 Dec 2017

@author: zhi liang
'''

from KNearestNeighbours import KNN
from logReg import logReg
from decisionTree import decisionTree
from SGD import SGD
from naiveBayes import naiveBayes

'edible = 0'
'poisonous = 1'

fileToRead = 'E:\Kaggle\Mushroom\mushrooms.csv'
resultColumnName = 'class'
#logReg(fileToRead, resultColumnName)
decisionTree(fileToRead, resultColumnName)
#SGD(fileToRead, resultColumnName)
#KNN(fileToRead, resultColumnName, 5)
#naiveBayes(fileToRead, resultColumnName)
