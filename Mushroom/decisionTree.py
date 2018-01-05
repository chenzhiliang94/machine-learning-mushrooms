'''
Created on 1 Jan 2018

@author: zhi liang
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pydotplus
from sklearn.model_selection import train_test_split
from Utils import convertToDummy

'''
Simple Decision Tree
'''
def decisionTree(csv_String, result_Name):
    
    simpleDecisionTree_data = convertToDummy(csv_String, result_Name, drop = False) 
    
    Y = simpleDecisionTree_data.values[:,0]
    X = simpleDecisionTree_data.values[:, 1:118] 
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.9, random_state = 100)
    
    '''DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                max_features=None, max_leaf_nodes=None, min_samples_leaf=5,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=100, splitter='best')
                '''

    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                      max_depth=5, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    
    y_pred = clf_gini.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
    #Visualise data
    dot_FileName = csv_String + '.dot'
    png_FileName = csv_String + '.png'
    tree.export_graphviz(clf_gini,feature_names=simpleDecisionTree_data.columns[1:118],
                         class_names=simpleDecisionTree_data[result_Name], filled=True, out_file=dot_FileName)
    graph = pydotplus.graph_from_dot_file(dot_FileName)
    graph.write_png(png_FileName)
    

