'''
Created on 1 Jan 2018

@author: zhi liang
'''

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from Utils import convertToDummy, valueOf

'''
K nearest neighbours
'''
def KNN(csv_String, result_Name, numNeighbours):
    
    KNN_data = convertToDummy(csv_String, result_Name)
    
    Y = KNN_data.values[:,0]
    X = KNN_data.values[:, 1:118] 
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 99)
    
    '''KNeighborsClassifier(n_neighbors=5, weights='uniform',
                            algorithm='auto', leaf_size=30, p=2, 
                            metric='minkowski', metric_params=None, 
                            n_jobs=1)
    '''
    
    
    KNN = KNeighborsClassifier(n_neighbors=numNeighbours, weights= 'uniform')
    KNN.fit(X_train, y_train) 
    
    #y_pred = KNN.predict(X_test)
    #print(accuracy_score(y_test, y_pred))
    wrong_Prediction = 0
    for test_data,test_result in zip(X_test, y_test):
        distances, N_neighbour_index = KNN.kneighbors([test_data]) #obtain K nearest neighbours of a give test data

        N_neighbour_index = list(N_neighbour_index.flatten()) #convert a ndarray to a single array of index of the N neighbours
                                                              #index corresponds to neighbour's index in training data
        
        neighbours_class = [valueOf(x, list(y_train)) for x in N_neighbour_index] #obtain the class value of the N neighbours
        
        agreement_percentage = neighbours_class.count(test_result)/numNeighbours #check for agreement percentage
        if (agreement_percentage < 0.5):
            wrong_Prediction+=1
    totalPrediction = len(X_test)
    print(1-wrong_Prediction/totalPrediction) #accuracy of prediction

    #print(KNN.predict_proba(X_test))

