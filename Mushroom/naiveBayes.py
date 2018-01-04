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

'''
naive bayes uses individual conditional probability to estimate the likelyhood of a classification given a certain
feature. The final probability is simply the product of each conditional probability (due to independence of each feature).
Subtlety of this method involves the distribution of a feature - normal distribution, bernoulli distribution, or multi normal distribution
'''

def naiveBayes(csv_String, result_Name):
    
    naiveBayes_data = convertToDummy(csv_String, result_Name)
    
    Y = naiveBayes_data.values[:,0]
    X = naiveBayes_data.values[:, 1:118] 
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    #print(accuracy_score(y_pred, y_test))
    probability_Result = clf.predict_proba(X_test)
    'e = [0], p = [1]'
    lst = []
    for x in range(len(X_test)):
        if (y_pred[x] != y_test[x]):
            edible_Prob, poisonous_Prob = probability_Result[x]
            difference = abs(edible_Prob - poisonous_Prob)
            lst.append(difference)
    print(lst)
    #print(reduce(lambda x, y: x + y, lst)/len(lst))
    plt.hist(lst, bins='auto')
    plt.show()



    