## The Data
Taken from Kaggle ([See here](https://www.kaggle.com/uciml/mushroom-classification/data)), the data contains more than 8000 different mushrooms species, each having 23 attributes, all relating to the physical observation of a mushroom species. In addition, the data also indicates whether a species of mushroom is either poisonous or edible.

### Features
23 categorical features for each observation in this following format:
Feature Information: 

cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

bruises: bruises=t,no=f

odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

gill-attachment: attached=a,descending=d,free=f,notched=n

gill-spacing: close=c,crowded=w,distant=d

gill-size: broad=b,narrow=n

gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

stalk-shape: enlarging=e,tapering=t

stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

veil-type: partial=p,universal=u

veil-color: brown=n,orange=o,white=w,yellow=y

ring-number: none=n,one=o,two=t

ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

### Class
final categorization of a mushroom is binary:
(classes: edible=e, poisonous=p)

## The methodology
Machine learning allows us to use techniques to study a training data set and subsequently predict the class of future data not in the training data set.
In particular for this session, we will use 10%-70% of 8000 mushroom species observations for training, and the rest to test for the prediction accuracy of different machine learning alogrithms.
We will be be using 5 classifier machine learning techniques from the scikit-learn python package:
- Simple Decision Tree
- Logistic Regression
- Naive Bayes
- K Nearest Neighbours
- Stochastic gradient Descent

In addition, due to the large amount of features for each observation, we be subjecting the data to feature selection. There are 3 kinds of feature selection - filter, wrapper and embedded methods. In the 5 algorithms mentioned above, some already have implicit feature selection steps while others require some preprocessing feature selection.

## The preprocessing - part 1
We first convert all categorical features into dummy variables. [See why we need to do this](https://stats.stackexchange.com/questions/115049/why-do-we-need-to-dummy-code-categorical-variables)
To prevent multicollinearity and high correlation between features, we will have to do further feature selection (See preprocessing part 2 later)

### Simple Decision Tree
Decision trees contain implicit feature selection. In fact, the alogrithm itself focuses on feature selection - it selects a few features which allow us to decide on the final classification of a certain mushroom data. [See here for more information on the library package](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

To decide what features to select in the tree, we use the information gain criterion - [See here under Metrics - Information gain](https://en.wikipedia.org/wiki/Decision_tree_learning). Intuitively, the information gain criterion dictates which features allow us to gain the most insight from - for example, if likelihood of a poisonous mushroom to have thin stalk or fat stalk is around the same, the feature of stalk size does not give us much information; on the other hand, if 95% of poisonous mushooms have thin stalks, then the fature of stalk size give us more information gain.

Deciding on the size of the decision tree is important as well - we set the depth of the tree to log(total number of features) to prevent overfitting. There are other methods to prune the size of a decision tree such as tree-pruning. However, because setting the max depth of the tree to 5 already gives remarkably result, we will not be implementing tree-pruning here.

This is the pyplot visualisation of the tree created from the data:




