#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:59:36 2019

@author: czoppson
"""

from sklearn import datasets
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


iris = datasets.load_iris()
X, y = iris.data, iris.target

iris.target_names #array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
iris.target # [0,2,3,,,,]
iris.feature_names # ['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']

# print the iris data (top 5 records)
print(iris.data[0:5])

iris.target_names # nazwy moich celów

data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

data.head()

X = data.iloc[:,0:-1]
Y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100,criterion = 'entropy')

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

sum(y_pred == y_test)/len(y_pred)

clf.predict([[3,5,4,2]])



import pandas as pd
feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
feature_imp

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# mogę wyrzucić nieistotną zmienną i czyli Sepla width i stworzyć model od początku 


#############################

#decision tree Nie random forest od razu zastosowałem pruning

import sklearn.datasets as datasets
import pandas as pd
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.tree import DecisionTreeClassifier




iris=datasets.load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target


dtree=DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)#min_samples_leaf = 20)
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)


sum(y_pred == y_test)/len(y_pred)



dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = iris.feature_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())



# rzeczywiście potwierdza się to że najważniejszymi wiadomościami
# z random forest jest petal width i length, a potwierdza się to faktem ze najczęściej występują w naszym drzewie decyzyjnym s

 def dt_depth_accuracy(max_depth):
    iris=datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
    dtree=DecisionTreeClassifier(criterion = 'entropy',max_depth = max_depth)
    dtree.fit(X_train,y_train)
    y_pred = dtree.predict(X_test)
    return sum(y_pred == y_test)/len(y_pred)

accuracy_3 = []
for i in range(0,100):
    accuracy_3.append(dt_depth_accuracy(3))
    print(f'jestem na {i + 1} iteracji')

%matplotlib inline

sns.distplot(accuracy)
plt.hist(accuracy,bins = 5)
plt.hist(accuracy_3,bins = 5)
    
    
