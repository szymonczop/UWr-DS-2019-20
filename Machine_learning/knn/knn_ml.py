#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:49:53 2019

@author: czoppson
"""

import numpy as np
from scipy import stats
import pandas as pd


#import httpimport
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import scipy.stats as sstats

import seaborn as sns
from sklearn import datasets
from statistics import mode 

x  = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
x.shape
x_reshape = x.reshape((1,3,3))
y_reshape = y.reshape(5,1,3)


z = ((x_reshape-y_reshape)**2)[:,:1,:] 

x_r = x.reshape((3,1,3)) - y.reshape(5,1,3)

z[:,:1,:] # to mi daje tylko odległości pierwszego unktu od pozostałych 

new_mx =z[:,:1,:] 
new_mx_1 = new_mx.reshape((5,3))**2
distance =np.sqrt(list(map(sum,new_mx_1)))

np.sort(distance)
np.argsort(distance)[0:2]

target = np.random.choice([1,0],size = 5)

lista = np.array([1,2,3,2,2,5,6,2,4,4,4,4,4,4])
(stats.mode(lista)[0] == 4)==True
if stats.mode(lista)[0] == 4:
    print('Działa')
    
    
#####################
def KNN(train_X, train_Y, test_X, ks, verbose=False):
    """
    Compute predictions for various k
    Args:
        train_X: array of shape Ntrain x D
        train_Y: array of shape Ntrain
        test_X: array of shape Ntest x D
        ks: list of integers
    Returns:
        preds: dict k: predictions for k
    """
    # Cats data to float32
    train_X = train_X.astype(np.float32)
    test_X = test_X.astype(np.float32)

    # Alloc space for results
    preds = {}

    #if verbose:
    print("Computing distances... ", end='')
    
    row1 = train_X.shape[0]
    col1 = train_X.shape[1]
    row2 = test_X.shape[0]
    col2 =  test_X.shape[1]
    x_test = test_X.reshape((1, row2,col2))
    x_train =  train_X.reshape((row1,1,col1))
    dists = np.zeros((row1,row2))
     
    for i in range(row2):
          new_mx= ((x_test-x_train)**2)[:,i:i+1,:] 
          new_mx = new_mx.reshape(row1,col1)
          p = np.sqrt(list(map(sum,new_mx)))
          dists[:,i] = p 
        
    #dists =  dists to jest moja macierz gdzie w kolumy trzymają odległości dla pojedyńczej obserwacji jednego punktu 

    if verbose:
        print("Sorting... ", end='')
    
    # TODO: findes closest trainig points
    # Hint: use argsort
    #arr = np.array([1, 3, 2, 4, 5])
    #arr.argsort()[:3]
    k_macierz =pd.DataFrame(np.zeros((row2,len(ks)))) # nie potrafie w numpy 
    for i in range(row2):
        wybory  = []
        for j in ks:
            point = dists[:,i]
            closest = point.argsort()[:j]
            decisions = train_Y[closest]
            wybory.append(mode(decisions))
        k_macierz.iloc[i,:] = wybory
        print(i)
            
        
    
    #closest = 

    if verbose:
        print("Computing predictions...", end='')
    
    #targets = train_Y[closest]
    ite = 0
    for k in ks:
        #... = sstats.mode(...)
        #predictions = predictions.ravel()[]
        preds[k] = k_macierz.values[:,ite]
        ite += 1
        
        
    if verbose:
        print("Done")
    return preds




iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# Use read_csv to load the data. Make sure you get 150 examples!
iris_df = pd.read_csv(iris_url,header = None)
# Set the column names to
# 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']


#iris_df_long = iris_df.melt('target')
#iris_df_long.head()

unknown_df = pd.DataFrame(
    [[1.5, 0.3, 'unknown'],
     [4.5, 1.2, 'unknown'],
     [5.1, 1.7, 'unknown'],
     [5.5, 2.3, 'unknown']],
     columns=['petal_length', 'petal_width', 'target'])



iris_x = np.array(iris_df[['petal_length', 'petal_width']])
iris_y = np.array(iris_df['target'])

unknown_x = np.array(unknown_df[['petal_length', 'petal_width']])

KNN(iris_x, iris_y, unknown_x, [1, 3, 5, 7])

######################## MESHGRID I RYSOWANIE PRZEDIZAŁÓW#####################
########################
########################
from matplotlib.colors import ListedColormap

iris_x = np.array(iris_df[['petal_length', 'petal_width']])
iris_y = np.array(iris_df['target'])

h = .04
mesh_x, mesh_y = np.meshgrid(np.arange(iris_x[:, 0].min(), iris_x[:, 0].max() + .01, h), 
                             np.arange(iris_x[:, 1].min(), iris_x[:, 1].max() + .01, h))

#use np.unique with suitable options to map the class names to numbers
target_names, iris_y_ids = np.unique(iris_y, return_inverse = True)

mesh_data = np.hstack([mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)])
#print(mesh_data)

preds = KNN(iris_x, iris_y_ids, mesh_data, [1, 3, 5, 7])
#print(preds)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for k, preds_k in preds.items():
    plt.figure()
    plt.title(f"Decision boundary for k={k}")
    plt.contourf(mesh_x, mesh_y, preds_k.reshape(mesh_x.shape), cmap = cmap_light)
    plt.scatter(iris_x[:,0], iris_x[:,1], c = iris_y_ids, cmap = cmap_bold, edgecolor = 'k')

siatka_1,siatka_2 = np.meshgrid(np.arange(1,10+.01,.05),np.arange(50,59 +.01,0.5))






############################ KOLEJNA CZESC
preds.items()


#for k, preds_k in  preds.items():
#    print(k)
#    print(preds_k)
#    



#TODO: write a function to compute error rates
def err_rates(preds, test_Y):
    ret = {}
    for k, preds_k in  preds.items():
        len(preds_k)-sum(preds_k == test_Y)
        #ret[k] = len(preds_k)-sum(preds_k == test_Y)
        ret[k] = 1 - (np.sum(preds_k == test_Y) / len(preds_k))
    return ret



iris_x = np.array(iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
iris_y = np.array(iris_df['target'])

ks = np.arange(1, 20, 2)
results = []

#import math


ks = range(1, 20, 2)
for _rep in tqdm_notebook(range(1000)):
    #TODO
    # Use np.split and np.permutation to get training and testing indices
    #train_idx, test_idx = np.split(...)
#    kak = np.split(np.random.permutation(iris_x.shape[0]),[math.floor(iris_x.shape[0]*0.664),iris_x.shape[0]])
#    train_idx = kak[0]
#    test_idx  = kak[1]
    
    train_idx, test_idx = np.array_split(np.random.permutation(np.arange(iris_df.shape[0])), [np.int64(np.ceil(2/3*iris_df.shape[0]))])

    #TODO: apply your kNN classifier to data subset
    train_X = iris_x[train_idx,:]
    test_X = iris_x[test_idx,:]
    train_Y = iris_y[train_idx]
    
    preds = KNN(train_X,train_Y,test_X,ks,verbose = False)
    errs = err_rates(preds, iris_y[test_idx])
    
    for k, errs_k in errs.items():
        results.append({'K':k, 'err_rate': errs_k})

results_df = pd.DataFrame(results)

#err = []
#for i in np.arange(1,20,2): # jak to zrobić za pomocą pandas?
  #  err.append(results_df.loc[results_df['K']==i].sum(axis = 0)[1]/results_df.loc[results_df['K']==i].shape[0])

#df = pd.DataFrame(list(zip(err,ks)),columns = ['err','K'])
plt.figure()
sns.regplot(x ='K',y = 'err_rate' ,x_estimator=np.mean,data = results_df,order = 2)
plt.figure()
sns.regplot(x ='K',y = 'err_rate' ,x_estimator=np.mean,data = results_df,order = 0)



############################# ENTROPIAAA

columns = [
 "target", "cap-shape", "cap-surface", "cap-color", "bruises?", "odor", 
 "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape", 
 "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", 
 "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", 
 "ring-number", "ring-type", "spore-print-color", "population", "habitat", ]

# Use read_csv to load the data.
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
mushroom_df = pd.read_csv(url, header=None, names=columns)
mushroom_df.head()

mushroom_df.count()
mushroom_df.value_counts()


def entropy(series):
    px = series.value_counts(normalize = True)
    entro = round(-(px*np.log2(px)).sum(),6)
    return entro 

entropy(s)


mushroom_df.apply(entropy,axis =0 )




####################### 
#
#TODO  : Zaawansowana etropia
#
mushroom_df
df = pd.DataFrame({'A':['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'], 'B':['one', 'one', 'two', 'three','two', 'two', 'one', 'three'], 'C':np.random.randn(8), 'D':np.random.randn(8)})
df.groupby('A').mean()
df.groupby('A').B.value_counts(normalize = True).sum() #H(A|B)

B = df['B']
A = df['A']
X = A
Y = B

lol =round( B.value_counts(normalize = True)*df.groupby('A').B.value_counts(normalize = True).sum(),5)
lol.sum()
df.groupby('A').B.count()


df.groupby('X').Y.value_counts(normalize = True)*np.log2(df.groupby('X').Y.value_counts(normalize = True))


def cond_entropy(df, X, Y): #H(X|Y)
    sum_1 =  (df.groupby(X)[Y].value_counts(normalize = True)*np.log2(df.groupby(X)[Y].value_counts(normalize = True))).sum()
    cond =round(df[X].value_counts(normalize = True)*sum_1,10)
    return -cond.sum()
X ="target"
Y = 'cap-shape'
df = mushroom_df

(sum(df.groupby(by=X)[Y].apply(entropy) * mushroom_df[X].value_counts().sort_index()) / df.shape[0])
sum(-df.groupby(X)[Y].value_counts(normalize = True)*np.log2(df.groupby(X)[Y].value_counts(normalize = True)))
df.groupby(by=X)[Y].value_counts(normalize = True)


df.groupby(by=X)[Y].apply(entropy).sum() ==sum(-df.groupby(X)[Y].value_counts(normalize = True)*np.log2(df.groupby(X)[Y].value_counts(normalize = True)))




sum(df.groupby(by=X)[Y].apply(entropy) * mushroom_df[X].value_counts(normalize = True))