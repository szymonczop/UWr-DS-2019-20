#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:41:37 2019

@author: czoppson
"""
import numpy as np 
import scipy.stats as sstats
from scipy import stats
import pandas as pd

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

    if verbose:
        print("Computing distances... ", end='')
    #
    # TODO: fill in an efficient distance matrix computation
    #    
    dists = np.sqrt(np.sum((train_X[:,:,None]-test_X.transpose()[None,:,:])**2, axis=1))

    if verbose:
        print("Sorting... ", end='')
    
    # TODO: findes closest trainig points
    # Hint: use argsort
    closest = np.argsort(dists,axis=0)

    if verbose:
        print("Computing predictions...", end='')
    
    targets = train_Y[closest]

    for k in ks:
        predictions = sstats.mode(targets[:k])[0]
        predictions = predictions.ravel()
        preds[k] = predictions
    if verbose:
        print("Done")
    return preds

iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# Use read_csv to load the data. Make sure you get 150 examples!
iris_df = pd.read_csv(iris_url)

# Set the column names to
# 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']


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

x  = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])

x[:,:,None].shape # (3,3,1)
y[None,:,:].shape # (1,5,3)
y.transpose()[None,:,:].shape #(1,3,5)

x[:,:,None] - y.transpose()[None,:,:]  (3,3,5)


dip=np.sqrt(np.sum((x[:,:,None]-y.transpose()[None,:,:])**2, axis=1)) 
# sumowanie po kolumnach kazdej z  3 macierzy (3,3,5) mam na mysli pierwsza 3!
(x[:,:,None]-y.transpose()[None,:,:]).shape

np.argsort(dip) # sortowanie po wierszach 
blisko = np.argsort(dip,axis = 0 ) # sortowanie po kolumnach 
np.argsort(dip,axis = 1 ) # sortowanie po wierszach 

odp = np.array(["Marta","Szymon",'Margaret'])
cele = odp[blisko] #Macierz wypełniona odpowiedziami czyli indeks odpowiada danej wartosci w wektorze odpowiedzi 
stats.mode(cele[0:k,:])[0].ravel()