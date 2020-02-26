#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 21:40:16 2019

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
    #print(train_Y)
    preds = {}

    if verbose:
        print("Computing distances... ", end='')
    #
    # TODO: fill in an efficient distance matrix computation
    #  
    
    dists = np.zeros((train_X.shape[0], test_X.shape[0]))
    #print(dists)
    for i in range(test_X.shape[0]):
        #print(np.sqrt(np.sum((train_X-test_X[i,:]) * (train_X-test_X[i,:]), axis = 1)))
        dists[:,i] = np.sqrt(np.sum((train_X-test_X[i,:]) * (train_X - test_X[i,:]), axis = 1))
    
    if verbose:
        print("Sorting... ", end='')
    
    # TODO: findes closest trainig points
    # Hint: use argsort
    closest = np.argsort(dists, axis = 0)
   # sortowanie po kolumnach
    #print(closest)

    if verbose:
        print("Computing predictions...", end='')
    
    targets = train_Y[closest]
    #print(train_Y)
    #print(closest)
    #print(targets)
    #print(sstats.mode(targets[:3]))
    
    for k in ks:
        predictions = sstats.mode(targets[:k])[0]
        predictions = predictions.ravel() # daje nam jeden wektor
        preds[k] = predictions
    if verbose:
        print("Done")
    return preds


KNN(iris_x, iris_y, unknown_x, [1, 3, 5, 7])







