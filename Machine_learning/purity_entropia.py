#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:10:05 2019

@author: czoppson
"""
import numpy as np 
import matplotlib.pyplot as plt
counts = np.arange(0,1+1e-10,0.01)

def mean_err_rate(counts):
    y = []
    for x in counts:
        p1 = x
        p2 = 1-p1
        y.append(1-max(p1,p2))
    return y

mean_err = mean_err_rate(counts)
plt.plot(counts,mean_err)


def entropy(counts):
    y = []
    for x in counts:
        p1 = x
        p2 = 1-p1
        y.append(-p1*np.log2(p1+1e-10)-p2*np.log2(p2+1e-10))
    return y


def gini(counts):
    y = []
    for x in counts:
        p1 = x
        p2 = 1-p1
        y.append(1 - p1**2 - p2**2)
    return y 


ent = entropy(counts)
gini  = gini(counts)
plt.plot(counts,ent,label = 'entropy')
plt.plot(counts,gini,label = 'gini')
plt.plot(counts,mean_err,label = 'mean err rate')
plt.legend(loc='lower center', ncol=1, frameon=True)




def entropy_of_one_division(division):
    s = 0 
    n = len(division)
    classes = list(set(division))
    for c in classes:
        p = sum(division == c)/n
        ent = -p*np.log2(p)
        s += ent
    print(f"Oto jest moja entropia {s} z daną liczba obiektow do sklasyfikowania {n}")
    return s,n
        



import math  
        
def entropy_func(c, n):
    """
    The math formula
    """
    return -(c*1.0/n)*math.log(c*1.0/n, 2)

def entropy_cal(c1, c2):
    """
    Returns entropy of a group of data
    c1: count of one class
    c2: count of another class
    """
    if c1== 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
        return 0
    return entropy_func(c1, c1+c2) + entropy_func(c2, c1+c2)





def entropy_of_one_division_1(division): 
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:   # for each class, get entropy
        n_c = sum(division==c)
        e = n_c*1.0/n * entropy_cal(sum(division==c), sum(division!=c)) # weighted avg
        s += e
    return s, n


division  = np.array([1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1])

entropy_of_one_division_1(division)
entropy_of_one_division(division)
    