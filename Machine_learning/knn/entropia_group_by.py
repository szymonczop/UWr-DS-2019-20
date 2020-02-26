#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:09:02 2019

@author: czoppson
"""

# importing pandas as pd 
import pandas as pd 
  
# Creating the dataframe  
df = pd.read_csv("nba.csv") 
  
# Print the dataframe 

df.colnames
gk = df.groupby('Team')

df.value_counts()



df = pd.DataFrame({'A':['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'], 'B':['one', 'one', 'two', 'three','two', 'two', 'one', 'three'], 'C':np.random.randn(8), 'D':np.random.randn(8)})
df.groupby('A').mean()

df.groupby('A').B.value_counts(normalize = True).sum()

df.groupby('A').C.sum()


A    B    
bar  one      1
     three    1
     two      1
foo  one      2
     two      2
     three    1
Name: B, dtype: int64



df.groupby('A').B.value_counts(normalize = True)

bar  one      0.333333
     three    0.333333
     two      0.333333
foo  one      0.400000
     two      0.400000
     three    0.200000
 

df.groupby('A').B.value_counts(normalize = True).sum()    
1.9999999999999998



def entropy(series):
    px = series.value_counts(normalize = True)
    entro = round(-(px*np.log2(px)).sum(),6)
    return entro

df.groupby(by='A').B.apply(entropy)

A
bar    1.584963
foo    1.521928
Name: B, dtype: float64






df["A"].value_counts(normalize = True)

foo    0.625
bar    0.375
Name: A, dtype: float64

df.appply(entropy,axis = 0)


df


df.apply(entropy)['D']

df[["C","D"]].apply(entropy)


df['C'].value_counts()

