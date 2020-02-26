#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:23:30 2019

@author: czoppson
"""
import numpy as np 


min -x1 - x2
sub_to:
    x1 + x2 + x3 = 10 
    -2x1 +x2 + x4  = 4

# creating table for all needed information 
def gen_matrix(var,cons):
    tab = np.zeros((cons+1,var +cons +2))
    return tab 

# chcecking if one mre pivot is needed
# due to a negative element in the furthest 
#right column, excluding the bottom value, of 
#course.
def next_round_r(table):    
    m = min(table[:-1,-1])    
    if m>= 0:        
        return False    
    else:        
        return True
# Similarly, we’ll check to see if 1+ pivots 
#are required due to a negative element in the 
#bottom row, excluding the final value.  
def next_round(table):    
    lr = len(table[:,0])   
    m = min(table[lr-1,:-1])    
    if m>=0:
        return False
    else:
        return True

gen_matrix(2,2)

M = np.random.randint(10,size = (3,6))
M[:-1,-1]

lr = len(M[:,0]) 



