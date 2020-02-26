#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:59:35 2019

@author: czoppson
"""

def Fib(n): 
    if n<2: 
        return n
    
    else: 
        return Fib(n-1)+Fib(n-2) 
  
# Driver Program 
  

f = lambda x: - x**2 + 21.6*x + 3


import numpy as np

def fib_search(f,a,b,n,eps = 1e-4):
    
    n = n+3
    
    x1 = a + (Fib(n-2)*(b-a))/Fib(n)
    x2 = a + (Fib(n-1)*(b-a))/Fib(n)
    
    if b-a < eps:
            return[a,b]
    
    for i in range(1,n-2):
        
        
        yx1 = f(x1)
        yx2 = f(x2)
        
        if yx1 < yx2:
            a = x1
            x1 = x2
            x2 = a + (Fib(n-1-i)*(b-a))/Fib(n-i)
        
        else:
            b = x2
            x2 = x1
            x1 =a + (Fib(n-2-i)*(b-a))/Fib(n-i)
        
        print(f"[{round(a,4)},{round(b,4)}] with len {round(b-a,5)}, iter:{i}")
    
    return [a,b], round((a+b)/2,4)



f2 = lambda x:  x**2 - np.exp(x)
f3 = lambda x: - x**2 - 3000- np.log2(x)
fib_search(f,0,100,20,eps = 1e-4)   

 
fib_search(f2,0,3,9,eps = 1e-4)  
        
        
#### jakie sa te odległości w fibonaccim 

n = 10
b = 20
a = 0 
x1 = a + (Fib(n-2)*(b-a))/Fib(n)
x2 = a + (Fib(n-1)*(b-a))/Fib(n)

[a,x1,x2,b]

(x1-a)/(b - x1)

from math import * 


def gs_search(f,a,b,n):
    
    gr = (sqrt(5) + 1) / 2
    
    x1 = b - (b - a) / gr # dzieli tak samo jak fibbonci
    x2 = a + (b - a) / gr
    
    for i in range(n):
        
        if f(x1) < f(x2):
            a = x1
        else:
            b = x2
            
        x1 = b - (b - a) / gr # dzieli tak samo jak fibbonci
        x2 = a + (b - a) / gr
        
        print(f"[{round(a,4)},{round(b,4)}] with len {round(b-a,5)}, iter:{i}")
        
    return[a,b]
        



eps = 1e-4
n =round( 20/(eps*log(gr)))
gs_search(f,0,20,30)
fib_search(f,0,20,30,eps = 1e-4)  
        
    
    [a,x1,x2,b]
    
        
    