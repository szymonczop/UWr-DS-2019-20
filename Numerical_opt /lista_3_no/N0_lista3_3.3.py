#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 20:55:54 2019

@author: czoppson
"""

from sympy import * 
import numpy as np

x = Symbol('x')
f = x**4 + 16*x**2 + 18*(x-4)*exp(x)
f_prime = diff(f)
f = lambdify(x,f)
f_prime = lambdify(x, f_prime)



##### Wszytskie funkcje musza byc zdefiniowane jako fx

def my_func(x,order = 0):
    value = f(x)
    if order == 0:
        return value
    elif order ==1:
        gradient = f_prime(x)
    return value,gradient

#
#def my_func2(x,order = 0):
#    value = f(x)
#    if order == 0:
#        return value
#    elif order ==1:
#        gradient = diff(fx)
#    return value,gradient

v,g = my_func(4,1)
    


####### TA metoda znajduje mi tylko minimum 

def bisection(MIN,MAX,epsilon = 1e-5, max_iter = 65536):
    counter = 0 
    while counter <= max_iter:
        counter += 1 
        MID = (MAX + MIN) / 2
        
        value,gradient = my_func(MID,order = 1)
        
        #TODO: suboptimality 
        
        suboptimality = MAX - MIN
        
        if suboptimality <= epsilon:
            break
        
        if gradient > 0:
            MAX = MID
            
        else:
            MIN = MID
        
        print(f"Interval is {[MIN,MAX]}")
        print(f"Number of iterations {counter}")
        print(f"Suboptimal point{MID}")
        print(f"Suboptimal value {value}")
        
    return MID

bisection(-10,10)
   


###### TA FUNKCJA PIEKNIE DZIAŁA         
    
def exact_line_search(f,k,direction, eps = 1e-9,max_iter = 65536):
    counter = 0 
    alpha = 10
    x_k = k
    value,grad = my_func(k,order = 1)
    
    while counter <= max_iter:
        
        counter += 1
        x_k = x_k + alpha*direction
        value_k,grad_k = my_func(x_k,order = 1)
        
        if np.sign(grad) != np.sign(grad_k):
            direction = -1*direction
            alpha *= 0.1
        
        grad = grad_k
        
        if abs(grad) <= eps:
            break
        
#        if alpha <= eps:
#            break
        
        print(f"Number of iterations {counter}")
        print(f"my x is  {x_k}")
        print(f"line_search value {value_k}")
        
    return x_k,value_k,grad


exact_line_search(f,0,1)


#def line_search_2(f, k, direction, eps=1e-9, maximum_iterations=65536 ):
#    counter = 0 
#    value,grad = my_func(k,order = 1)
#    alpha = 100
#    if grad*direction < 0 :
#        pass
#    else:
#        direction  = -1*direction
#        
#    x_k = k + alpha*direction
#    value_k,grad_k = my_func(x_k,order = 1)
#    
#    while value_k >= value:
#        if counter >= maximum_iterations:
#            break
#        alpha = alpha - 0.001
#        x_k = k + alpha*direction
#        value_k,grad_k = my_func(x_k,order = 1)
#        counter +=1
#    
#        print(f"Number of iterations {counter}")
#        print(f"My alpha is {alpha}")
#        print(f"My value is {value_k}")
#    return x_k ,alpha


x = Symbol('x')
f = x**2
f_prime = diff(f)
f = lambdify(x,f)
f_prime = lambdify(x, f_prime)

line = lambda x: x**2

#line_search_2(line,5,1)
#line_search_2(line,0.5,1)
    



def backtracking_line_search( f, k, direction, alpha=0.4, beta=0.9, maximum_iterations=65536 ):
    counter = 0 
    value,grad = my_func(k,order = 1)
    x_k = k + alpha*direction
    value_k,grad_k = my_func(x_k,order = 1)
    
    if grad*direction < 0 :
        pass
    else:
        direction = -1*direction
    
    while value + alpha*beta*grad*direction < value_k:
        alpha = 1/2 *alpha
        x_k = k + alpha*direction
        value_k,grad_k = my_func(x_k,order = 1)
        counter += 1 
        
        print(f"my alpha {alpha}")
        
        if counter > maximum_iterations:
            break
    
    return alpha
        
        
 
x = Symbol('x')
f = x**2
f_prime = diff(f)
f = lambdify(x,f)
f_prime = lambdify(x, f_prime)       

backtracking_line_search( f, 0.01, 1, alpha=0.4, beta=0.9, maximum_iterations=65536 )
        