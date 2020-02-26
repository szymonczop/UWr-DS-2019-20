#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:30:15 2019

@author: czoppson
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

cosh = lambda x : np.cosh(x)
multi_cosh = lambda x: np.cosh(x[0]) + np.cosh(x[1])
grad_multi = lambda x : np.array([np.sinh(x[0]),np.sinh(x[1])])




MX,MY = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
Z = np.array([MX,MY]).reshape(2,-1)
VR = multi_cosh(Z)
plt.contour(MX,MY,VR.reshape(MX.shape), 100)


"""Hessian like on the exercises is like 
    [[cosh(x1),0],
    [ 0 , cosh(x2)]]
    
    our xi in in (-3,3)
    """
# we know that the maximum value of the function are on the edges
M = cosh(3)
m = cosh(0)
k = M/m
step_size = 1/M


eps=0.001

x_0 = [3,2]
x_optimal =[0,0]

boundry = (1/np.log(k/(k-1)))*np.log((multi_cosh(x_0) - multi_cosh(x_optimal))/eps )

def gradient(start_x): 
    """ta funkcja to jest najprostrzy gradient z możliwych nie ma tutaj zadnego dobotu t tylko 
        od razu narzucony z góry learning rate"""
    steps = [start_x]
    i = 0
    while True:
        i +=1
        direc = -grad_multi(start_x)
        grad_length = grad_multi(start_x) @ grad_multi(start_x)
        t =step_size
        start_x = start_x + t*direc
        steps.append(start_x)
        if(grad_length < eps):
            break
        #print(f"my points {start_x}")
    return multi_cosh(start_x), start_x, i ,steps

"""Zabawa ze zbyt duza funkcja"""

def gradient_too_BIG(start_x): 
    """ta funkcja to jest najprostrzy gradient z możliwych nie ma tutaj zadnego dobotu t tylko 
        od razu narzucony z góry learning rate"""
    steps = [start_x]
    i = 0
    while True:
        i +=1
        direc = -grad_multi(start_x)
        grad_length = grad_multi(start_x) @ grad_multi(start_x)
        t =2/M
        start_x = start_x + t*direc
        steps.append(start_x)
        if(grad_length < eps or i > 10000):
            break
        #print(f"my points {start_x}")
        print(i)
    return multi_cosh(start_x), start_x, i ,steps

val,point,iteration,steps= gradient_too_BIG([10,10]) # to juz nie zbiega 
plt.contour(MX,MY,VR.reshape(MX.shape), 100)
path = pd.DataFrame(steps)

val,point,iteration,steps= gradient([3,3])

edge = 3
num_random_points = 10
rand_points=2*edge*np.random.rand(num_random_points,2) - edge
epss=1e-3



for i in rand_points:
    boundry = (1/np.log(k/(k-1)))*np.log((multi_cosh(i) - multi_cosh(x_optimal))/eps )
    val,point,iteration,steps= gradient(i)
    plt.contour(MX,MY,VR.reshape(MX.shape), 100)
    path = pd.DataFrame(steps)
    plt.plot(path.iloc[:,0], path.iloc[:,1], '*-k')
    plt.show()
    print(f"iterations {iteration}, number of predicted evaluation {boundry}")


""" Zabawa z Rosenbrockiem To nie zawsze działa i trzeba wybrać funkcje blisko alei z minimum"""


multi_cosh = lambda x: (1 - x[0])**2 + 100*(x[1]- x[0]**2)**2
grad_multi = lambda x : np.array([2*(200*x[0]**3 - 200*x[0]*x[1] + x[0]-1),200*(x[1]-x[0]**2)])

M = multi_cosh([7,7])
m = 0.01
k = M/m
step_size = 1/M* 10**2

eps=0.001


def gradient(start_x): 
    """ta funkcja to jest najprostrzy gradient z możliwych nie ma tutaj zadnego dobotu t tylko 
        od razu narzucony z góry learning rate"""
    steps = [np.array(start_x)]
    i = 0
    while True:
        i +=1
        direc = -grad_multi(start_x)
        grad_length = grad_multi(start_x) @ grad_multi(start_x)
        t = step_size
        start_x = start_x + t*direc
        steps.append(start_x)
        if(grad_length < eps or i > 10000):
            break
        #print(f"my points {start_x}")
        print(i)
    return multi_cosh(start_x), start_x, i ,steps








MX,MY = np.meshgrid(np.linspace(-5,5,100), np.linspace(-5,5,100))
Z = np.array([MX,MY]).reshape(2,-1)
VR = multi_cosh(Z)
plt.contour(MX,MY,VR.reshape(MX.shape), 100)



val,point,iteration,steps = gradient([-5,-5])
plt.contour(MX,MY,VR.reshape(MX.shape), 100)
path = pd.DataFrame(steps)
plt.plot(path.iloc[:,0], path.iloc[:,1], '*-k')
plt.show()




""" Kończenie zadnaia """

def boyd_example_func(x, order=0):
    a=np.array([1,3])
    b=np.array([1,-3])
    c=np.array([-1,0])
    x=np.array(x)
    value = np.exp(a@x)+np.exp(b@x)+np.exp(c@x)
    if order==0:
        return value
    elif order==1:
        gradient = a.T*np.exp(a@x)+b.T*np.exp(b@x-0.1)+c.T*np.exp(c@x)
        return (value, gradient)
    elif order==2:
        gradient = a.T*np.exp(a@x)+b.T*np.exp(b@x)+c.T*np.exp(c@x)
        hessian = a.T*a*np.exp(a@x)+b.T*b*np.exp(b@x)+c.T*c*np.exp(c@x)
        return (value, gradient, hessian)
    else:
        raise ValueError("The argument order should be 0, 1 or 2")
        
        
boyd_example_func([1,2],order = 1)[1]

boyd_example_func([1,2],order = 2)

boyd_func = lambda x : boyd_example_func(x,order = 0)
grad_boyd = lambda x : boyd_example_func(x,order = 1)[1] # całkiem ważne do poniższych eksperymentów


def gradient(gradient_function,start_x,step_size): 
    """ta funkcja to jest najprostszy gradient z możliwych nie ma tutaj zadnego doboru tylko 
        od razu narzucony z góry learning rate"""
    steps = [start_x]
    i = 0
    while True:
        i +=1
        direc = -gradient_function(start_x)
        grad_length = gradient_function(start_x) @ gradient_function(start_x)
        t = step_size
        start_x = start_x + t*direc
        steps.append(start_x)
        if(grad_length < eps or i > 10000 ):
            break
       # print(f"my points {i}")
    return start_x, i ,steps

gradient(grad_boyd,[1,1],0.01)

x0= [1,1]
for step in [0.2,0.1,0.01,0.001]:
    val,iteration,steps = gradient(grad_boyd,[1,1],step)
    print(f"optimal point {val} reached after {iteration} iterations for step_size = {step} value {boyd_func(val)}" )     
print(f"starting from x_0 = {x0}")

""" comparing results with usage of bactracking search """





def back_track_multi(function,gradient_function,alpha,beta,start_x):
    t =1
    direction =np.sign(-gradient_function(start_x)) #np.sign(- grad_fun_multi(start_x))
    while function(start_x + t*direction)> function(start_x)+ alpha*t*gradient_function(start_x).dot(direction):
        t = beta*t
        #print(f" Wartość funkcji {fun_multi(start_x + t*direction)} t-value = {t},dir = {direction}")
    return t 

back_track_multi(boyd_func,grad_boyd,0.3,0.5,[0.01,0.01])


def gradient(gradient_function,start_x,eps): 
    """ta funkcja to jest najprostszy gradient z możliwych nie ma tutaj zadnego doboru tylko 
        od razu narzucony z góry learning rate"""
    steps = [start_x]
    i = 0
    while True:
        i +=1
        direc = np.sign(-gradient_function(start_x))
        grad_length = gradient_function(start_x) @ gradient_function(start_x)
        t = back_track_multi(boyd_func,grad_boyd,0.3,0.001,start_x)
        start_x = start_x + t*direc
        steps.append(start_x)
        if(grad_length < eps or i > 1000 ):
            break
        #print(f"my points {i},I'm in point {start_x}")
    return start_x, i ,steps

#back_track_multi(boyd_func,grad_boyd,0.03,0.5,x0) # działa
    

val,iteration,steps = gradient(grad_boyd,x0,0.001)

val,iteration,steps = gradient(grad_boyd,[1,1],step)
print(f"optimal point {val} reached after {iteration} iterations for step_size = {step} value {boyd_func(val)}" )     
print(f"starting from x_0 = {x0}")




def gradient(gradient_function,start_x): 
    steps = [start_x]
    i = 0
    while True:
        i +=1
        direc = np.sign(-gradient_function(start_x))
        grad_length = gradient_function(start_x) @ gradient_function(start_x)
        t =i/(i+1)
        start_x = start_x + t*direc
        steps.append(start_x)
        if(grad_length < eps or i > 1000):
            break
        print(f"my points {start_x} {i}")
    return start_x, i ,steps



gradient(grad_boyd,x0)


val,iteration,steps = gradient(grad_boyd,x0)
print(f"optimal point {val} reached after {iteration} iterations for step_size = {step} value {boyd_func(val)}" )     
print(f"starting from x_0 = {x0}")




























fun_multi = lambda x: x[0]**2 + x[1]**2
grad_fun_multi = lambda x: 2*x[0] + 2*x[1]



def back_track_multi(alpha, beta,start_x):
    t =5
    direction =np.sign(-np.array([2*start_x[0],2*start_x[1]])) #np.sign(- grad_fun_multi(start_x))
    while fun_multi(start_x + t*direction)> fun_multi(start_x)+ alpha*t*np.array([2*start_x[0],2*start_x[1]]).dot(direction):
        t = beta*t
        #print(f" Wartość funkcji {fun_multi(start_x + t*direction)} t-value = {t},dir = {direction}")
    return t 

back_track_multi(0.1,0.5,[2,2])

#
def gradient(start_x):
    while abs(grad_fun_multi(start_x))> 1e-4:
        direc = np.sign(-np.array([2*start_x[0],2*start_x[1]]))
        t = back_track_multi(0.1, 0.3,start_x)
        start_x = start_x + t*direc
        print(f"my points {start_x}")
    return fun_multi(start_x), start_x

#### inny grad 
def gradient(start_x):
    while True : #abs(grad_fun_multi(start_x))> 1e-4:
        direc = -np.array([2*start_x[0],2*start_x[1]])
        t = back_track_multi(0.1, 0.3,start_x)
        start_x = start_x + t*direc
        if direc @ direc < 0.01:
            break
        print(f"my points {start_x}")
    return fun_multi(start_x), start_x

gradient([5,5])

