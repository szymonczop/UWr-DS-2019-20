#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:48:55 2020

@author: czoppson
"""

import numpy as np
import matplotlib.pyplot as plt

def rosenbrock_v(x):
    """Returns the value of Rosenbrock's function at x"""
    return (1 - x[0])**2 + 100*(x[1]- x[0]**2)**2

rosenbrock_v([1,1])


def rosenbrock_hessian(x):
    #TODO: compute the value, gradient and Hessian of Rosenbrock's function'
        val = rosenbrock_v(x)
        dVdX= np.array([2*(200*x[0]**3 - 200*x[0]*x[1] + x[0]-1),200*(x[1]-x[0]**2)])
        H = np.array([[1200*x[0]**2-400*x[1] + 2,-400*x[0]],
                      [-400*x[0],200]])
        return [val,dVdX, H]
    
rosenbrock_hessian([1,1])[2]

#x = [1,1]
#
#def Newton(f, Theta0, alpha, stop_tolerance=1e-2, max_steps=1000):
#    
#    # TODO:
#    #  - implement the newton method and a simple line search
#    #  - make sure your function is resilient at critical points (such as seddle points)
#    #  - if the Newton direction is not minimizing the function, use the gradient for a few steps
#    #  - try to beat L-BFGS on the bmber of function evaluations needed!
#        history = []
#        i = 0 
#        x_start = Theta0
#        history.append(x_start)
#        while True:
#            i += 1
#            grad = f(x_start)[1]
#            H = f(x_start)[2]
#            if np.all(np.linalg.eigvals(H))>0:
#                x_start = x_start - alpha* (np.linalg.inv(H) @ grad)
#                history.append(x_start)
#                
#                print(f" iteracja {i}, jeste w punkcie {x_start}")
#                
#                if(grad @ grad < stop_tolerance or i > max_steps):
#                    break
#            else:
#                x_start  = x_start - alpha*grad
#                history.append(x_start)
#                
#                print(f" iteracja {i}, jeste w punkcie {x_start}")
#                
#                if(grad @ grad < stop_tolerance or i > max_steps):
#                    break
#
#        Theta = x_start
#        fun_evals = i 
#
#    
#        return Theta, history, fun_evals
#
#x_start = [0.,2.]
#Xopt, Xhist, fun_evals = Newton(rosenbrock_hessian, x_start, alpha=1e-2, stop_tolerance=1e-2, max_steps=10000)
#
#Xhist_ = np.array([[x[0], x[1]] for x in Xhist])
#
#print("Found optimum at %s in %d steps (%d function evals)(true minimum is at [1,1])" % (Xopt, len(Xhist), fun_evals))
#
#MX,MY = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
#Z = np.array([MX,MY]).reshape(2,-1)
#VR = rosenbrock_v(Z)
#plt.contour(MX,MY,VR.reshape(MX.shape), 100)
#plt.plot(Xhist_[:,0], Xhist_[:,1], '*-k')
#
#
#
#def f(x):
#    #TODO: compute the value, gradient and Hessian of Rosenbrock's function'
#    val = rosenbrock_v(x)
#    dVdX= np.array([2*(200*x[0]**3 - 200*x[0]*x[1] + x[0]-1),200*(x[1]-x[0]**2)])
#    H = np.array([[1200*x[0]-400*x[1] + 2,-400*x[0]],
#                  [-400*x[0],200]])
#    return [val,dVdX, H]
    

def Newton2(f, Theta0, alpha, stop_tolerance=1e-2, max_steps=1000):
    
    # TODO:
    #  - implement the newton method and a simple line search
    #  - make sure your function is resilient at critical points (such as seddle points)
    #  - if the Newton direction is not minimizing the function, use the gradient for a few steps
    #  - try to beat L-BFGS on the bmber of function evaluations needed!
        history = []
        i = 0 
        x_start = Theta0
        history.append(x_start)
        while True:
            i += 1
            grad = f(x_start)[1]
            H = f(x_start)[2]
            if np.all(np.linalg.eigvals(H))>0:
                
                direc_len = alpha* (np.linalg.inv(H) @ grad)
                x_start = x_start - direc_len
                x_step_forward = x_start - direc_len
                while f(x_step_forward)[0]<f(x_start)[0]:
                    x_start = x_step_forward
                    x_step_forward = x_start - direc_len
                    
                
#                while sum( sign_of_gradient == np.sign(f(x_start)[1]))== len(sign_of_gradient):
#                    x_start = x_start - direc_len
                
                #x_start = x_start - alpha* (np.linalg.inv(H) @ grad)
                
                
                
                history.append(x_start)
                
                print(f" iteracja {i}, jeste w punkcie {x_start}")
                
                if(grad @ grad < stop_tolerance or i > max_steps):
                    break
            else:
#                sign_of_gradient = np.sign(grad)
#                x_start  = x_start - alpha*grad
#                while  sum( sign_of_gradient == np.sign(f(x_start)[1]))== len(sign_of_gradient):
#                    x_start  = x_start - alpha*grad
                x_start = x_start - alpha*grad
                x_step_forward = x_start - alpha*grad
                while f(x_step_forward)[0]<f(x_start)[0]:
                    x_start = x_step_forward
                    x_step_forward = x_start - direc_len
                
                
                history.append(x_start)
                
                print(f" iteracja {i}, jeste w punkcie {x_start}")
                
                if(grad @ grad < stop_tolerance or i > max_steps):
                    break

        Theta = x_start
        fun_evals = i 

    
        return Theta, history, fun_evals


x_start = [0,2]
Xopt, Xhist, fun_evals = Newton2(rosenbrock_hessian, x_start, alpha=1e-2, stop_tolerance=1e-2, max_steps=10000)

Xhist_ = np.array([[x[0], x[1]] for x in Xhist])

print("Found optimum at %s in %d steps (%d function evals)(true minimum is at [1,1])" % (Xopt, len(Xhist), fun_evals))

MX,MY = np.meshgrid(np.linspace(-2,2,100), np.linspace(-2,2,100))
Z = np.array([MX,MY]).reshape(2,-1)
VR = rosenbrock_v(Z)
plt.contour(MX,MY,VR.reshape(MX.shape), 100)
plt.plot(Xhist_[:,0], Xhist_[:,1], '*-k')
