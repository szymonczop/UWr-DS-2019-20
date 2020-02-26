#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:00:18 2019

@author: czoppson"""

%matplotlib inline

import os

from io import StringIO
import itertools
import httpimport
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

import scipy.stats as sstats
import scipy.optimize as sopt

import seaborn as sns
import sklearn.tree
import sklearn.ensemble

import graphviz




import numpy as np 
import matplotlib.pyplot as plt
N = 30 
X = np.random.uniform(0,10,N)
Y = np.random.normal(1+20*X -1.3*X**2,scale= 7)

def make_dataset(N):
    X = np.random.uniform(0,10,N)
    Y = np.random.normal(1+20*X -1.3*X**2,scale= 7)
    return X, Y

data = make_dataset(30)
plt.scatter(data[0], data[1])





def powers_of_X(X, degree):
    powers = np.arange(degree + 1).reshape(-1,1)
    return (X**powers).T

power_X = powers_of_X(X,2)

theta = np.array([5,3,1])

power_X @ theta

def compute_polynomial(X, Theta):
    XP = powers_of_X(X, len(Theta) - 1) 
    Y = XP @ Theta
    return Y

plot_x_space = np.linspace(0,10,100)
plt.scatter(data[0], data[1])
for degree in range(4):
    X = powers_of_X(data[0], degree) 
    Y = data[1].reshape(-1,1)       
    Theta = np.linalg.inv(X.T @ X) @ X.T @ Y
    plt.plot(plot_x_space, compute_polynomial(plot_x_space, Theta).ravel(), 
         label="degree: %d" %(degree, ))
    print(Theta, Theta.shape)
plt.legend(loc='lower right')

################################### RIDGE REGRESSION

#
# The true polynomial relation:
# y(x) = 1 + 2x -5x^2 + 4x^3
#
# TODO: write down the proper coefficients
#
true_poly_theta = np.array([1, 2, -5, 4])

def make_dataset(N, theta=true_poly_theta, sigma=0.1):
    """ Sample a dataset """
    X = np.random.rand(N)
    Y_clean = compute_polynomial(X, theta)
    Y = Y_clean + np.random.randn(N) * sigma
    return X,Y

train_data = make_dataset(30)
XX = np.linspace(0,1,100)
#YY = compute_polynomial(XX, true_poly_theta)
plt.scatter(train_data[0], train_data[1], label='train data', color='r')
plt.plot(XX, compute_polynomial(XX, true_poly_theta).ravel(), label='ground truth')
plt.legend(loc='upper left')






def poly_fit(data, degree, _lambda):
    "Fit a polynomial of a given degree and weight decay parameter _lambda"
    X = powers_of_X(data[0], degree) # Matrix d x N
    Y = data[1].reshape(-1, 1)       # Matrix 1 x N
    #
    # TODO: implement the closed-form solution for Theta
    #
    # Please note that np.inv may be numerically unstable.
    # It is better to use np.linalg.solve or even a QR decomposition.
    lambda_matrix = np.identity((X.T @ X).shape[0],dtype = float ) * _lambda
    Theta = np.linalg.pinv(X.T@X + lambda_matrix)@X.T@Y
    return Theta


#data =  make_dataset(30)
#degree = 3
#Theta = np.linalg.solve(np.linalg.inv(X.T@X + lambda_matrix)@X.T,(1/Y).ravel())
    
num_test_samples = 100
num_train_samples = [30]
lambdas = [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
degrees = range(15)
num_repetitions = 30


#sample a single test dataset for all experiments
test_data = make_dataset(num_test_samples)
results = []



for (repetition,
     num_train,
     _lambda,
     degree,) in itertools.product(
         range(num_repetitions),
         num_train_samples,
         lambdas,
         degrees):
    train_data = make_dataset(num_train)
    Theta = poly_fit(train_data, degree, _lambda)
    
    X = powers_of_X(train_data[0], degree)
    Y = train_data[1].reshape(-1,1)
    
    X_test = powers_of_X(test_data[0], degree)
    Y_test = test_data[1].reshape(-1,1)
    
    train_err =sum((X @ Theta - Y)**2)/X.shape[0]
    test_err = sum((X_test @ Theta - Y_test)**2)/X_test.shape[0]
    
    
    results.append({'repetition': repetition,
                    'num_train': num_train,
                    'lambda': _lambda,
                    'degree': degree,
                    'dataset': 'train',
                    'err_rate': train_err[0]})
    results.append({'repetition': repetition,
                    'num_train': num_train,
                    'lambda': _lambda,
                    'degree': degree,
                    'dataset': 'test',
                    'err_rate': test_err[0]})
results_df = pd.DataFrame(results)
results_df.head()


import seaborn as sns
tips = sns.load_dataset("tips")
tips


"""magiczna funckcja to sns.relplot"""

####################### Rosenbrock function 





def rosenbrock_v(x):
    """Returns the value of Rosenbrock's function at x"""
    return (1 - x[0])**2 + 100*(x[1]- x[0]**2)**2

def rosenbrock(x):
    """Returns the value of rosenbrock's function and its gradient at x
    """
    val = rosenbrock_v(x)
    dVdX= np.array([2*(200*x[0]**3 - 200*x[0]*x[1] + x[0]-1),200*(x[1]-x[0]**2)])
    return val, dVdX

import httpimport
with httpimport.github_repo('janchorowski', 'nn_assignments', 
                            module='common', branch='nn18'):
    from common.gradients import check_gradient

rosenbrock_v([1,1])
rosenbrock([0,0])


import scipy.optimize as sopt
lbfsg_hist = []
def save_hist(x):
    lbfsg_hist.append(np.array(x))
    
rosen = lambda x: (1 - x[0])**2 + 100*(x[1]- x[0]**2)**2
rosen([1,1])

x_start = [0.,2.]
lbfsgb_ret = sopt.fmin_l_bfgs_b(rosenbrock, x_start, callback=save_hist)


path = pd.DataFrame(lbfsg_hist)

MX,MY = np.meshgrid(np.linspace(-1,2,100), np.linspace(-1,2,100))
Z = np.array([MX,MY]).reshape(2,-1)
VR = rosenbrock_v(Z)
plt.contour(MX,MY,VR.reshape(MX.shape), 100)
plt.plot(path.iloc[:,0], path.iloc[:,1], '*-k')






########################### LOG REG


from sklearn import datasets
iris = datasets.load_iris()
print('Features: ', iris.feature_names)
print('Targets: ', iris.target_names)
petal_length = iris.data[:,iris.feature_names.index('petal length (cm)')]
petal_width = iris.data[:, iris.feature_names.index('petal width (cm)')]


IrisX = np.vstack([np.ones_like(petal_length), petal_length, petal_width])
IrisX = IrisX[:, iris.target!=0]

# Set versicolor=0 and virginia=1
IrisY = (iris.target[iris.target!=0]-1).reshape(1,-1).astype(np.float64)

plt.scatter(IrisX[1,:], IrisX[2,:], c=IrisY.ravel(), cmap='spring')
plt.xlabel('petal_length')
plt.ylabel('petal_width')

def logreg_loss(Theta, X, Y):
    #
    # Write a logistic regression cost suitable for use with fmin_l_bfgs
    #

    g = lambda X,Theta : 1/ (1+np.exp(-X @ Theta))
    #nll = -1*(IrisY * np.log(g(X,Theta1)) + (1-IrisY)*np.log(1-g(X,Theta1))).sum()
    nll = -1*(Y * np.log(g(X,Theta)) + (1-Y)*np.log(1-g(X,Theta))).sum()
    #grad = (Y - g(X,Theta)).reshape(1,-1) @ X
    #grad = (IrisY - g(X,Theta1)).reshape(1,-1) @ X
    grad = (Y - g(X,Theta)).reshape(1,-1) @ X

    #reshape grad into the shape of Theta, for fmin_l_bfsgb to work
    return (nll, -grad.reshape(Theta.shape))

logreg_loss(np.array([0,0,0]),X,IrisY)

"""  Do tego momentu wszytsko mi działało i okazało się że jedyne co nie jets tak jak powinno to znak gradientu
     nie rozumiem czemu akurat w tym przypadku to nie zadziałało spytam się na zajęciach """




gradient(np.array([0,0,0]),X,IrisY)
#
#Theta0 = np.zeros((3))
#Theta = np.array([1,2,3])
#
#
lbfsg_hist2 = []
def save_hist2(x):
    lbfsg_hist2.append(np.array(x))

ThetaOpt = sopt.fmin_l_bfgs_b(lambda Theta: logreg_loss(Theta, IrisX.T, IrisY), Theta0)

plt.scatter(IrisX[1,:], IrisX[2,:], c=IrisY.ravel(), cmap='spring')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
pl_min, pl_max = plt.xlim()
pl = np.linspace(pl_min, pl_max, 1000)
plt.plot(pl, -(ThetaOpt[0]+ThetaOpt[1]*pl)/ThetaOpt[2])
plt.xlim(pl_min, pl_max)







def gradient( theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return   cost_function( theta, x, y) ,( y- sigmoid(net_input(theta,   x))) @ x

y = IrisY
net_input(np.array([0,0,0]),X)
sigmoid(X)
gradient(np.array([0,0,0]),X,IrisY)



ThetaOpt = sopt.fmin_l_bfgs_b(lambda Theta: gradient(Theta, IrisX.T, IrisY), Theta0,callback=save_hist2)[0]

plt.scatter(IrisX[1,:], IrisX[2,:], c=IrisY.ravel(), cmap='spring')
plt.xlabel('petal_length')
plt.ylabel('petal_width')
pl_min, pl_max = plt.xlim()
pl = np.linspace(pl_min, pl_max, 1000)
plt.plot(pl, -(ThetaOpt[0]+ThetaOpt[1]*pl)/ThetaOpt[2])
plt.xlim(pl_min, pl_max)


""" Teraz biorę się za tą regresje kwantylową """

import pandas as pd
import scipy.optimize as sopt
from scipy.stats import norm

data = pd.read_csv(
    'https://raw.githubusercontent.com/janchorowski/nn_assignments/nn18/assignment3/03-house-prices-outliers.csv',
    index_col=0)
data.head()

X = np.vstack((np.ones_like(data.rooms), data.area))
Y = np.asarray(data.price)[None,:]

X = X.T
Theta = np.linalg.pinv(X.T @ X) @ X.T @ Y.reshape(-1,1)


plot_x_space = np.linspace(0,250,1000)
X_lin_matrix = np.column_stack([np.ones_like(plot_x_space ),plot_x_space])
Y_val = X_lin_matrix @ Theta
plt.scatter(data['area'], data['price'],alpha = 0.3)
plt.plot(plot_x_space,Y_val.ravel() , 
         label="OLS regression",color = 'red')
plt.legend(loc='lower right')
plt.xlim([0,250])
plt.ylim([0,3000])

#### nie bedzie lekko bo sa po dwie wrtosci dla jednej
import seaborn as sns
data['price'].unique.shape
Y_hat = X @ Theta
err = Y.reshape(1,-1) - Y_hat.ravel()
#plt.hist(err)
#x = np.random.normal(size=100)
sns.distplot(err,fit =norm)
#plt.xlim(-5000,5000)
np.percentile(err,q = 75)
np.sum(err**2)

Y_hat_list = Y_hat.reshape(1,-1)
err_low_75 = Y[err<11] - Y_hat_list[err<11]
sns.distplot(err_low_75,fit =norm)
np.sum(err_low_75 **2)

pom = err

def quantile_loss(Theta,X,Y,quantile):
    value = X @ Theta
    err =  Y - value.ravel() 
    nll = (err[err>0]*quantile).sum() + (err[err<0]*(quantile-1)).sum()
    err[err>0] = quantile
    err[err<0] = quantile - 1
    theta0 = np.sum(err)
    theta1 = np.dot(err, X[:,1])
    grad = np.array([theta0,theta1[0]])
    #grad = np.array([np.sum(err), np.dot(err, X[:,1])])
    
    return(nll,grad.reshape(Theta.shape))
    
quantile_loss(np.array([0,2]),X,Y,0.3)




Theta0 = np.array([4,2])
ThetaOpt = sopt.fmin_l_bfgs_b(lambda Theta: quantile_loss(Theta0, X, Y,0.3), np.array(Theta0))[0]





gradient = np.array([np.sum(err), np.dot(err, X[:,1])])


X[:,1].shape[0]





