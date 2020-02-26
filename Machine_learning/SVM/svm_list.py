import sklearn.datasets
import sklearn.ensemble
import sklearn.svm
import sklearn.tree
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


iris = sklearn.datasets.load_iris()
print('Features: ', iris.feature_names)
print('Targets: ', iris.target_names)
petal_length = iris.data[:,iris.feature_names.index('petal length (cm)')]
petal_width = iris.data[:, iris.feature_names.index('petal width (cm)')]

IrisX = np.array(iris.data.T)
IrisX = IrisX[:, iris.target!=0]

IrisX2F = np.vstack([petal_length, petal_width])
IrisX2F = IrisX2F[:, iris.target!=0]

# Set versicolor=0 and virginia=1
IrisY = (iris.target[iris.target!=0]-1).reshape(1,-1).astype(np.float64)

plt.scatter(IrisX2F[0,:], IrisX2F[1,:], c=IrisY.ravel(),
            cmap='spring', edgecolors='k')
plt.xlabel('petal_length')
plt.ylabel('petal_width')










import sklearn.datasets
import sklearn.ensemble
import sklearn.svm
import sklearn.tree



svm_model = sklearn.svm.SVC(C = 10,kernel = 'linear')
svm_model.fit(IrisX2F.T,IrisY.T)

print("libsvm error rate: %f" % ((svm_model.predict(IrisX2F.T)!=IrisY).mean(),))


petal_lengths, petal_widths = np.meshgrid(np.linspace(IrisX2F[0,:].min(), IrisX2F[0,:].max(), 100),
                                          np.linspace(IrisX2F[1,:].min(), IrisX2F[1,:].max(), 100))

IrisXGrid = np.vstack([petal_lengths.ravel(), petal_widths.ravel()])
predictions_Grid = svm_model.predict(IrisXGrid.T)

plt.contourf(petal_lengths, petal_widths, predictions_Grid.reshape(petal_lengths.shape), cmap='spring')

plt.scatter(IrisX2F[0,:], IrisX2F[1,:], c=IrisY.ravel(),
            cmap='spring', edgecolors='k')

plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision boundary found by libsvm')
    


svm_model.support_
svm_model.dual_coef_.ravel()
    
svm_support = svm_model.support_vectors_

support_vector_indices = svm_model.support_
support_vector_coefficients = svm_model.dual_coef_


plt.contourf(petal_lengths, petal_widths, predictions_Grid.reshape(petal_lengths.shape), cmap='spring')
plt.scatter(svm_model.support_vectors_[:, 0] ,svm_model.support_vectors_[:, 1],
        cmap='spring',
        edgecolors='k',s = 10 *abs(support_vector_coefficients).ravel()  ,
        c = IrisY[:,svm_model.support_].ravel())

#################

sigma =0.1
gamma = 1/(2*sigma**2)

svm_gauss_model = sklearn.svm.SVC(C =1e6,kernel = 'rbf',gamma = gamma)
svm_gauss_model.fit(IrisX2F.T,IrisY.T)

support_vector_coefficients_gauss = svm_gauss_model.dual_coef_

print("libsvm error rate: %f" % ((svm_gauss_model.predict(IrisX2F.T)!=IrisY).mean(),))


petal_lengths, petal_widths = np.meshgrid(np.linspace(IrisX2F[0,:].min(), IrisX2F[0,:].max(), 100),
                                          np.linspace(IrisX2F[1,:].min(), IrisX2F[1,:].max(), 100))

IrisXGrid = np.vstack([petal_lengths.ravel(), petal_widths.ravel()])
predictions_Grid = svm_gauss_model.predict(IrisXGrid.T)

plt.contourf(petal_lengths, petal_widths, predictions_Grid.reshape(petal_lengths.shape), cmap='spring')

plt.scatter(IrisX2F[0,:], IrisX2F[1,:], c=IrisY.ravel(),
            cmap='spring', edgecolors='k')

plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Decision boundary found by libsvm')



plt.contourf(petal_lengths, petal_widths, predictions_Grid.reshape(petal_lengths.shape), cmap='spring')
plt.scatter(svm_gauss_model.support_vectors_[:, 0] ,svm_gauss_model.support_vectors_[:, 1],
        cmap='spring',
        edgecolors='k',s = 50,
        c = IrisY[:,svm_gauss_model.support_].ravel())


abs(support_vector_coefficients_gauss.ravel())



#####################################

import seaborn as sns

res = []
for rep in range(100):
    perm = np.random.permutation(100)
    x_train= perm[:80]
    x_test = perm[80:]
    bootstrap_sel = np.random.choice(x_train,80)
    test_sel = x_test

    bootstrap_IrisX = IrisX[:,bootstrap_sel]
    bootstrap_IrisY = IrisY[:,bootstrap_sel]
    
    test_IrisX = IrisX[:,test_sel]
    test_IrisY = IrisY[:,test_sel]
    
    #
    # TODO: Loop over a list of exponents.
    #
    for Cexponent in np.arange(-4,6.1,0.5):
        C = 10.0**Cexponent
        svm_model = sklearn.svm.SVC(C =C,kernel = 'rbf')
        svm_model.fit(bootstrap_IrisX.T, bootstrap_IrisY.T )
        train_acc = svm_model.score(bootstrap_IrisX.T,bootstrap_IrisY.T)
        test_acc = svm_model.score(test_IrisX.T,test_IrisY.T )
        
        res.append(dict(Cexponent=Cexponent, err=1-test_acc, subset='test'))
        res.append(dict(Cexponent=Cexponent, err=1-train_acc, subset='train'))

res = pd.DataFrame(res)
chart = sns.catplot(kind='box', x='Cexponent', y='err', col='subset', 
            color='blue', data=res)
chart.set_xticklabels(rotation=45)
None











X  = IrisX2F[0,:]
Y = IrisX2F[1,:]

def GaussianKernel(x,b):
    K = (1/np.sqrt(2*np.pi))*np.exp(-0.5 * (x/b)**2)
    return K 


b = 10

kernelEstimateX  = np.arange(3,7,0.01)
matrix = np.zeros((len(kernelEstimateX ),2))

i = 0 
for x_i in kernelEstimateX:
    xx = X-x_i
    K = GaussianKernel(xx,b)
    Ksum  = np.sum(K)
    weight = K/Ksum
    yk = sum(weight*Y)
    xkyk = np.array([x_i,yk])
    matrix[i,:] = xkyk
    i += 1


plt.scatter(IrisX2F[0,:], IrisX2F[1,:], c=IrisY.ravel(),
            cmap='spring', edgecolors='k')
plt.plot(matrix[:,0],matrix[:,1])
plt.xlabel('petal_length')
plt.ylabel('petal_width')

