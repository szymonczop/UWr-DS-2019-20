#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 17:09:32 2019

@author: czoppson
"""
import numpy as np 

# układ sprzeczny
A = np.array([[1,3,1],[1,1,-1],[3,11,5]])
b = np.array([9,1,35])

# najprostrzy przypadek który niczego nie psuje 
A = np.array([[60, 91, -26], [-60, 3, -75], [45, 90, 31]], dtype='float')
beta = np.array([1, 0, 0])

# przypadek z nieskończenie wieloma rozwiązaniami 
A = np.array([[60, 91, 26,8], [60, 3, 75,13], [45, 90, 31,38]], dtype='float')# nie ma problemu dla dodatkowych wymiarów 
beta = np.array([1, 0, 0])
# tojest układ sprzeczny 

A =  np.array([[60, 91], [60, 3], [45, 90],[32,12],[3,2]], dtype='float') #  nie wiem jak będzie zachowywać sie tutaj :
beta = np.array([1, 0, 0,2,7])

####
A = np.array([[1,1],[2,1],[1,2],[1,-1]],dtype = 'float')
beta = np.array([5, 7, 8,-1])


A = np.array([[2,1,-1],[-3,-1,2],[-2,1,2]])
beta = np.array([8,-11,-3])




# tutaj jeśli mam dużo więcej wierszy niż kolumn to na początek chce rozwiazać problem jak dla macierzy 
# kwadratowej a potem sprawdzić czy wyniki mi sie pokrywają
def solve(A,beta,scizor = False):
    A = A.astype(float)
    rows,col = A.shape
    if rows > col:
        pom_A  = A
        pom_beta  = beta
        scizor = True
        A = A[:col,:] # zmniejaszam do macierzy nxn
        beta = beta[:col] # zmiejszam wektor żeby zrobić hstack
    
    
    X = np.hstack([A,beta.reshape(-1,1)]) # reshape(-1,1) znaczy 'zgadnij ile mam wierszy ale ustalam że mam tylko 1 kolumne 
    dim = X.shape[0] #X[2] # trzeci wiersz w mojej macierzy.
    
    while A[1,1] ==0: # nie chce zaczynać od tego że w górnym rogu mam zero
        A = A[np.random.permutation(np.arange(dim)),:]
    
    
    for i in range(dim): # to dziła  bo jak w nastepnej pętli mam od range(dim,dim) to nie nic mi nie drukuje
        a = X[i]
        for j in range(i+1,dim):
            b = X[j]
            mnoznik = b[i]/a[i]
            X[j] = b - mnoznik*a
            
    if b[j]==0 and b[j+1] != 0: # tutaj cofam o jeden indeks bo 
        raise ValueError('Układ jest sprzeczny')
        
    X = X[~np.all(X == 0, axis=1)]
    dim = X.shape[0]
    
    for i in range(dim-1,-1,-1):
        a = X[i]
        X[i] = X[i]/X[i,i]
        for j in range(i -1,-1,-1):
            b = X[j]
            mnoznik_2= b[i]/a[i]
            X[j] = b - mnoznik_2*a
    
    if scizor == True:
        beta2 = pom_beta[col:]
        rozw = X[:,-1]
        if sum(pom_A[dim:,:] @ rozw == beta2) != len(beta2):
            raise ValueError('Układ jest sprzeczny')
            
        
    
    return X
    
    
########## testowanie (wszędzie używam randomowego generowania wektorów)
    
# macierz diagonalna
A = np.diag(np.arange(1,11))
x = np.random.rand(10)*10 -5
beta = (A @ x).reshape(1,-1) # musi być podawane jako wektor poziomy nie pionowy 

solution = solve(A,beta)

err = sum((solution[:,-1] - x)**2)

# tutaj macierz symetryczna

A = np.random.randint(1000,size = (5,5))
A = A + A.T - np.diag(A.diagonal())
x = np.random.rand(5)*100 -20
beta = (A @ x).reshape(1,-1)
solution = solve(A,beta)

err = sum((solution[:,-1] - x)**2)

A = np.random.rand(5,5)*10 -5
A = A + A.T - np.diag(A.diagonal())
x = np.random.rand(5)*100 -20
beta = (A @ x).reshape(1,-1)

solution = solve(A,beta)

err = sum((solution[:,-1] - x)**2)

# macierz 5x2 ze sprzecznym rozwiązaniem 
A =  np.array([[60, 91], [60, 3], [45, 90],[32,12],[3,2]], dtype='float') #  nie wiem jak będzie zachowywać sie tutaj :
beta = np.array([1, 0, 0,2,7])
solve(A,beta)

# macierz 5x2 z istniejącym rozwiązaniem 
A = np.array([[1,1],[2,1],[1,2],[1,-1]],dtype = 'float')
beta = np.array([5, 7, 8,-1])
solve(A,beta)


# nieskończenie wiele rozwiązań
A = np.array([[1,3,1],[1,1,-1],[3,11,5]])
beta = np.array([9,1,35])
solve(A,beta)

# najprostrzy układ równań z jednym rozwiązaniem 
# najprostrzy przypadek który niczego nie psuje 
A = np.array([[60, 91, -26], [-60, 3, -75], [45, 90, 31]], dtype='float')
beta = np.array([1, 0, 0])
solve(A,beta)

# macierz dodatnio określona 
A = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
x = np.random.rand(3)*200 -100
beta = (A @ x).reshape(1,-1)
solve(A,beta)

### wszystko działa bardzo elegancko 
    
####### tutaj znajduje mi macierz odwrotną 
A = np.array([[60, 91, 26], [60, 3, 75], [45, 90, 31]], dtype='float')
b = np.array([1, 0, 0])
X = np.hstack([A,np.diag(np.ones(len(b)))])
b_len = len(b)

for i in range(b_len): # to dziła  bo jak w nastepnej pętli mam od range(b_len,b_len) to 
    a = X[i]
    for j in range(i+1,b_len):
        b = X[j]
        #print(b)
        mnoznik = b[i]/a[i]
        X[j] = b - mnoznik*a
        
if b[j]==0 and b[j+1] != 0:
    raise ValueError('Układ jest sprzeczny')

for i in range(b_len-1,-1,-1):
    a = X[i]
    X[i] = X[i]/X[i,i]
    for j in range(i -1,-1,-1):
        b = X[j]
        mnoznik_2= b[i]/a[i]
        X[j] = b - mnoznik_2*a

print(X[:,b_len:(2*b_len)])



