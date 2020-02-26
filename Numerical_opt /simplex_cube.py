#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 23:25:13 2019

@author: czoppson
"""

from mpl_toolkits.mplot3d import axes3d,art3d
import  matplotlib.pyplot as plt
import numpy as np 

figure = plt.figure()
axis = figure.add_subplot(111, projection = '3d')

x = [0,4,4,1,0,0,0,0,0,0,1,0,0,4,4,0]
y = [0,0,0,0,0,1,4,4,0,0,0,1,4,0,0,4]
z = np.array([[0,0,1,4,4,4,1,0,0,4,4,4,1,1,0,0],[0,0,1,4,4,4,1,0,0,4,4,4,1,1,0,0]])

axis.plot_wireframe(x, y, z)

axis.set_xlabel('x-axis')
axis.set_ylabel('y-axis')
axis.set_zlabel('z-axis')

plt.show()




################## Sposóv Szymona 
walls = [
    [
        [0, 0, 4],
        [1, 0, 4],
        [0, 1, 4]
    ],
    [
        [0, 0, 0],
        [0, 4, 0],
        [4, 0, 0],
    ],
    [
        [4, 0, 0],
        [4, 0, 1],
        [0, 4, 1],
        [0, 4, 0]
    ],
    [
        [1, 0, 4],
        [4, 0, 1],
        [0, 4, 1],
        [0, 1, 4]
    ],
    [
        [0, 0, 0],
        [0, 0, 4],
        [0, 1, 4],
        [0, 4, 1],
        [0, 4, 0]
    ],
    [
        [0, 0, 0],
        [0, 0, 4],
        [1, 0, 4],
        [4, 0, 1],
        [4, 0, 0]
    ]
]
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
ax.set_xlim3d(0, 4)
ax.set_ylim3d(0, 4)
ax.set_zlim3d(0, 4)

for wall in walls:
    tri = art3d.Poly3DCollection([wall])
    tri.set_alpha(0.8)
    tri.set_edgecolor('k')
    ax.add_collection3d(tri)
    
    
    
    
    
    
    
figure = plt.figure()
axis = figure.add_subplot(111, projection = '3d')

x = [0,5,5,0,0]
y = [0,0,0,0,0]
z = np.array([[0,0,5,5,0],[0,0,5,5,0]])

axis.plot_wireframe(x, y, z)

axis.set_xlabel('x-axis')
axis.set_ylabel('y-axis')
axis.set_zlabel('z-axis')

plt.show()