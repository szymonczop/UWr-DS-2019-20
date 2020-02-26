#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 15:05:42 2019

@author: czoppson
"""

from gurobipy import *

######A


m = Model("a")

x = m.addVar(lb = 1, name="x")
y = m.addVar(lb = 2, name="y")

m.setObjective(x + y, GRB.MINIMIZE)

m.addConstr(x + 2 * y  == 7, "c0")

m.optimize()

for v in m.getVars():
    print(v.varName, v.x)
print('Obj:', m.objVal)


###########B

m = Model("b")

x = m.addVar(lb = 0, name="x")
y = m.addVar(lb = 0, name="y")
z = m.addVar(lb = 0, name="z")

m.setObjective(x + 2*y + 3*z, GRB.MINIMIZE)

m.addConstr( x + y <= 3 , "c0")
m.addConstr( 2 <= x + y  , "c5")
m.addConstr( x + z <= 5 , "c1")
m.addConstr( 4 <= x + z  , "c6")
m.addConstr( x - y <= 2 , "c3")

m.update()

m.optimize()


for v in m.getVars():
    print(v.varName, v.x)
print('Obj:', m.objVal)



###############C

m = Model("c")

x = m.addVar(lb = float('-inf'), name="x")
y = m.addVar(lb = float('-inf'), name="y")

m.setObjective(2*x + y , GRB.MAXIMIZE)

m.addConstr( x + y <= 5 , "c0")
m.addConstr( 1 <=  x + y  , "c1")
m.addConstr( x - y <= 4 , "c2")

m.update()

m.optimize()


for v in m.getVars():
    print(v.varName, v.x)
print('Obj:', m.objVal)


















