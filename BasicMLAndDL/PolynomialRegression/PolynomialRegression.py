# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 01:03:55 2017

@author: Aditya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
plt.scatter(X,y,color='red')
plt.title('Data Plot(Level Vs Salary)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()        

from sklearn.linear_model import LinearRegression
regressor_lin=LinearRegression()
regressor_lin.fit(X,y)
plt.scatter(X,y,color='red')
plt.plot(X,regressor_lin.predict(X),color='blue')
plt.title('Data Plot(Level Vs Salary) Linear Regression')

plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()        

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid)),1)

from sklearn.preprocessing import PolynomialFeatures
regression_poly=PolynomialFeatures(4)
X_poly=regression_poly.fit_transform(X)
regressor_lin_2=LinearRegression()
regressor_lin_2.fit(X_poly,y)
y_pred=regressor_lin_2.predict(X_poly)

plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor_lin_2.predict(regression_poly.fit_transform(X_grid)),color='blue')
plt.title('Data Plot(Level Vs Salary) Poly Regression degree=4')

plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()        
regressor_lin_2.predict(regression_poly.fit_transform(6.5))
