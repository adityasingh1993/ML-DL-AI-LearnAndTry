# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 00:40:58 2017

@author: Aditya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Loading Data 
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values 

#Filling the missing values with mean of respective column
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

# Encoding Categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#categorizing the data into numerical format, since countries are not in number
labelEncoder_X=LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0])
#Using hot encoding to remove the confusion between numbers
oneHotencoder_x=OneHotEncoder(categorical_features=[0])
X=oneHotencoder_x.fit_transform(X).toarray()
labelEncoder_Y=LabelEncoder()
y=labelEncoder_Y.fit_transform(y)

#splitting the data into train and test
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_x.fit_transform(X)
X_test=sc_x.transform(X)
