# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:33:46 2020

@author: vedav
"""
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv("1DataPreprocessing.csv")

#Create Matrix of features
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3:].values

#Replace missing Age column with the mean 


#Identify the columns having null data with true or false as output
dataset.isnull()

#Identify the number of null values per column
dataset.isnull().sum()


#Replcae missing values in the columns Age(Column 1) and Salary(Column 2) with mean
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])


#Check if the nan values are replaced with mean
X

#Print the Country Column alone
X[:,:1]
X[:,0]



#Encode country column containing categorical data using LabelEncoder
from sklearn.preprocessing import LabelEncoder 
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])


X


#Now we should use dummy variables for the Country column

from sklearn.preprocessing import OneHotEncoder
#Below zero represents the column i.e. Country where OneHotEncoder should be performed
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

X

#There is no necessity to use OnehotEncoder on Y i.e. Purchased column because
#this columns contain only two different values so we can use Label Encoder
y
labelencoder_y=LabelEncoder()
y[:,0]=labelencoder_y.fit_transform(y[:,0])
y



#Splitting into training dataset and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train
X_test
y_train
y_test


#Feature Scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
