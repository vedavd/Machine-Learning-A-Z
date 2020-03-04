# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:14:59 2020

@author: vedav
"""

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Read the dataset
dataset=pd.read_csv("2-SimpleLinearRegressionSalary_Data.csv")

#Create independent variable i.e. Years of Experience
X=dataset.iloc[:,:-1].values


#Create dependent variable i.e. Salary
y=dataset.iloc[:,1].values


#Create Train and Test Data Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


#Fit Simple Regression to Training Dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predict the Test Set results
y_pred=regressor.predict(X_test)


#Visualizing the results
#First,plot the original points
# The below will plot the training datapoints in red color
plt.scatter(X_train,y_train, color='red')
#The below will plot the Xtrain and its predicted values by the model
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Training Set Data Results")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#Visualizing test data points
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_pred)
plt.title("Test Data Results")
plt.xlabel("Years of Expereince")
plt.ylabel("Salary")
plt.show()