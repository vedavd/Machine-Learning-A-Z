# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:53:00 2020

@author: vedav
"""


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read the data
dataset=pd.read_csv("8LogisticRegressionSocial_Network_Ads_Data.csv")


#Create the independent variable. Consider only two columns Age(index 2)
# & EstimatedSalary(index 3)
X=dataset.iloc[:,[2,3]].values

#Create the dependent variable i.e. Purchased column(index 4)
y=dataset.iloc[:,4].values


#Split the data into training set and test set
#Total observations=400
#Training set=300 observations
#Test set=100 observations which is 25% of total observations
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



#We should apply feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
#Apply feature scaling on Xtrain and Xtest
#Use fittransform for the X training set
X_train=sc_x.fit_transform(X_train)
#Use transform for X test set
X_test=sc_x.transform(X_test)



#We do not apply feature scaling on dependent variable i.e. Y because it is a
#categorical variable

#Fit the logistic regression to the training dataset
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


#Predicting the test set results
y_pred=classifier.predict(X_test)


#Create confusion matrice
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


#Visualise training set results
from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title("Logistic Regression(Training Data)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()



#Visualise test set results
from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title("Logistic Regression(Test Data)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()