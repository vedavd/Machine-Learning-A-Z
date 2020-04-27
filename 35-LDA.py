# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:54:32 2020

@author: vedav
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read the data
dataset=pd.read_csv("35_LDA_Wine_Data.csv")


#Create the independent variable.(index 0 to index 12)
X=dataset.iloc[:,0:13].values

#Create the dependent variable i.e. Purchased column(index 13)
y=dataset.iloc[:,13].values


#Split the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



#We should apply feature scaling
#Feature Scaling is must before applying Dimensionality Reduction techniques such as PCA or LDA
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
#Apply feature scaling on Xtrain and Xtest
#Use fittransform for the X training set
X_train=sc_x.fit_transform(X_train)
#Use transform for X test set
X_test=sc_x.transform(X_test)



#Applying LDA
#Since the LDA is a supervised, we should include the dependent variable in the fit transform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
X_train=lda.fit_transform(X_train,y_train)
X_test=lda.transform(X_test)


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
             alpha=0.75,cmap=ListedColormap(('red','green','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green','blue'))(i),label=j)
plt.title("Logistic Regression(Training Data)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()



#Visualise test set results
from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green','blue')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green','blue'))(i),label=j)
plt.title("Logistic Regression(Test Data)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()
