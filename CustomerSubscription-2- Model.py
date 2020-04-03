# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:48:00 2020

@author: vedav
"""

#### Importing Libraries ####

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

#Read the new dataset that we created
dataset = pd.read_csv('new_appdata10.csv')


#### Data Pre-Processing ####

# Splitting Independent and Response Variables
response = dataset["enrolled"]
# The following will be the independent variable
dataset = dataset.drop(columns="enrolled")

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response,
                                                    test_size = 0.2,
                                                    random_state = 0)

## Balancing the Training Set
#import random
#y_train.value_counts()
#
#pos_index = y_train[y_train.values == 1].index
#neg_index = y_train[y_train.values == 0].index
#
#if len(pos_index) > len(neg_index):
#    higher = pos_index
#    lower = neg_index
#else:
#    higher = neg_index
#    lower = pos_index
#
#random.seed(0)
#higher = np.random.choice(higher, size=len(lower))
#lower = np.asarray(lower)
#new_indexes = np.concatenate((lower, higher))
#
#X_train = X_train.loc[new_indexes,]
#y_train = y_train[new_indexes]


# Removing Identifiers
# We do not need user id for training the model
# But we do need it in the end to map the end results to the userid
#So, let us save user id details of training data in a seperate column train_identity
train_identity = X_train['user']
X_train = X_train.drop(columns = ['user'])
#So, let us save user id details of test data in a seperate column test_identity
test_identity = X_test['user']
X_test = X_test.drop(columns = ['user'])

# Feature Scaling
# Standard Scaler returns a numpy array
# It loses the column names and the index
# We care about the index because we identify a field using the index 
# We care about the columns as well as we build the models using them
# So we save the scaled results in a seperate dataframe

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
#We only transform here as it is already fitted
X_test2 = pd.DataFrame(sc_X.transform(X_test))

#We should make sure the standardiset set has the columns as well
# We are taking the column names from original X_train and X_test 
# and assigning the same names to the standardised set created
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values


#We are taking the row numbers.i.e from the original datset
#Assigning them to the newly created dataset
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values

#Change the names of teh new dataset to X_train and X_test just for understanding
X_train = X_train2
X_test = X_test2


#### Model Building ####


# Fitting Model to the Training Set
# adding penalty will change the model from regular logistic regression model to
# L1 regularization model
# Because of its nature, screens can be correlated with each other
# Maybe the finance screen is next to the loans screen
# Therefore if you click on the Finance screen you also click on the loan's screen
# We addressed lot of these features by creating funnels
# Even though the screens are not belonging to the same set there might be some
#correlation existing 
#If you are working for a Mobile app  model in which you are using mobile screens
#always take into consideration that screens can be correlated
# What L1 does is, it penalises any particular field that is extremely correlated
# So, if one screen is extremely correlated to the response variable meaning 
# there is just one screen that always happens before your enrollment
# And as a result it is always strongly correlated to the response variable to one another
# That screen may get higher coefficient in the logistic regression model 
#SO, the L1 regularization penalizes those cases in which one particular feature
#gets very big coefficient
#This is important in models in which you are working with mobile app screens
#l1 parameter is throwing error so, used l2
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, penalty = 'l2')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)

# Evaluating Results
#76 accuracy is very good model
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred)

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))


# Applying k-Fold Cross Validation
# Done just to ensure that there is no overfit in the model
# WE apply the model to different subsets of the training set
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Logistic Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

# Analyzing Coefficients
pd.concat([pd.DataFrame(dataset.drop(columns = 'user').columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)


#### Model Tuning ####

## Grid Search (Round 1)
from sklearn.model_selection import GridSearchCV

# Select Regularization Method
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Combine Parameters
parameters = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


## Grid Search (Round 2)

# Select Regularization Method
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = [0.1, 0.5, 0.9, 1, 2, 5]

# Combine Parameters
parameters = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters
grid_search.best_score_


#### End of Model ####


# Formatting Final Results
final_results = pd.concat([y_test, test_identity], axis = 1).dropna()
final_results['predicted_reach'] = y_pred
final_results = final_results[['user', 'enrolled', 'predicted_reach']].reset_index(drop=True)

final_results


