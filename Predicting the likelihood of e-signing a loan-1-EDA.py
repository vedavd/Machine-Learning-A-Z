# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 08:09:42 2020

@author: vedav
"""

#### Importing Libraries ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('financial_data.csv')


### EDA ###

dataset.head()
dataset.columns
dataset.describe()


## Cleaning Data

# Removing NaN
dataset.isna().any() # No NAs


## Histograms
#Remove categorical variable and entry_id 
dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


## Correlation with Response Variable (Note: Models like RF are not linear like these)
# The bigger the bar, the bigger the correlation i.e relationship
#If the bar is bigger on the negative side, as the variable increases,
#the likelihood of response variable being positive decreases
#---Age seems to be very powerfully correlated with the response variable being negative
#--- meaning that as you grow older you are less likely to put up with entire onboarding
#--- process and get to the final stage
#--- home owner is negatively correlated with signing loan
#--- if you are homeowner you are less likely to reach the final step
#--- of the process
#--- personal account month negatively correlated
#--- which is something difficult to understand
#if the bar is bigger on the positive end, when the variable increases,
#the likelihood of response variable being positive increases
#--- income is positively correlated
#--- if the initial_risk score increase, their  relationship with the response variable increases
#--- this is because we have not normalized our relationship yet
#--- none of these features are normalized yet
#--- we will deal with it in the data preprocessing part
#--- Finallu inquiries, the more inquiries you have the less likely you are to reach the
#--- e sign process

dataset2.corrwith(dataset.e_signed).plot.bar(
        figsize = (20, 10), title = "Correlation with E Signed", fontsize = 15,
        rot = 45, grid = True)


## Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#---The more income you make in a monththe more likely you are to request higher
#---amounts of money for your loan application
#---if you are employed for a decade in the same job  that means
#---you are going to live in the same state,city
#---And from such people,a subset of them are going to be living in the 
#---same house or same apartment. SO, it makes sense that a subset of them
#---are very equally correlated with the years employed as well as current year
#---Amount rquested is strongly correlated with first risk score which makes sense
#---because we know that these scores depend on multiple features and each of them
#---are different. So it makes sence that atleast one of them, especially that the first risk 
#---score is dependent on the amount requested
#---The odds are, if the risk score is high,very very high for one, itz going to be
#---somewhat high for the other ones as well
#---
#---