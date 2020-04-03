# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 17:55:19 2020

@author: Vedav
"""
#### Importing Libraries ####

import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('appdata10.csv')


#### EDA ####


dataset.head(10) # Viewing the Data
dataset.describe() # Distribution of Numerical Variables
#we do not find hour in the above


# First set of Feature cleaning
# convert hour into string and then slice(consider only 1 & 2 numbers only)
#then finally convert the above into integer
#Now hour column will be a numerical column of integer type
#We include this in original data
dataset["hour"] = dataset.hour.str.slice(1, 3).astype(int)


### Plotting
### We create a temporary dataset and include only those
### columns that we need for plotting
dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date',
                                           'first_open', 'enrolled'])
dataset2.head()

## Histograms
## We plot histograms to view the distributions of data
## dataset2.shape[1] gives the number of columns
## As we have 7 columns 3*3 would be good
## gca command cleans up everything
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
#    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])
#Number of unique values in a column will be the number of bins
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig('app_data_hist.jpg')

## Correlation of fields with Response Variable
## Before building the model, it is important to know
## how each and independent variable affects the response variable
## corrwith is pandas function that will return list
## of correlations between all the fields in dataframe
## Here we are calculating the correlation of the response
## variable enrolled
## premium feature is neagtively correlated which means
## the early the hour, the more likely you are to enroll
## age is neagatively correlated
## the older the preson, the less likely to enroll
## So, younger people seem to be more likely to enroll
## num_screens is positively correlated
## means more scrrens you see more engaged with the product
## means you are more likely to buy the product
##mini game positively correlated
## if you are likely to play the minigame you are more engaged
## you are more likely to enroll
##Interesting revealation: premium feature is negatively correlated
## the more likely you use the premium feature
## the less likely you are to enroll
## these revealations are just conclusion for informative purpose
## Nothing is confirmed yet
dataset2.corrwith(dataset.enrolled).plot.bar(figsize=(20,10),
                  title = 'Correlation with Reposnse variable',
                  fontsize = 15, rot = 45,
                  grid = True)


## Correlation Matrix
## This is correlation between each individual fields
## and not between fields and response variables
## this will inform us about which field may be linearly depending
## on another. which field may be a linear combination of another
## meaning a function of one or the other fields
## this will help us in model building process
## because we do not want any field depending on another
## the assumption in building Machine Learning models is
## the features are independent variables
sn.set(style="white", font_scale=2)

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


 
#### Feature Engineering ####
## Main part is finetuning the response variable
## Often, we need to set a limit when a user will convert to paid member
## To understand what timelimit is optimal we are plotting distribution of hour differences between
## the first_open and enrollment_date


# Formatting Date Columns
dataset.dtypes
#first_open and enrolled_date are objects now
#We are converting the first_open to date time object
dataset["first_open"] = [parser.parse(row_date) for row_date in dataset["first_open"]]
#We are converting the enrolled_date to date time object
# we want to apply on those which are strings and not all rows
dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]
#first_open and enrolled_date are date time objects now
dataset.dtypes

# Selecting Time For Response
# We plot this to select best cut-off time for response variable
# The plot says majority enrolled within first 500 hours
# So 500 will be our cut-off point
# This is a right tail distribution here 
# This can happen in first 100 hours also, we do not know
dataset["difference"] = (dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]')
response_hist = plt.hist(dataset["difference"].dropna(), color='#3F5D7D')
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

#So, now we plot histogram with a range here
#again, the distribution is right tailed
# which means everything happens between the first 10 hours
# BUt we still see some enrollments happening upto 40 hours
#To include the majority of people we are going to restrict ourselves to first 50 hours
#To be more specific we can restrict to 48 hours as it is equal to 2 days
plt.hist(dataset["difference"].dropna(), color='#3F5D7D', range = [0, 100])
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

#We will set the enrolled column to zero where the difference is 48 hours
# May be the person who is not enrolled will be zero already

dataset.loc[dataset.difference > 48, 'enrolled'] = 0

#We drop the columns that we no longer need
dataset = dataset.drop(columns=['enrolled_date', 'difference', 'first_open'])

## Formatting the screen_list Field
#We are not going to consider all the screens
# Instead we received the top screens for the data anlyst
# Load Top Screens
top_screens = pd.read_csv('top_screens.csv').top_screens.values
top_screens
 
# Mapping Screens to Fields.
#We are going to create columns only for the popular screens
dataset["screen_list"] = dataset.screen_list.astype(str) + ','



for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc+",", "")


#We will create another column for all those screens that are not popular
# This column indicates how many left over screens do we have
    # We will count how many commas we have
dataset['Other'] = dataset.screen_list.str.count(",")
#Now that we have completed Feature Engineering, let us drop the columns that are not required
dataset = dataset.drop(columns=['screen_list'])

# Funnels
# Funnels are grouping screens that belong to a same set
# There are many screens that are correlating to each other
# In order to get rid of that correlation, we are going to group such screens into one funnel
# If they belong to one Funnel, A becomes one column of how many screens it contains, removes 
# the correlation and we still keep the data

#Create list of all the screens that belong to a funnel
savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]

#Create a column for the savings_screens funnel
#It counts all the columns that have these names and sums it all up
dataset["SavingCount"] = dataset[savings_screens].sum(axis=1)
#Remove those columns now
dataset = dataset.drop(columns=savings_screens)

#We will perform the above for multiple funnels

cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)



cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)


loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)



#### Saving Results ####
### How does the dataset look after creating funnels ###
dataset.head()
dataset.describe()
dataset.columns
dataset.shape

#Now we will save the dataset to a csv with the columns created and changes made until now
dataset.to_csv('new_appdata10.csv', index = False)

