# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:55:35 2019

@author: Bob
"""

import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

boston = pd.read_csv('boston.csv')

"""First we'll describe the basic stats related to the dataset"""
print(boston.describe())

"""And here is a basic distribution plot of the target column.  Clearly, the
data is mostly a normal distribution, with some skew on the right side."""

sbn.distplot(boston['MEDV'], bins=40)
plt.show()

"""Now lets take a look at each of the columns against the target column"""
columns = list(boston.columns)[:-1]
for column in columns:
    sbn.scatterplot(
        x=boston[column],
        y=boston.MEDV)
    plt.show()
    
"""To see which of these columns have relate to the house price most, lets
take a look at the heat map"""

correlation_matrix = boston.corr(method="pearson")
sbn.heatmap(
    data=correlation_matrix,
    annot=True,
    linewidths=.5,)
plt.show()

"""From this, we can see that the best single predictors of housing costs
are the number of rooms (RM) and the inverse of the % lower status of the
population.  The other metrics are not as usefull.  It makes sense that
these metrics are the most important: houses in low-income neighborhoods
will not be able to sell for as much, and larger houses will cost more.

Some examples of colinearlity are, again, LSTAT and RM, as well as DIS -
the distance to a major population center, AGE - the proportion of houses
built before 1949-, and INDUS - proportion of non-retail business acres per
town.  Other heatmap methods are below."""

print('Searman')
correlation_matrix = boston.corr(method="spearman")
sbn.heatmap(
    data=correlation_matrix,
    annot=True,
    linewidths=.5,)
plt.show()

print('Kendall')
correlation_matrix = boston.corr(method="kendall")
sbn.heatmap(
    data=correlation_matrix,
    annot=True,
    linewidths=.5,)
plt.show()

"""Box plots and density plotsshow the spread of the data"""

sbn.set(color_codes=True)
colors = ['y', 'b', 'g', 'r']

for column in columns:
    sbn.boxplot(
        boston[column],
        color=np.random.choice(colors))
    plt.show()  
    
    sbn.distplot(
        boston[column],
        color=np.random.choice(colors))
    plt.show()
    
