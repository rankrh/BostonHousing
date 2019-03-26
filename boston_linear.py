# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:47:53 2019

@author: Bob
"""

import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

"""As a first pass, we'll check how well a simple linear regression fits
the housing data.  If this works well, further analysis is not needed."""
boston = pd.read_csv('boston.csv')

boston_x = boston.drop('MEDV', axis=1)
boston_y = boston.MEDV

x_train, x_valid, y_train, y_valid = train_test_split(
    boston_x,
    boston_y,
    test_size=0.3,
    random_state=108)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

y_pred = lin_reg.predict(x_valid)

plot = sbn.regplot(
    x=y_valid,
    y=y_pred,
    )
plot.set(
    xlabel="Prices: $Y_i$",
    ylabel="Predicted prices: $\hat{Y}_i$",
    title="Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

plt.show()

mean_squared_error = sklearn.metrics.mean_squared_error(y_valid, y_pred)
print('Mean Squared Error: ', mean_squared_error)

"""The MSE here is ~10K, which is significantly higher than we would like.
we'll have to try a more refined regression tactic"


