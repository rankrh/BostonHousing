# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:53:30 2019

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

x = boston.drop('MEDV', axis=1)
y = boston.MEDV

x.LSTAT = np.log(x.LSTAT)
a, b, c  = np.polyfit(x.LSTAT, y, 2)
x.LSTAT = a * x.LSTAT**2 + b * x.LSTAT + c


x_train, x_valid, y_train, y_valid = train_test_split(
    x,
    y,
    test_size = 0.3,
    random_state=108)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

y_pred = lin_reg.predict(x_valid)

plot = sbn.scatterplot(
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











