# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:47:53 2019

@author: Bob
"""

import numpy as np
import pandas as pd

boston = pd.read_csv('boston.csv')
print(boston.head())
print(boston.describe())

boston.plot()