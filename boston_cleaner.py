# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:50:11 2019

@author: Bob
"""

import numpy as np
import pandas as pd

boston = list(open('boston.txt', 'r'))

columns = boston[7:21]
columns = [line.split()[0] for line in columns]

boston_data = []

i = 22
while i < len(boston[23::2]):
    line = boston[i].split() + boston[i+1].split()
    boston_data += [line]
    
    i += 2

boston_data = pd.DataFrame(
    data=boston_data,
    columns=columns,)

print(boston_data)
boston_data.to_csv('boston.csv', index=False)