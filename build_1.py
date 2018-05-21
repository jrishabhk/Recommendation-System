#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:01:37 2018

@author: j
"""

import pandas as pd
import numpy as np
import als 
import mf_sgd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import surprise

data = pd.read_csv('Dataset/refine_data4.csv', sep =',')
rating_data = pd.read_csv('Dataset/rating_mat_212.csv')
rating_data = rating_data.fillna(0)
rating_data = np.matrix( rating_data.iloc[:,1:])

#x,y = rating_data.nonzero()
pred = np.matrix(np)

matric = mf_sgd.MF(rating_data, 20, alpha = 0.01,  beta = 0.01, iterations = 50)
training_process = matric.train()
matric.plot_RMSE()



