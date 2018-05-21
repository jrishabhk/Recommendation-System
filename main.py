#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 23:54:37 2018

@author: J Rishabh Kumar
@content: Main
"""

from mat_fact import *
from dataset import *
from reader import *
from split import *
from utility import *
from cross_validate import *
import pandas as pd


reader = Reader()
df = pd.read_csv('Dataset/user_i.csv')
data = Dataset.load_from_df(df,reader)


trainset, testset = train_test_split(data)

algo = NMF()
algo.fit(trainset)
prediction = algo.test(testset)
rmse_ = rmse(prediction)
mae_ = mae(prediction)

algo2 = SVD()
algo2.fit(trainset)
prediction2 = algo2.test(testset)
rmse_ = rmse(prediction2)
mae_ = mae(prediction2)


cross_validate(algo, data)


"""
below code for 
practice

"""

line_format='user item rating'
splitted_format = line_format.split()
entities = ['user', 'item', 'rating']
indexes = [splitted_format.index(entity) for entity in 
           entities]

with open(os.path.expanduser('Dataset/user_i.csv')) as f:
    raw_ratings = [ reader.parse_line(line) for line in 
                   itertools.islice(f, reader.skip_lines, None)]


line = raw_ratings[1]
line = line.split(',')
uid, iid, r = (line[i+1].strip() for i in indexes)



raw_ratings2 = [(uid, iid, float(r)) for (_, uid, iid, r) in
                df.itertuples(index=False)]

shape = (trainset.n_users, trainset.n_items)
pu = np.ndarray(shape = (shape[0], algo.n_factors))
qi = np.ndarray(shape = (self.shape[1],self.n_factors)) 

