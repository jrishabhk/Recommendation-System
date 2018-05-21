#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:17:00 2018

@author: j
"""

import pandas as pd
from collections import defaultdict, namedtuple
import sys
import os
import itertools
import random
import warnings
from itertools import chain
from math import ceil, floor
import numbers

from six import iteritems
from six import string_types

import numpy as np

from utility import get_rng

from six.moves import input
from six.moves import range



class Reader():
   
    def __init__(self, line_format='user item rating', sep=',',
                 rating_scale=(1, 5), skip_lines=0):
       
            self.sep = sep
            self.skip_lines = skip_lines
            self.rating_scale = rating_scale

            lower_bound, higher_bound = rating_scale
            self.offset = -lower_bound + 1 if lower_bound <= 0 else 0

            splitted_format = line_format.split()

            entities = ['user', 'item', 'rating']
            
            # check that all fields are correct
            if any(field not in entities for field in splitted_format):
                raise ValueError('line_format parameter is incorrect.')

            self.indexes = [splitted_format.index(entity) for entity in
                            entities]

    def parse_line(self, line):

        line = line.split(self.sep)
        uid, iid, r = (line[i+1].strip() for i in self.indexes)
        return uid, iid, float(r) + self.offset




class Trainset:

    def __init__(self, ur, ir, n_users, n_items, n_ratings, rating_scale,
                 offset, raw2inner_id_users, raw2inner_id_items):

        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.rating_scale = rating_scale
        self.offset = offset
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._global_mean = None
        self._inner2raw_id_users = None
        self._inner2raw_id_items = None
        

    def knows_user(self, uid):
        
        return uid in self.ur

    def knows_item(self, iid):

        return iid in self.ir
    
    def to_inner_uid(self, ruid):
        
        try:
            return self._raw2inner_id_users[ruid]
        except KeyError:
            raise ValueError('User ' + str(ruid) +
                             ' is not part of the trainset.')
    
    def to_inner_iid(self, riid):
       
        try:
            return self._raw2inner_id_items[riid]
        except KeyError:
            raise ValueError('Item ' + str(riid) +
                             ' is not part of the trainset.')
            
    def to_raw_uid(self, iuid):
        if self._inner2raw_id_users is None:
            self._inner2raw_id_users = {inner: raw for (raw, inner) in
                                        iteritems(self._raw2inner_id_users)}

        try:
            return self._inner2raw_id_users[iuid]
        except KeyError:
            raise ValueError(str(iuid) + ' is not a valid inner id.')

    def to_raw_iid(self, iiid):

        if self._inner2raw_id_items is None:
            self._inner2raw_id_items = {inner: raw for (raw, inner) in
                                        iteritems(self._raw2inner_id_items)}

        try:
            return self._inner2raw_id_items[iiid]
        except KeyError:
            raise ValueError(str(iiid) + ' is not a valid inner id.')


    def all_ratings(self):

        for u, u_ratings in iteritems(self.ur):
            for i, r in u_ratings:
                yield u, i, r

    def all_users(self):
        
        return range(self.n_users)

    def all_items(self):
       
        return range(self.n_items)

    @property
    def global_mean(self):
        """Return the mean of all ratings.

        It's only computed once."""
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in
                                         self.all_ratings()])

        return self._global_mean


class Dataset:
    def __init__(self, reader):

        self.reader = reader

   
    @classmethod
    def load_from_file(cls,file_path, reader):
        df = pd.read_csv(file_path)
        return DatasetAutoFolds( reader=reader, df = df)

    def folds(self):
        

        warnings.warn('Using data.split() or using load_from_folds() '
                      'without using a CV iterator is now deprecated. ',
                      UserWarning)

        for raw_trainset, raw_testset in self.raw_folds():
            trainset = self.construct_trainset(raw_trainset)
            testset = self.construct_testset(raw_testset)
            yield trainset, testset

    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r  in raw_trainset:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(ur,
                            ir,
                            n_users,
                            n_items,
                            n_ratings,
                            self.reader.rating_scale,
                            self.reader.offset,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_trans)
                for (ruid, riid, r_ui_trans) in raw_testset]


class DatasetUserFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are predefined."""

    def __init__(self, folds_files=None, reader=None):

        Dataset.__init__(self, reader)
        self.folds_files = folds_files

        # check that all files actually exist.
        for train_test_files in self.folds_files:
            for f in train_test_files:
                if not os.path.isfile(os.path.expanduser(f)):
                    raise ValueError('File ' + str(f) + ' does not exist.')

    def raw_folds(self):
        for train_file, test_file in self.folds_files:
            raw_train_ratings = self.read_ratings(train_file)
            raw_test_ratings = self.read_ratings(test_file)
            yield raw_train_ratings, raw_test_ratings


class DatasetAutoFolds(Dataset):

    def __init__(self, reader=None, df=None):

        Dataset.__init__(self, reader)
        self.has_been_split = False  # flag indicating if split() was called.

        if df is not None:
            self.df = df
            self.raw_ratings = [(uid, iid, float(r) )
                                for (_, uid, iid, r) in
                                self.df.itertuples(index=False)]
        else:
            raise ValueError('Must specify ratings file or dataframe.')

    def build_full_trainset(self):

        return self.construct_trainset(self.raw_ratings)

    def raw_folds(self):

        if not self.has_been_split:
            self.split()

        def k_folds(seq, n_folds):
            """Inspired from scikit learn KFold method."""

            start, stop = 0, 0
            for fold_i in range(n_folds):
                start = stop
                stop += len(seq) // n_folds
                if fold_i < len(seq) % n_folds:
                    stop += 1
                yield seq[:start] + seq[stop:], seq[start:stop]

        return k_folds(self.raw_ratings, self.n_folds)

    def split(self, n_folds=5, shuffle=True):
       
        if n_folds > len(self.raw_ratings) or n_folds < 2:
            raise ValueError('Incorrect value for n_folds. Must be >=2 and '
                             'less than the number or entries')

        if shuffle:
            random.shuffle(self.raw_ratings)

        self.n_folds = n_folds
        self.has_been_split = True



class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """

    pass

class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'r_ui', 'est', 'details'])):
   
    __slots__ = ()  # for memory saving purpose.

    def __str__(self):
        s = 'user: {uid:<10} '.format(uid=self.uid)
        s += 'item: {iid:<10} '.format(iid=self.iid)
        if self.r_ui is not None:
            s += 'r_ui = {r_ui:1.2f}   '.format(r_ui=self.r_ui)
        else:
            s += 'r_ui = None   '
        s += 'est = {est:1.2f}   '.format(est=self.est)
        s += str(self.details)

        return s


class NMF():

    def __init__(self, n_factors=20, n_iter=200, biased=True, reg_pu=.06,
                 reg_qi=.06, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005,
                 init_low=0, init_high=1, random_state=None):

        self.n_factors = n_factors
        self.n_iter = n_iter
        self.biased = biased
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.init_low = init_low
        self.init_high = init_high
        self.random_state = random_state

        if self.init_low < 0:
            raise ValueError('init_low should be greater than zero')


    def fit(self, trainset):

        self.trainset = trainset
        self.shape = (trainset.n_users, trainset.n_items)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # user and item factors
        pu = np.ndarray((self.shape[0],self.n_factors))
        qi = np.ndarray((self.shape[1],self.n_factors)) 
        # user and item biases
        bu = np.ndarray(self.shape[0], dtype = float)
        bi = np.ndarray(self.shape[1], dtype = float)
        
        # regularization parameters and learning parameters
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi
        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        global_mean = self.trainset.global_mean

        # Randomly initialize user and item factors
        rng = get_rng(self.random_state)
        pu = rng.uniform(self.init_low, self.init_high,
                         size=(trainset.n_users, self.n_factors))
        qi = rng.uniform(self.init_low, self.init_high,
                         size=(trainset.n_items, self.n_factors))

        bu = np.zeros(trainset.n_users, float)
        bi = np.zeros(trainset.n_items, float)

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_iter):
            # (re)initialize nums and denoms to zero
            user_num = np.zeros((trainset.n_users, self.n_factors))
            user_denom = np.zeros((trainset.n_users, self.n_factors))
            item_num = np.zeros((trainset.n_items, self.n_factors))
            item_denom = np.zeros((trainset.n_items, self.n_factors))

            # Compute numerators and denominators for users and items factors
            for u, i, r in trainset.all_ratings():

                # compute current estimation and error
                dot = 0.0  # <q_i, p_u>
               # u = trainset.to_inner_uid(u)
                #i = trainset.to_inner_iid(i)
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                r_ui = global_mean + bu[u] + bi[i] + dot
                err = r - r_ui

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # compute numerators and denominators
                for f in range(self.n_factors):
                    user_num[u, f] += qi[i, f] * r
                    user_denom[u, f] += qi[i, f] * r_ui
                    item_num[i, f] += pu[u, f] * r
                    item_denom[i, f] += pu[u, f] * r_ui

            # Update user factors
            for u in trainset.all_users():
                n_ratings = len(trainset.ur[u])
                for f in range(self.n_factors):
                    user_denom[u, f] += n_ratings * reg_pu * pu[u, f]
                    pu[u, f] *= user_num[u, f] / user_denom[u, f]

            # Update item factors
            for i in trainset.all_items():
                n_ratings = len(trainset.ir[i])
                for f in range(self.n_factors):
                    item_denom[i, f] += n_ratings * reg_qi * qi[i, f]
                    qi[i, f] *= item_num[i, f] / item_denom[i, f]

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):   
        
        known_user = self.trainset.knows_user(self.trainset.to_inner_uid(u))
        known_item = self.trainset.knows_item(self.trainset.to_inner_iid(i))
        print(u,i)
        if self.biased:
            r_ui = self.trainset.global_mean
            if known_user:
                r_ui += self.bu[u]
            if known_item:
                r_ui += self.bi[i]
            if known_user and known_item:
                r_ui += np.dot(self.qi[i], self.pu[u])
        else:
            if known_user and known_item:
                r_ui = np.dot(self.qi[i], self.pu[u])
        return r_ui

    def test(self, testset, verbose=False):
        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions
    
    def default_prediction(self):

        return self.trainset.global_mean
    
    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):

        details = {}
        try:
            est = self.estimate(uid, iid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)

        
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred



class ShuffleSplit():

    def __init__(self, n_splits=5, test_size=.2, train_size=None,
                 random_state=None, shuffle=True):

        if n_splits <= 0:
            raise ValueError('n_splits = {0} should be strictly greater than '
                             '0.'.format(n_splits))
        if test_size is not None and test_size <= 0:
            raise ValueError('test_size={0} should be strictly greater than '
                             '0'.format(test_size))

        if train_size is not None and train_size <= 0:
            raise ValueError('train_size={0} should be strictly greater than '
                             '0'.format(train_size))

        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle

    def validate_train_test_sizes(self, test_size, train_size, n_ratings):

        if test_size is not None and test_size >= n_ratings:
            raise ValueError('test_size={0} should be less than the number of '
                             'ratings {1}'.format(test_size, n_ratings))

        if train_size is not None and train_size >= n_ratings:
            raise ValueError('train_size={0} should be less than the number of'
                             ' ratings {1}'.format(train_size, n_ratings))

        if np.asarray(test_size).dtype.kind == 'f':
            test_size = ceil(test_size * n_ratings)

        if train_size is None:
            train_size = n_ratings - test_size
        elif np.asarray(train_size).dtype.kind == 'f':
            train_size = floor(train_size * n_ratings)

        if test_size is None:
            test_size = n_ratings - train_size

        if train_size + test_size > n_ratings:
            raise ValueError('The sum of train_size and test_size ({0}) '
                             'should be smaller than the number of '
                             'ratings {1}.'.format(train_size + test_size,
                                                   n_ratings))

        return int(train_size), int(test_size)

    def split(self, data):

        train_size, test_size = self.validate_train_test_sizes(
            self.test_size, self.train_size, len(data.raw_ratings))
        rng = get_rng(self.random_state)

        for _ in range(self.n_splits):

            if self.shuffle:
                permutation = rng.permutation(len(data.raw_ratings))
            else:
                permutation = np.arange(len(data.raw_ratings))

            raw_trainset = [data.raw_ratings[i] for i in
                            permutation[:train_size]]
            raw_testset = [data.raw_ratings[i] for i in
                           permutation[train_size:(test_size + train_size)]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits


def train_test_split(data, test_size=.2, train_size=None, random_state=None,
                     shuffle=True):
    
    ss = ShuffleSplit(n_splits=1, test_size=test_size, train_size=train_size,
                      random_state=random_state, shuffle=shuffle)
    return next(ss.split(data))
    

def get_cv(cv):
    '''Return a 'validated' CV iterator.'''

    if cv is None:
        return KFold(n_splits=5)
    if isinstance(cv, numbers.Integral):
        return KFold(n_splits=cv)
    if hasattr(cv, 'split') and not isinstance(cv, string_types):
        return cv  # str have split

    raise ValueError('Wrong CV object. Expecting None, an int or CV iterator, '
                     'got a {}'.format(type(cv)))



def rmse(predictions, verbose=True):
   
    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mae(predictions, verbose=True):
    
    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_


"""
        __main()__
        
"""

reader = Reader()
data = Dataset.load_from_file('Dataset/user_i.csv',reader)


trainset, testset = train_test_split(data)

algo = NMF()
algo.fit(trainset)
prediction = algo.test(testset)
rmse_ = rmse(prediction)
mae_ = mae(prediction)