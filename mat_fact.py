#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:00:48 2018

@author: J Rishabh Kumar
@content: NMF matrix factorization

"""
import pandas as pd
import numpy as np
from six.moves import range
import numbers
from utility import *
from cross_validate import *

class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """

    pass


class SVD():
    

    def __init__(self, n_factors=20, n_iter=50, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.001, reg_all= 0.001, 
                 reg_pu=.06,reg_qi=.06, reg_bu=.02, reg_bi=.02, lr_bu=.005, lr_bi=.005,
                 lr_pu = None, lr_qi = None, random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_iter = n_iter
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu 
        self.lr_bi = lr_bi 
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu 
        self.reg_bi = reg_bi 
        self.reg_pu = reg_pu 
        self.reg_qi = reg_qi 
        self.random_state = random_state
        self.verbose = verbose


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

        global_mean = self.trainset.global_mean

        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        lr_pu = self.lr_pu
        lr_qi = self.lr_qi

        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        reg_pu = self.reg_pu
        reg_qi = self.reg_qi

        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, float)
        bi = np.zeros(trainset.n_items, float)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_iter):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = 0.0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        u = int(u)
        i = int(i)
        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unkown.')

        return est
    
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
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)

        # Remap the rating into its initial rating scale (because the rating
        # scale was translated so that ratings are all >= 1)
        #est -= self.trainset.offset

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred
    

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
        
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        u = int(u)
        i = int(i)
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
            else:
                raise PredictionImpossible('User and item are unkown.')
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

    
