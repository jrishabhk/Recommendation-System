#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:11:09 2018

@author: J Rishabh Kumar
@content cross validation module
"""

import time

import numpy as np
from joblib import Parallel
from joblib import delayed
from six import iteritems
from collections import defaultdict, namedtuple

from split import get_cv


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


def cross_validate(algo, data, measures=['rmse', 'mae'], cv=None,
                   return_train_measures=False, n_jobs=1,
                   pre_dispatch='2*n_jobs', verbose=False):

    measures = [m.lower() for m in measures]

    cv = get_cv(cv)

    delayed_list = (delayed(fit_and_score)(algo, trainset, testset, measures,
                                           return_train_measures)
                    for (trainset, testset) in cv.split(data))
    out = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch)(delayed_list)

    (test_measures_dicts,
     train_measures_dicts,
     fit_times,
     test_times) = zip(*out)

    test_measures = dict()
    train_measures = dict()
    ret = dict()
    for m in measures:
        test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
        ret['test_' + m] = test_measures[m]
        if return_train_measures:
            train_measures[m] = np.asarray([d[m] for d in
                                            train_measures_dicts])
            ret['train_' + m] = train_measures[m]

    ret['fit_time'] = fit_times
    ret['test_time'] = test_times

    if verbose:
        print_summary(algo, measures, test_measures, train_measures, fit_times,
                      test_times, cv.n_splits)

    return ret


def fit_and_score(algo, trainset, testset, measures,
                  return_train_measures=False):
    start_fit = time.time()
    algo.fit(trainset)
    fit_time = time.time() - start_fit
    start_test = time.time()
    predictions = algo.test(testset)
    test_time = time.time() - start_test

    if return_train_measures:
        train_predictions = algo.test(trainset.build_testset())

    test_measures = dict()
    train_measures = dict()
    for m in measures:
        f = m.lower()
        test_measures[m] = f(predictions, verbose=0)
        if return_train_measures:
            train_measures[m] = f(train_predictions, verbose=0)

    return test_measures, train_measures, fit_time, test_time


def print_summary(algo, measures, test_measures, train_measures, fit_times,
                  test_times, n_splits):

    print('Evaluating {0} of algorithm {1} on {2} split(s).'.format(
          ', '.join((m.upper() for m in measures)),
          algo.__class__.__name__, n_splits))
    print()

    row_format = '{:<18}' + '{:<8}' * (n_splits + 2)
    s = row_format.format(
        '',
        *['Fold {0}'.format(i + 1) for i in range(n_splits)] + ['Mean'] +
        ['Std'])
    s += '\n'
    s += '\n'.join(row_format.format(
        key.upper() + ' (testset)',
        *['{:1.4f}'.format(v) for v in vals] +
        ['{:1.4f}'.format(np.mean(vals))] +
        ['{:1.4f}'.format(np.std(vals))])
        for (key, vals) in iteritems(test_measures))
    if train_measures:
        s += '\n'
        s += '\n'.join(row_format.format(
            key.upper() + ' (trainset)',
            *['{:1.4f}'.format(v) for v in vals] +
            ['{:1.4f}'.format(np.mean(vals))] +
            ['{:1.4f}'.format(np.std(vals))])
            for (key, vals) in iteritems(train_measures))
    s += '\n'
    s += row_format.format('Fit time',
                           *['{:.2f}'.format(t) for t in fit_times] +
                           ['{:.2f}'.format(np.mean(fit_times))] +
                           ['{:.2f}'.format(np.std(fit_times))])
    s += '\n'
    s += row_format.format('Test time',
                           *['{:.2f}'.format(t) for t in test_times] +
                           ['{:.2f}'.format(np.mean(test_times))] +
                           ['{:.2f}'.format(np.std(test_times))])
    print(s)



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