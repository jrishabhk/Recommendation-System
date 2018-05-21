#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 17:00:09 2018

@author: J Rishabh Kumar
@content: class implementation for loading dataset into appropriate format

"""


from collections import defaultdict
import sys
import os
import itertools
import random
import warnings

from six.moves import input
from six.moves import range

from reader import Reader
from trainset import Trainset


class Dataset:
    def __init__(self, reader):

        self.reader = reader

   
    @classmethod
    def load_from_file(cls, file_path, reader):
       
        return DatasetAutoFolds(ratings_file=file_path, reader=reader)

    @classmethod
    def load_from_df(cls, df, reader):
       
        return DatasetAutoFolds(reader=reader, df=df)

    def read_ratings(self, file_name):
        """Return a list of ratings (user, item, rating) read from file_name"""

        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_ratings

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

    def __init__(self, ratings_file=None, reader=None, df=None):

        Dataset.__init__(self, reader)
        self.has_been_split = False  # flag indicating if split() was called.

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)
        elif df is not None:
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

