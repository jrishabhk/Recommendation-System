#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:57:55 2018

@author: J Rishabh Kumar
@content: implementation of reader class
"""


class Reader():
   
    def __init__(self, line_format='user item rating', sep=',',
                 rating_scale=(1, 5), skip_lines=0):
       
            self.sep = sep
            self.skip_lines = skip_lines
            self.rating_scale = rating_scale

            lower_bound, higher_bound = rating_scale

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
        return uid, iid, float(r) 
