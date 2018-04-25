#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:51:31 2018

@author: marcello
"""

import pickle
def save_obj(obj, file_name ):
    with open('out/'+ file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
def load_obj(name ):
    with open('out/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
file_name = 'RFminSLeaf5'
dic2 = load_obj(file_name )
print(dic2)