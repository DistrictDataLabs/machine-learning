# utils
# Utility functions for handling data
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Feb 26 17:47:35 2015 -0500
#
# Copyright (C) 2015 District Data Labs
# For license information, see LICENSE.txt
#
# ID: utils.py [] benjamin@bengfort.com $

"""
Utility functions for handling data
"""

##########################################################################
## Imports
##########################################################################

import os
import csv
import time
import json
import numpy as np

from sklearn.datasets.base import Bunch

##########################################################################
## Module Constants
##########################################################################

SKL_DATA = "SCIKIT_LEARN_DATA"
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CODE_DIR = os.path.join(BASE_DIR, "code")

##########################################################################
## Helper Functions
##########################################################################

def timeit(func):
    """
    Returns how long a function took to execute, along with the output
    """

    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start

    return timeit

##########################################################################
## Dataset Loading
##########################################################################

def get_data_home(data_home=None):
    """
    Returns the path of the data directory
    """
    if data_home is None:
        data_home = os.environ.get(SKL_DATA, DATA_DIR)

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home


def load_data(path, descr=None, target_index=-1):
    """
    Returns a scklearn dataset Bunch which includes several important
    attributes that are used in modeling:

        data: array of shape n_samples * n_features
        target: array of length n_samples
        feature_names: names of the features
        target_names: names of the targets
        filenames: names of the files that were loaded
        DESCR: contents of the readme

    This data therefore has the look and feel of the toy datasets.

    Pass in a path usually just the name of the location in the data dir.
    It will be joined with the result of `get_data_home`. The contents are:

        path
            - abalone.names     # The file to load into DESCR
            - meta.json     # A file containing metadata to load
            - dataset.txt   # The numpy loadtxt file
            - dataset.csv   # The pandas read_csv file

    You can specify another descr, another feature_names, and whether or
    not the dataset has a header row. You can also specify the index of the
    target, which by default is the last item in the row (-1)
    """

    root          = os.path.join(get_data_home(), path)
    filenames     = {
        'meta': os.path.join(root, 'meta.json'),
        'rdme': os.path.join(root, 'abalone.names'),
        'data': os.path.join(root, 'dataset.csv'),
    }

    target_names  = None
    feature_names = None
    DESCR         = None

    with open(filenames['meta'], 'r') as f:
        meta = json.load(f)
        target_names  = meta['target_names']
        feature_names = meta['feature_names']

    with open(filenames['rdme'], 'r') as f:
        DESCR = f.read()

    # skip header from csv, load data
    dataset = np.loadtxt(filenames['data'], delimiter=',', skiprows=1)
    data    = None
    target  = None
    
    # Target assumed to be either last or first row
    if target_index == -1:
        data   = dataset[:,0:-1]
        target = dataset[:,-1]
    elif target_index == 0:
        data   = dataset[:,1:]
        target = dataset[:,0]
    else:
        raise ValueError("Target index must be either -1 or 0")

    return Bunch(data=data,
                 target=target,
                 filenames=filenames,
                 target_names=target_names,
                 feature_names=feature_names,
                 DESCR=DESCR)

def load_abalone():
    return load_data('abalone')
