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
import time
import numpy as np

## Pandas is not a required dependency
try:
    import pandas as pd
except ImportError:
    pd =  None

##########################################################################
## Module Constants
##########################################################################

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
CODE_DIR = os.path.join(BASE_DIR, "code")

##########################################################################
## Helper Functions
##########################################################################

def load_data(path):
    """
    Returns a Numpy array, loaded from the data on disk
    """
    path = os.path.join(DATA_DIR, path)
    return np.loadtxt(path)

def load_data_frame(path):
    """
    Returns a Pandas DataFrame, loaded from data on disk
    """
    if pd is None:
        raise ImportError("pandas could not be imported")

    path = os.path.join(DATA_DIR, path)
    return pd.read_csv(path)

def timeit(func):
    """
    Returns how long a function took to execute, along with the output
    """

    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start

    return timeit
