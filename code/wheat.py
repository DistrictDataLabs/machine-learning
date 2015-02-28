# wheat
# Classification and Clustering of Wheat Dataset
#
# Author:   Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Feb 26 17:56:52 2015 -0500
#
# Copyright (C) 2015 District Data Labs
# For license information, see LICENSE.txt
#
# ID: wheat.py [] benjamin@bengfort.com $

"""
Classification and Clustering of Wheat Dataset
"""

##########################################################################
## Imports
##########################################################################

import time
import pickle

from utils import *
from sklearn import metrics
from sklearn import cross_validation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

##########################################################################
## Wheat Kernel Classification
##########################################################################

if __name__ == '__main__':

    start_time = time.time()

    # Load the dataset
    dataset    = load_wheat()
    data       = dataset.data
    target     = dataset.target

    # Get training and testing splits
    splits     = cross_validation.train_test_split(data, target, test_size=0.2)
    data_train, data_test, target_train, target_test = splits

    load_time  = time.time()

    # Fit the training data to the model
    model      = KNeighborsClassifier()
    # model      = SVC()
    model.fit(data_train, target_train)

    build_time = time.time()

    print model

    # Make predictions
    expected   = target_test
    predicted  = model.predict(data_test)

    # Evaluate the predictions
    print metrics.classification_report(expected, predicted, target_names=dataset.target_names.values())
    print metrics.confusion_matrix(expected, predicted)

    eval_time  = time.time()

    print "Times: %0.3f sec loading, %0.3f sec building, %0.3f sec evaluation" % \
          (load_time-start_time, build_time-load_time, eval_time-build_time,)
    print "Total time: %0.3f seconds" % (eval_time-start_time)

    # Save the model to disk
    with open('model.pickle', 'w') as f:
        pickle.dump(model, f)
