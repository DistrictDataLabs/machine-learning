# algviz
# Generate visualizations of classification, regression, and clustering
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri Feb 27 13:07:28 2015 -0500
#
# Copyright (C) 2015 District Data Labs
# For license information, see LICENSE.txt
#
# ID: algviz.py [] benjamin@bengfort.com $

"""
Generate visualizations of classification, regression, and clustering
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.lda import LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.cluster import KMeans

rgb_colors = ['#FF0000', '#00FF00', '#0000FF']
cm_bright  = ListedColormap(rgb_colors)

def visualize_classification(estimator, n_samples=100, n_features=2):
    # Create the linear dataset and estimator
    kwargs = {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': 3,
        'n_redundant': 0,
        'n_clusters_per_class': 1,
        'class_sep': 1.22,
    }
    X, y = datasets.make_classification(**kwargs)

    # Create the figure
    fix, axes = plt.subplots()

    # no ticks
    axes.set_xticks(())
    axes.set_yticks(())
    axes.set_ylabel('$x_1$')
    axes.set_xlabel('$x_0$')

    # Plot the surface
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cm_bright, alpha=0.3)

    # Plot the points on the grid
    axes.scatter(X[:,0], X[:,1], c=y, s=30, cmap=cm_bright)

    # Show the plot
    plt.axis("tight")
    plt.show()

def visualize_regression(n_samples=100):
    def f(x):
        return np.sin(2 * np.pi * x)

    # Generate data
    X = np.random.uniform(0, 1, size=n_samples)[:,np.newaxis]
    y = f(X) + np.random.normal(scale=0.3, size=n_samples)[:,np.newaxis]

    # Create the linespace
    x_plot = np.linspace(0, 1, 100)[:,np.newaxis]

    poly = PolynomialFeatures(degree=6)
    lreg = LinearRegression()

    pipeline = Pipeline([("polynomial_features", poly),
                             ("linear_regression", lreg)])
    pipeline.fit(X, y)

    # Create the figure
    fix, axes = plt.subplots()

    # no ticks
    axes.set_xticks(())
    axes.set_yticks(())
    axes.set_ylabel('$y$')
    axes.set_xlabel('$x$')

    # Plot the estimator and the true line
    axes.plot(x_plot, pipeline.predict(x_plot), color='red', label="estimated")
    axes.plot(x_plot, f(x_plot), color='green', label='true function')

    # Plot the points
    axes.scatter(X, y)

    plt.legend(loc="best")
    plt.show()

def visualize_clustering(n_samples=350, n_centers=3, n_features=2):
    # Create the data
    X,y = datasets.make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features)

    # Create the estimator
    estimator = KMeans(n_clusters=n_centers, n_init=10)
    estimator.fit(X)

    centroids = estimator.cluster_centers_

    # Create the figure
    fig, axes = plt.subplots()

    # Plot the clusters
    for k, col in zip(xrange(n_centers), rgb_colors):
        m = estimator.labels_ == k
        center = centroids[k]
        # axes.plot(X[m,0], X[m, 1], 'w', markerfacecolor=col, marker='.', markersize=10)
        axes.plot(center[0], center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=200, alpha=.15)

    # no ticks
    axes.set_xticks(())
    axes.set_yticks(())
    axes.set_ylabel('$x_1$')
    axes.set_xlabel('$x_0$')

    # Plot the points
    axes.scatter(X[:,0], X[:,1], c='k')

    plt.show()

if __name__ == '__main__':
    # visualize_classification(KNeighborsClassifier(n_neighbors=3))
    # visualize_regression()
    visualize_clustering()
