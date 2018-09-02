import os
import time
from sklearn import metrics
import numpy as np
import cPickle as pickle


def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model


def gaussian_naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(train_x, train_y)
    return model

# KNN Classifier


def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


def decesion_tree_classifer(train_x, train_y):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model

# Random Forest Classifier


def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


def sgd_classifier(train_x, train_y):
    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier()
    model.fit(train_x, train_y)
    return model

# Decision Tree Classifier


def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

# SVM Classifier using cross validation


def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10,
                        100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = SVC(kernel='rbf', C=best_parameters['C'],
                gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

#  Regression*********************************************************


def linear_regression(train_x, train_y):
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    model.fit(train_x, train_y)
    return model


def bayesian_ridge_regression(train_x, train_y):
    from sklearn import linear_model
    model = linear_model.BayesianRidge()
    model.fit(train_x, train_y)
    return model


def svm_regression(train_x, train_y):
    from sklearn import svm
    model = svm.SVR()
    model.fit(train_x, train_y)
    return model


def decesion_tree_regression(train_x, train_y):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(train_x, train_y)
    return model

def gradient_boosting_regression(train_x, train_y):
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor()
    model.fit(train_x, train_y)
    return model
