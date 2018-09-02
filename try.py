# from classification import model
from classification import model
from dataset import get_mnist
from metric import metric
import os
import time
import numpy as np


if __name__ == "__main__":
    data_path = './dataset/mnist.pkl.gz'
    pqzh_data = get_mnist.read_mnist_data(data_path)
    # test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    test_classifiers = []
    classifiers = {'NB': model.naive_bayes_classifier,
                   'KNN': model.knn_classifier,
                   'LR': model.logistic_regression_classifier,
                   'RF': model.random_forest_classifier,
                   'DT': model.decision_tree_classifier,
                   'SVM': model.svm_classifier,
                   'SVMCV': model.svm_cross_validation,
                   'GBDT': model.gradient_boosting_classifier,
                   'GNB': model.gaussian_naive_bayes_classifier
                   }

    regressions = {'LR': model.linear_regression,
                   'GBR': model.gradient_boosting_regression,
                   'DTR': model.decesion_tree_regression,
                   'SVR': model.svm_regression,
                   'BRR': model.bayesian_ridge_regression
                   }
    test_regressions = ['LR']
    for classifier in test_classifiers:
        print '************%s************' % classifier
        start_time = time.time()

        model = classifiers[classifier](
            pqzh_data['train_data'], pqzh_data['train_label'])
        print 'training took %fs!' % (time.time() - start_time)

        predict = model.predict(pqzh_data['valid_data'])
        accuracy = metric.classification_acc(pqzh_data['valid_label'], predict)
        print 'accuracy: %.2f%%' % (100 * accuracy)

    for regresion in test_regressions:
        print '************%s************' % regresion
        start_time = time.time()

        model = regressions[regresion](
            pqzh_data['train_data'], pqzh_data['train_label'])
        print 'training took %fs!' % (time.time() - start_time)
        predict = model.predict(pqzh_data['valid_data'])
        l2 = metric.regression_l2(pqzh_data['valid_label'], predict)
        print 'Mean squared error regression loss: %.2f' % (l2)

        print ''
