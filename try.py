# from classification import model
import model
import get_mnist
import os
import time
from sklearn import metrics
import numpy as np


if __name__ == "__main__":
    data_path = './dataset/mnist.pkl.gz'
    pqzh_data = get_mnist.read_mnist_data(data_path)
    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
    classifiers = {'NB': model.naive_bayes_classifier,
                   'KNN': model.knn_classifier,
                   'LR': model.logistic_regression_classifier,
                   'RF': model.random_forest_classifier,
                   'DT': model.decision_tree_classifier,
                   'SVM': model.svm_classifier,
                   'SVMCV': model.svm_cross_validation,
                   'GBDT': model.gradient_boosting_classifier
                   }

    for classifier in test_classifiers:
        print '************%s************' % classifier
        start_time = time.time()

        model = classifiers[classifier](pqzh_data['train_data'], pqzh_data['train_label'])
        print 'training took %fs!' % (time.time() - start_time)

        predict = model.predict(pqzh_data['valid_data'])
        accuracy = metrics.accuracy_score(pqzh_data['valid_label'], predict)
        print 'accuracy: %.2f%%' % (100 * accuracy)




