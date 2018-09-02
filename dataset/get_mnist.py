# -*-coding:utf-8-*-
import pickle
import gzip
import os


def pack_data(training_data, training_label, valid_data, valid_label):
    # 给数据打包成标准格式 字典
    # 字典key为train_data，train_label，valid_data，valid_label
    qzph_data = {}
    qzph_data['train_data'] = training_data
    qzph_data['train_label'] = training_label
    qzph_data['valid_data'] = valid_data
    qzph_data['valid_label'] = valid_label
    return qzph_data


def load_data(mnist_path):
    with gzip.open(mnist_path) as fp:
        training_data, valid_data, test_data = pickle.load(fp)
    return training_data, valid_data, test_data


def read_mnist_data(mnist_path='./mnist.pkl.gz'):
    # 读取mnist数据集，使用read_mnist_data(mnist_path)
    # mnist_path 为存放mnist.plk.gz的路径，包含文件名
    if not os.path.isfile(mnist_path):
        print "path_error!!!!!!%s can not find file" % (mnist_path)
        return None
    training_data, valid_data, test_data = load_data(mnist_path)
    qzph_data = pack_data(
        training_data[0], training_data[1], valid_data[0], valid_data[1])
    return qzph_data


# qzph_data = read_mnist_data()
