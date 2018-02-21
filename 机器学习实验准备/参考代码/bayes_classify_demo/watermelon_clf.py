# coding=utf-8
import csv
import numpy as np
from math import sqrt

attr_num = [3, 3, 3, 3, 3, 2]


def loadCsv(filename):
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(1, len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    result = np.array(dataset[1:])
    return result[:, 1:]


def pre_problity(datasets):
    pos_prob = 1.0 * (np.sum(datasets[:, -1] == 1.0) + 1) / (np.shape(datasets)[0] + 2)
    neg_prob = 1.0 * (np.sum(datasets[:, -1] == 0.0) + 1) / (np.shape(datasets)[0] + 2)
    return [pos_prob, neg_prob]


def cond_attr_problity(datasets, testdata):
    cond_result = np.zeros([np.shape(datasets)[1] - 1, 2])
    pos_data = datasets[datasets[:, -1] == 1.0, :]
    neg_data = datasets[datasets[:, -1] == 0.0, :]
    for i in range(len(attr_num)):
        cond_result[i, 0] = 1.0 * (np.sum(pos_data[:, i] == testdata[0, i]) + 1) / (
        np.sum(datasets[:, -1] == 1.0) + attr_num[i])
        cond_result[i, 1] = 1.0 * (np.sum(neg_data[:, i] == testdata[0, i]) + 1) / (
        np.sum(datasets[:, -1] == 0.0) + attr_num[i])

    for j in range(6, 8):
        #         mean,std computation
        pos_mean = np.mean(datasets[(datasets[:, -1] == 1.0), j])
        pos_std = np.std(datasets[(datasets[:, -1] == 1.0), j])
        neg_mean = np.mean(datasets[(datasets[:, -1] == 0.0), j])
        neg_std = np.std(datasets[(datasets[:, -1] == 0.0), j])
        cond_result[j, 0] = 1.0 / (sqrt(2 * np.pi) * pos_std) * np.exp(
            -1 * (testdata[0, j] - pos_mean) ** 2 / (2 * pos_std ** 2))
        cond_result[j, 1] = 1.0 / (sqrt(2 * np.pi) * neg_std) * np.exp(
            -1 * (testdata[0, j] - neg_mean) ** 2 / (2 * neg_std ** 2))
    return cond_result


def classify_data(cond_result, pre_result):
    pos_result = pre_result[0]
    neg_result = pre_result[1]
    for i in range(np.shape(cond_result)[0]):
        pos_result *= cond_result[i, 0]
        neg_result *= cond_result[i, 1]
    if pos_result > neg_result:
        print('好瓜')
        print(pos_result)
    else:
        print('坏瓜')
        print(neg_result)


def main():
    filename = 'watermelon3_0_En.csv'
    dataset = loadCsv(filename)
    testname = 'test.csv'
    testdata = loadCsv(testname)
    pre_result = pre_problity(dataset)
    cond_result = cond_attr_problity(dataset, testdata)
    classify_data(cond_result, pre_result)


main()