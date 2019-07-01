# encoding = utf8
import os
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from scipy import sparse

features = ['userId', 'movieId', 'tag']
file_path = '../data/data.csv'


def data_load(file_path,nrows=None):
    if not os.path.exists(file_path):
        return
    data = pd.read_csv(file_path, nrows=nrows)
    for s in features:
        le = LabelEncoder()
        data[s] = data[s].apply(str)
        data[s] = le.fit_transform(data[s])
    labels = data.pop('label')
    return data, labels


def main():
    data, labels = data_load(file_path=file_path, nrows=None)
    num = data.shape[0] * 4 // 5
    train_x, valid_x, train_y, valid_y = data[:num], data[num:], labels[:num], labels[num:]
    model = lgb.LGBMClassifier(num_leaves=48, n_estimators=2000, categorical_feature=[0,1,2])

    model.fit(
        train_x, train_y,
        eval_set = [(valid_x, valid_y)],
        eval_metric = 'auc',
        early_stopping_rounds = 200,
        categorical_feature = [0,1,2],
        verbose = True
    )



if __name__ == '__main__':

    main()
