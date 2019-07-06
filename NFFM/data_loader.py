# encoding = utf8
import os
import pandas as pd 
import numpy as numpy 
from sklearn.preprocessing import LabelEncoder

def data_load(file_path,nrows=None):
    if not os.path.exists(file_path):
        return
    data = pd.read_csv(file_path,nrows=nrows)
    features = ['userId', 'movieId', 'tag']
    feature_field = {
        'userId': 0,
        'movieId': 1,
        'tag': 2
    }
    feature_size = {}
    for s in features:
        le = LabelEncoder()
        data[s] = data[s].apply(str)
        data[s] = le.fit_transform(data[s].values)
        feature_size[s] = len(le.classes_)
    return feature_size, feature_field, data 
if __name__ == '__main__':

    feature_size, feature_field, df = data_load('../data/data.csv')
