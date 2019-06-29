# encoding = utf8
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pprint import pprint
import pickle
from args import args
from sklearn.preprocessing import LabelEncoder


def data_load(file_path, nrows=None):
    if not os.path.exists(file_path):
        return
    data = pd.read_csv(file_path, nrows=nrows)
    for s in args.CATEGORICAL_FEATURES:
        le = LabelEncoder()
        data[s] = data[s].apply(str)
        data[s] = le.fit_transform(data[s].values)
    return data


def parse_one_record(line):
    return line[:3], line[3:]


def read_csv_data(df, num_epochs=1):
    return tf.data.Dataset.from_tensor_slices(df) \
             .map(parse_one_record) \
             .repeat(num_epochs) \
             .batch(args.batch_size)


if __name__ == '__main__':

    df = data_load('../data/data.csv', nrows=5)

    with tf.Session() as sess:
        data_set = read_csv_data(df)

        itor = data_set.make_one_shot_iterator()

        step = 0
        try:
            while 1:
                temp = itor.get_next()
                step += 1
                result = sess.run([temp])
                pprint(result[0])
        except tf.errors.OutOfRangeError:
            pass
