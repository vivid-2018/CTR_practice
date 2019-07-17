# coding = utf-8

import tensorflow as tf
import pandas as pd
from args import *
from data_load import process_data
from DIN import DIN


def run_din():
    train = pd.read_csv(train_path, nrows=None).fillna('0')
    test = pd.read_csv(test_path, nrows=None).fillna('0')

    train_x, train_y, train_hist, train_len = process_data(train)

    test_x, test_y, test_hist, test_len = process_data(test)

    model = DIN(features, feature_tags, max_len=max_len, activate=tf.nn.relu)

    model.fit(train_x, train_hist, train_len, train_y,
              test_x, test_hist, test_len, test_y, epoch=10)


if __name__ == '__main__':
    run_din()

