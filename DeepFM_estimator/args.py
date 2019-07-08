# encoding = utf8
import os
import tensorflow as tf


class args:

    file_path = '../data/data.csv'

    nrows = None

    epochs = 10

    batch_size = 1024

    embedding_size = 8

    buffer_size = 200

    CATEGORICAL_FEATURES = ['userId', 'movieId', 'tag']

    learning_rate = 0.001

    activation = tf.nn.relu

    model_path = ''

    last_model_path = None

    dropout = 0.7

    layers = [200, 200, 200]

    batch_norm = False

    feature_size = {
        'userId': 7801,
        'movieId': 19545,
        'tag': 38644
    }


