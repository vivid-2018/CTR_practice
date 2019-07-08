# encoding = utf8

import pickle
import tensorflow as tf
import numpy as np
from args import args
from data_loader import read_csv_data
from tensorflow.contrib.layers import batch_norm


def init_weights(params):
    weights = dict()
    embedding_size = params['embedding_size']
    layers = params['layers']
    feature_size = params['feature_size']

    # FM & LR embedding 权重
    for col in args.CATEGORICAL_FEATURES:
        n_values = feature_size[col]
        weights[col+'_fm'] = tf.Variable(
            tf.random_normal([n_values, embedding_size], 0.0, 0.01),
            name=col+'_fm_embedding'
        )

        weights[col + '_lr'] = tf.Variable(
            tf.random_normal([n_values, 1], 0.0, 0.01),
            name=col + '_lr_embedding'
        )
    # LR偏置
    weights['bias_lr'] = tf.Variable(
        tf.random_uniform([1, 1], 0.0, 1.0),
        name= 'bias_lr'
    )
    # 全连接 权重&偏置
    for i in range(len(layers)):
        input_size = layers[i-1] if i > 0 else embedding_size
        output_size = layers[i]

        glorot = np.sqrt(2.0 / (input_size + output_size))

        weights['weight_%d' % i] = tf.Variable(
            tf.random_normal([input_size, output_size], 0.0, glorot),
            name='weight_%d' % i
        )

        weights['bias_%d' % i] = tf.Variable(
            tf.random_uniform([1, output_size], 0.0, 1.0),
            name='bias_%d' % i
        )
    # 投影权重&偏置
    input_size = layers[-1] + 1
    output_size = 1
    glorot = np.sqrt(2.0 / (input_size + output_size))
    weights['project_weight'] = tf.Variable(
        tf.random_normal([input_size, 1], 0.0, glorot),
        name='project_weight'
    )

    weights['project_bias'] = tf.Variable(
        tf.random_uniform([1, 1], 0.0, 1.0),
        name='project_bias'
    )

    return weights


def creat_graph(features, labels, mode, params):
    lr_embedding_vecs = []
    fm_embedding_vecs = []
    weights = init_weights(params)

    for i in range(len(args.CATEGORICAL_FEATURES)):
        col = args.CATEGORICAL_FEATURES[i]
        feature_ids = tf.cast(features[:, i], dtype=tf.int32)  # 拿到特征的id列

        lr_embedding = tf.gather(weights[col+'_lr'], feature_ids)
        lr_embedding_vecs.append(lr_embedding)

        fm_embedding = tf.gather(weights[col+'_fm'], feature_ids) # None * 32
        fm_embedding_vecs.append(fm_embedding)

    # wide part
    wide_part = tf.add(weights['bias_lr'], sum(lr_embedding_vecs))

    # deep part
    deep_part = tf.concat(fm_embedding_vecs, axis=1)
    deep_part = tf.reshape(deep_part, shape=(-1, args.embedding_size, len(args.CATEGORICAL_FEATURES)))
    deep_part = tf.square(tf.reduce_sum(deep_part, axis=2)) - tf.reduce_sum(tf.square(deep_part), axis=2)
    deep_part = 0.5 * deep_part
    num_layers = len(params['layers'])
    for i in range(num_layers):
        deep_part = tf.matmul(deep_part, weights['weight_%d' % i])
        deep_part = tf.add(deep_part, weights['bias_%d' % i])
        deep_part = tf.nn.dropout(deep_part, params['dropout'])
        if params['batch_norm']:
            deep_part = batch_norm(deep_part)

        deep_part = params['activation'](deep_part)

    out = tf.concat([wide_part, deep_part], axis=1)
    out = tf.matmul(out, weights['project_weight'])
    out = tf.add(out, weights['project_bias'])
    out = tf.sigmoid(out)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.log_loss(labels, out)
        global_step = tf.train.get_or_create_global_step()
        train_op = tf.train.AdamOptimizer(params['learning_rate'],
            beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss=loss, global_step=global_step)
        eval_metric_ops = {
            'auc': tf.metrics.auc(labels=labels, predictions=out)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            predictions={'y_pre': out}
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.log_loss(labels, out)
        global_step = tf.train.get_or_create_global_step()
       
        eval_metric_ops = {
            'auc': tf.metrics.auc(labels=labels, predictions=out)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops=eval_metric_ops,
            predictions={'y_pre': out}
        )
    else:  # mode == tf.estimator.ModeKeys.PREDICT
        return tf.estimator.EstimatorSpec(
            mode, predictions={'y_pre': out}
        )


