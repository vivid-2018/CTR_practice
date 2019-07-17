# coding = utf-8

import tensorflow as tf


def dice(inp, name=''):
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha'+name, inp.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        beta = tf.get_variable('beta'+'name', inp.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                               dtype=tf.float32)

    x_normed = tf.layers.batch_normalization(inp, center=False, scale=False, name=name, reuse=tf.AUTO_REUSE)
    x_p = tf.sigmoid(beta * x_normed)

    return alphas * (1.0 - x_p) * inp + x_p * inp


def p_relu(inp):
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha', inp.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
    pos = tf.nn.relu(inp)
    neg = alphas * (inp - abs(inp)) * 0.5

    return pos + neg

