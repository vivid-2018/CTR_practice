# encoding = utf8
"""
This is an implementation of NFM model with Tensorflow.
@author:vivid.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
import tensorflow as tf 
import numpy as np 
from time import time 
from tensorflow.contrib.layers import batch_norm

class NFM(BaseEstimator, TransformerMixin):

    def __init__(self, features, feature_size, embedding_size=16, learning_rate=0.001,
                 layers=[200, 200, 200], batch_norm=True, dropout=0.7,
                 optimizer_type='Adam',loss_type='logloss', 
                 verbose=True, greater_is_better=True, eval_metric=roc_auc_score, random_seed=2019):
        
        self.features = features
        self.feature_size = feature_size
        
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate

        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
 
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.verbose = verbose

        self.greater_is_better = greater_is_better
        self.eval_metric = eval_metric
        self.random_seed = random_seed

        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.input = tf.placeholder(tf.int32, [None,len(self.feature_size)])
            self.label = tf.placeholder(tf.int32, [None,1])
            self.drop_out = tf.placeholder(tf.float32)

            self.weights = self._init_weights()

            lr_result = []
            fm_result = []
            for i in range(len(self.features)):
                s = self.features[i]
                lr_vec = tf.gather(self.weights[s+'_lr_embedding'],self.input[:,i])
                fm_vec = tf.gather(self.weights[s+'_fm_embedding'],self.input[:,i])

                lr_result.append(lr_vec)
                fm_result.append(fm_vec)

            # wide part
            wide_part = tf.add(self.weights['bias_lr'], sum(lr_result))

            fm_wide = tf.concat(fm_result, axis=1)
            fm_wide = tf.reshape(fm_wide, (-1, self.embedding_size, len(self.features)))
            fm_wide = tf.square(tf.reduce_sum(fm_wide, axis=2)) - tf.reduce_sum(tf.square(fm_wide), axis=2)
            fm_wide = tf.reduce_sum(fm_wide, axis=1, keepdims=True)*0.5

            wide_part = tf.add(wide_part, fm_wide)

            # deep part
            deep_part = tf.concat(fm_result, axis=1)
            deep_part = tf.reshape(deep_part, (-1, len(self.features), self.embedding_size))
            deep_part = tf.square(tf.reduce_sum(deep_part, axis=1)) - tf.reduce_sum(tf.square(deep_part), axis=1)
            deep_part = 0.5 * deep_part

            num_layers = len(self.layers)
            for i in range(num_layers):
                deep_part = tf.matmul(deep_part, self.weights['weight_%d' % i])
                deep_part = tf.add(deep_part, self.weights['bias_%d' % i])
                deep_part = tf.nn.dropout(deep_part,self.drop_out)
                if self.batch_norm:
                    deep_part = batch_norm(deep_part)

                deep_part = tf.nn.relu(deep_part)

            out = tf.concat([wide_part, deep_part], axis=1)

            out = tf.matmul(out, self.weights['project_weight'])
            self.out = tf.add(out, self.weights['project_bias'])

            if self.loss_type == 'logloss':
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label,self.out)
            elif self.loss_type == 'mse':
                self.loss = tf.nn.l2_loss(tf.subtract(self.label,self.out))
            else:
                self.loss = self.loss_type(self.label,self.out)


            if self.optimizer_type.lower() == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type.lower() == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type.lower() == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type.lower() == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)


    def get_batch(self, X, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return X[start:end], y[start:end]


    def fit_on_batch(self, X, y):
        feed_dict = {
            self.input: X,
            self.label: y,
            self.drop_out: self.dropout
        }
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def fit(self, X, y, epoch=10, batch_size=512,
            X_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        has_valid = X_valid is not None
        train_result = []
        if has_valid:
            valid_result = []
        for epoch in range(epoch):
            t1 = time()
            self.shuffle_in_unison_scary(X, y)
            total_batch = int(len(y) / batch_size)
            for i in range(total_batch):
                X_batch, y_batch = self.get_batch(X, y, batch_size, i)
                self.fit_on_batch(X_batch, y_batch)

            train_result.append(self.evaluate(X, y))
            if has_valid:
                valid_result.append(self.evaluate(X_valid, y_valid))
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result[-1], valid_result[-1], time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result[-1], time() - t1))
            if has_valid and early_stopping and self.training_termination(valid_result):
                break

        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(valid_result)
            else:
                best_valid_score = min(valid_result)
            best_epoch = valid_result.index(best_valid_score)
            best_train_score = train_result[best_epoch]
            X_train = np.concatenate([X, X_valid], axis=0)
            y = np.concatenate([y, y_valid], axis=0)
            for epoch in range(100):
                self.shuffle_in_unison_scary(X_train, y)
                total_batch = int(len(y) / batch_size)
                for i in range(total_batch):
                    X_batch, y_batch = self.get_batch(X_train, y, batch_size, i)
                    self.fit_on_batch(X_batch, y_batch)
                # check
                train_result = self.evaluate(X_train, y)
                if abs(train_result - best_train_score) < 0.001 or \
                    (self.greater_is_better and train_result > best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4] and \
                    valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4] and \
                    valid_result[-4] > valid_result[-5]:
                    return True
        return False


    def predict(self, X, batch_size=512):
        dummy_y = [1] * len(X)
        batch_index = 0
        X_batch, y_batch = self.get_batch(X, dummy_y, batch_size, batch_index)
        y_pred = []
        while len(X_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {
                self.input: X_batch,
                self.label: np.reshape(y_batch,[-1,1]),
                self.drop_out: 1.0
            }
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)
            y_pred.append(batch_out)
            batch_index += 1
            X_batch, y_batch = self.get_batch(X, dummy_y, batch_size, batch_index)

        return np.concatenate(y_pred,axis=0)


    def _init_weights(self):
        weights = {}
        for s in self.feature_size:
            n_value = self.feature_size[s]
            weights[s+'_lr_embedding'] = tf.Variable(
                tf.random_normal([n_value,1],0.0,0.01),
                name=s+'_lr_embedding'
            )
            weights[s+'_fm_embedding'] = tf.Variable(
                tf.random_normal([n_value,self.embedding_size],0.0,0.01),
                name=s+'_fm_embedding'
            )
        weights['bias_lr'] = tf.Variable(
            tf.random_uniform([1,1],0.0,1.0),
            name = 'bias_lr'
        )

        for i in range(len(self.layers)):
            input_size = self.layers[i-1] if i > 0 else self.embedding_size
            output_size = self.layers[i]

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
        input_size = self.layers[-1] + 1
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



    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return self.eval_metric(y, y_pred)
