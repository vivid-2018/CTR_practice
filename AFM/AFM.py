# encoding = utf8
"""
This is an implementation of AFM model with Tensorflow.
@author:vivid.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
import tensorflow as tf 
import numpy as np 
from time import time 

class AFM(BaseEstimator, TransformerMixin):

    def __init__(self, features, feature_size, attention_size=8, embedding_size=16, learning_rate=0.001,
                 first_order=True, second_order=True, optimizer_type='Adam',loss_type='logloss', 
                 verbose=True, greater_is_better=True, eval_metric=roc_auc_score, random_seed=2019):
        assert first_order or second_order, "Must use first_order or second_order"
        
        self.features = features
        self.feature_size = feature_size
        
        self.attention_size = attention_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate

        self.first_order = first_order
        self.second_order = second_order

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

            self.weights = self._init_weights()

            lr_result = []
            fm_result = []
            for i in range(len(self.features)):
                s = self.features[i]
                lr_vec = tf.gather(self.weights[s+'_lr_embedding'],self.input[:,i])
                fm_vec = tf.gather(self.weights[s+'_fm_embedding'],self.input[:,i])

                lr_result.append(lr_vec)
                fm_result.append(fm_vec)

            self.lr_out = tf.add(self.weights['bias'],sum(lr_result))

            fm_result = tf.reshape(tf.concat(fm_result, axis=1), (-1, len(self.features), self.embedding_size))
            temp = []
            for i in range(len(self.features)):
                for j in range(i+1, len(self.features)):
                    temp.append(tf.expand_dims(tf.multiply(fm_result[:, i, :], fm_result[:, j, :]), dim=1))

            fm_result = tf.concat(temp, axis=1)

            fm_weight = self._attention_weight(fm_result)

            self.fm_out = tf.multiply(fm_result, fm_weight)
            self.fm_out = tf.reduce_sum(self.fm_out, axis=1)
            self.fm_out = tf.reduce_sum(self.fm_out, axis=1, keepdims=True)


            self.out = self.fm_out*self.second_order + self.lr_out*self.first_order

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


    def _attention_weight(self, fm_result):
        attention_weight = tf.reshape(fm_result, (-1, self.embedding_size))
        attention_weight = tf.matmul(attention_weight, self.weights['attention_weight'])
        attention_weight = tf.add(attention_weight, self.weights['attention_bias'])
        attention_weight = tf.nn.relu(attention_weight)
        attention_weight = tf.matmul(attention_weight, self.weights['attention_output'])
        num_ = len(self.features) * (len(self.features) - 1) // 2
        attention_weight = tf.reshape(attention_weight, shape=(-1, num_, 1))
        attention_weight = tf.nn.softmax(attention_weight, axis=1)
        return attention_weight


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
            weights['bias'] = tf.Variable(
                tf.random_uniform([1,1],0.0,1.0),
                name = 'bias'
            )
        weights['attention_weight'] = tf.Variable(
            tf.random_normal([self.embedding_size, self.attention_size]),
            name = 'attention_weight'
        )
        weights['attention_bias'] = tf.Variable(
            tf.random_uniform([1, self.attention_size]),
            name = 'attention_bias'
        )
        weights['attention_output'] = tf.Variable(
            tf.random_normal([self.attention_size, 1]),
            name = 'attention_output'
        )
        return weights


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
            self.label: y
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
                self.label: np.reshape(y_batch,[-1,1])
            }
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)
            y_pred.append(batch_out)
            batch_index += 1
            X_batch, y_batch = self.get_batch(X, dummy_y, batch_size, batch_index)

        return np.concatenate(y_pred,axis=0)


    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return self.eval_metric(y, y_pred)
