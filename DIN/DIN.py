from time import time
from args import *
from dice import dice, p_relu
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers import batch_norm


class DIN(object):
    def __init__(self, features, feature_tags, max_len, dropout=0.7,
                 attention_size=[80, 40], layers=[200, 200, 200], batch_norm=True,
                 activate=dice, l2_reg=0.0001,
                 embedding_size=16, learning_rate=0.0002, eval_metric=roc_auc_score,
                 verbose=True, verbose_step=2000, random_seed=2019):

        self.features = features
        self.feature_tags = feature_tags
        self.max_len = max_len

        self.dropout = dropout
        self.attention_size = attention_size
        self.layers = layers
        self.batch_norm = batch_norm

        self.activate = activate
        self.l2_reg = l2_reg

        self.embedding_size = embedding_size
        self.learning_rate = learning_rate

        self.eval_metric = eval_metric
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.random_seed = random_seed

        self._init_graph()

    def _init_weights(self):
        weights = {}
        for s in self.feature_tags:
            n_value = len(self.feature_tags[s]) + 1

            weights[s+'_embedding'] = tf.Variable(
                tf.random_normal([n_value, self.embedding_size], 0.0, 0.01),
                name=s+'_embedding'
            )

        for i in range(len(self.layers)):
            input_size = self.layers[i-1] if i > 0 else len(self.features)*self.embedding_size
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
        input_size = self.layers[-1]
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

    def _attention(self, hist_vec, item_vec):
        max_item_num = hist_vec.get_shape()[1]
        item_vec = tf.tile(item_vec, (1, max_item_num))
        item_vec = tf.reshape(item_vec, [-1, max_item_num, self.embedding_size])

        out_put = tf.concat([item_vec, hist_vec], axis=2)

        for i in range(len(self.attention_size)):
            n_units = self.attention_size[i]
            out_put = tf.layers.dense(out_put, n_units, activation=tf.nn.sigmoid)

        out_put = tf.layers.dense(out_put, 1)
        out_put = tf.reshape(out_put, (-1, max_item_num, 1))

        masks = tf.sequence_mask(self.hist_len, max_item_num)
        masks = tf.expand_dims(masks, -1)
        masks = tf.cast(masks, tf.float32)
        out_put = tf.multiply(out_put, masks)

        user_vec = tf.reduce_sum(tf.multiply(out_put, hist_vec), axis=1)

        return user_vec

    def _init_graph(self):

        self.input = tf.placeholder(tf.string, [None, len(self.features)])
        self.hist_item = tf.placeholder(tf.int32, [None, self.max_len])
        self.hist_len = tf.placeholder(tf.int32, [None, ])
        self.label = tf.placeholder(tf.int32, [None, 1])

        self.drop_out = tf.placeholder(tf.float32)

        self.weights = self._init_weights()

        fm_result = []
        for i in range(len(self.features)):
            s = self.features[i]
            if s == history_col:
                continue

            table = tf.contrib.lookup.index_table_from_tensor(mapping=self.feature_tags[s], default_value=0)
            inp = self.input[:, i]
            split_tags = tf.string_split(inp, '|')

            sp_tensor = tf.SparseTensor(
                indices=split_tags.indices,
                values=table.lookup(split_tags.values),
                dense_shape=split_tags.dense_shape
            )

            fm_vec = tf.nn.embedding_lookup_sparse(self.weights[s + '_embedding'], sp_ids=sp_tensor, sp_weights=None)
            if s == item_col:
                item_vec = fm_vec
            fm_result.append(fm_vec)

        # get embedding vectors for history_items
        history_vec = tf.nn.embedding_lookup(self.weights[item_col+'_embedding'], self.hist_item)
        history_vec = self._attention(history_vec, item_vec)
        fm_result.append(history_vec)

        deep_part = tf.concat(fm_result, axis=1)
        num_layers = len(self.layers)
        for i in range(num_layers):
            deep_part = tf.matmul(deep_part, self.weights['weight_%d' % i])
            deep_part = tf.add(deep_part, self.weights['bias_%d' % i])
            deep_part = tf.nn.dropout(deep_part, self.drop_out)
            if self.batch_norm:
                deep_part = batch_norm(deep_part)

            deep_part = self.activate(deep_part)

        out = tf.matmul(deep_part, self.weights['project_weight'])
        self.out = tf.add(out, self.weights['project_bias'])

        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(self.label, self.out)
        if self.l2_reg > 0:
            self.loss += tf.contrib.layers.l2_regularizer(
                self.l2_reg)(self.weights["project_weight"])
            for i in range(len(self.layers)):
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["weight_%d" % i])

            for embedding_vec in fm_result:
                self.loss += tf.contrib.layers.l2_regularizer(
                self.l2_reg)(embedding_vec)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)

        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        table_init = tf.tables_initializer()
        self.sess = self._init_session()
        self.sess.run([init, table_init])

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def shuffle_in_unison_scary(self, *args):
        rng_state = np.random.get_state()
        for item in args:
            np.random.shuffle(item)
            np.random.set_state(rng_state)

    def get_batch(self, train_x, train_hist, train_len, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return train_x[start:end], train_hist[start:end], train_len[start:end], y[start:end]

    def fit_on_batch(self, x_batch, hist_batch, len_batch, y):
        feed_dict = {
            self.input: x_batch,
            self.hist_item: hist_batch,
            self.hist_len: len_batch,
            self.label: np.reshape(y, [-1, 1]),
            self.drop_out: self.dropout
        }
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def fit(self, train_x, train_hist, train_len,  y,
            valid_x=None, valid_hist=None, valid_len=None, valid_y=None,
            epoch=5, batch_size=512):
        has_valid = valid_x is not None
        train_result = []
        if has_valid:
            valid_result = []
        for epoch in range(epoch):
            t1 = time()
            self.shuffle_in_unison_scary(train_x, train_hist, train_len, y)
            total_batch = int(len(y) / batch_size)
            for i in range(total_batch):
                x_batch, hist_batch, len_batch, y_batch = self.get_batch(train_x, train_hist, train_len, y, batch_size, i)
                self.fit_on_batch(x_batch, hist_batch, len_batch, y_batch)
                if self.verbose > 0 and (i+1) % self.verbose_step == 0:
                    train_res = self.evaluate(x_batch, hist_batch, len_batch, y_batch)
                    if has_valid:
                        valid_res = self.evaluate(valid_x, valid_hist, valid_len, valid_y)
                        print("epoch%2d step %4d train-result %.6f valid-result %.6f [%.1f s]"
                              % (epoch+1, i+1, train_res, valid_res, time()-t1))
                    else:
                        print("epoch%2d step %4d train-result %.6f [%.1f s]"
                              % (epoch+1, i+1, train_res, time()-t1))

            train_result.append(self.evaluate(train_x, train_hist, train_len, y))
            if has_valid:
                valid_result.append(self.evaluate(valid_x, valid_hist, valid_len, valid_y))
                print("epoch%2d train-result %.6f valid-result %.6f [%.1f s]"
                      % (epoch + 1, train_result[-1], valid_result[-1], time() - t1))
            else:
                print("epoch%2d train-result %.6f [%.1f s]"
                      % (epoch + 1, train_result[-1], time() - t1))
            print('****epoch %2d finied!****' % (epoch + 1))

    def predict(self, train_x, train_hist, train_len, batch_size=512):
        dummy_y = [1] * len(train_x)
        batch_index = 0
        x_batch, hist_batch, len_batch, y_batch = self.get_batch(train_x, train_hist, train_len, dummy_y, batch_size, batch_index)
        y_pred = []
        while len(x_batch) > 0:
            feed_dict = {
                self.input: x_batch,
                self.hist_item: hist_batch,
                self.hist_len: len_batch,
                self.label: np.reshape(y_batch, [-1, 1]),
                self.drop_out: 1.0
            }
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)
            y_pred.append(batch_out)
            batch_index += 1
            x_batch, hist_batch, len_batch, y_batch = self.get_batch(train_x, train_hist, train_len, dummy_y, batch_size, batch_index)

        return np.concatenate(y_pred, axis=0)

    def evaluate(self, train_x, train_hist, train_len, y):
        y_pred = self.predict(train_x, train_hist, train_len)
        return self.eval_metric(y, y_pred)