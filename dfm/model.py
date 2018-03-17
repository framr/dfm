import numpy as np
from time import time

from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import tensorflow as tf


class AttrDict(dict):
    def __getattr__(self, item):
        val = self[item]
        if isinstance(val, dict):
            return AttrDict(val)
        return val


class Model(object):
    def __init__(self, conf):
        pass
    def _create_graph(self):
        pass
    def _create_session(self):
        pass
    def _create_placeholders(self):
        pass
    def _create_weights(self):
        pass
    def _fit_on_batch(self):
        pass
    def predict(self):
        pass


DEFAULT_DEEPFM_CONF = {
    "embedding_size" : 8,
    "dropout_fm" : [1.0, 1.0],
    "deep_layers" : [32, 32],
    "dropout_deep" : [0.5, 0.5, 0.5],
    "deep_layers_activation" : tf.nn.relu,
    "epoch" : 10,
    "batch_size" : 256,
    "learning_rate" : 0.001,
    "optimizer_type" : "adam",
    "batch_norm" : 0,
    "batch_norm_decay" : 0.995,
    "verbose" : False,
    "random_seed" : 2016,
    "use_fm" : True,
    "use_deep" : True,
    "loss_type" : "logloss",
    "l2_reg" : 0.0
}

class DeepFMModel(Model):
    def __init__(self, conf):
        self.nfeatures = conf["num_features"]          # = M
        self.nfields = conf["num_fields"]              # = F
        self.embedding_size = conf["embedding_size"]   # = D
        # batch_size                                     = N
        self.debug_period_batches = 100

        self.dropout_fm = conf["dropout_fm"]
        self.deep_layers = conf["deep_layers"]
        self.dropout_deep = conf["dropout_deep"]
        self.deep_layers_activation = conf["deep_layers_activation"]
        self.use_fm = conf["use_fm"]
        self.use_deep = conf["use_deep"]
        self.l2_reg = conf["l2_reg"]
        self.num_epochs = conf["num_epochs"]
        self.batch_size = conf["batch_size"]
        self.learning_rate = conf["learning_rate"]
        self.optimizer_type = conf["optimizer_type"]
        self.batch_norm = conf["batch_norm"]
        self.batch_norm_decay = conf["batch_norm_decay"]
        self.verbose = conf.get("verbose", True)
        self.random_seed = conf.get("random_seed", 100500)
        self.loss_type = conf["loss_type"]
        self.eval_metric = conf["eval_metric"]
        self.train_result, self.valid_result = [], []

    def _create_placeholders(self):
        inputs = AttrDict()
        # N x F
        inputs.features = tf.placeholder(tf.int32, shape=[None, None], name="features")
        # N x 1
        inputs.labels = tf.placeholder(tf.float32, shape=[None, 1], name="labels")
        inputs.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
        inputs.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
        inputs.train_phase = tf.placeholder(tf.bool, name="train_phase")
        return inputs

    def _create_deep_layer_weights(self, size_in, size_out):
        glorot = np.sqrt(2.0 / (size_in + size_out))
        layer = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(size_in, size_out), dtype=np.float32))
        bias = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, size_out)), dtype=np.float32)
        return layer, bias

    def _create_weights(self):
        weights = AttrDict()
        # unary feature weights, shape = nfeatures x 1
        weights.unary = tf.Variable(tf.random_uniform([self.nfeatures, 1], 0.0, 1.0), name="unary")
        # embeddings, shape = nfeatures x D
        weights.embeddings = tf.Variable(
            tf.random_normal([self.nfeatures, self.embedding_size], 0.0, 0.01), name="feature_embeddings")

        # deep layers
        num_layers = len(self.deep_layers)
        input_size = self.nfields * self.embedding_size

        weights.layer_0, weights.bias_0 = self._create_deep_layer_weights(input_size, self.deep_layers[0])
        for i in xrange(1, num_layers):
            weights["layer_%d" % i], weights["bias_%d"] = self._create_deep_layer_weights(
                self.deep_layers[i - 1], self.deep_layers[i])

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.nfields + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.nfields + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights.concat_projection = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32)  # layers[i-1]*layers[i]
        weights.concat_bias = tf.Variable(tf.constant(0.01), dtype=np.float32)
        return weights

    def _create_prepare_embeddings_subgraph(unary, embeddings, features):
        feat_unary = tf.nn.embedding_lookup(unary, features)  # * F * 1
        feat_embeddings = tf.nn.embedding_lookup(embeddings, features)
        return feat_unary, feat_embeddings

    def _create_fm_subgraph(self, feat_unary, feat_embeddings):
        unary = tf.nn.dropout(feat_unary, self.dropout_keep_fm[0])  # N * F
        sum_embedding = tf.reduce_sum(feat_embeddings, 1)  # N * D
        sum_embedding_square = tf.square(sum_embedding)  # N * D
        # square_sum part
        squared_features_emb = tf.square(feat_embeddings)
        squared_sum_embedding = tf.reduce_sum(squared_features_emb, 1)  # N * D
        # second order
        quadratic = 0.5 * tf.subtract(sum_embedding_square, squared_sum_embedding)  # N * D
        quadratic = tf.nn.dropout(quadratic, self.dropout_keep_fm[1])  # N * D
        return unary, quadratic

    def _create_deep_subgraph(self, weights, feat_embeddings, num_layers):
        y_deep = tf.reshape(feat_embeddings, shape=[-1, self.nfields * self.embedding_size])  # None * (F*D)
        y_deep = tf.nn.dropout(y_deep, self.dropout_keep_deep[0])
        for i in xrange(len(num_layers)):
            y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" % i]), weights["bias_%d" % i])  # None * layer[i] * 1
            if self.batch_norm:
                y_deep = self.batch_norm_layer(y_deep, train_phase=self.train_phase, scope_bn="bn_%d" % i)  # None * layer[i] * 1
            y_deep = self.deep_layers_activation(y_deep)
            y_deep = tf.nn.dropout(y_deep, self.dropout_keep_deep[1 + i])
        return y_deep

    def _create_loss_subgraph(self, labels, predictions, weights):
        if self.loss_type == "logloss":
            out = tf.nn.sigmoid(predictions)
            loss = tf.losses.log_loss(labels, predictions)
        elif self.loss_type == "mse":
            loss = tf.nn.l2_loss(tf.subtract(labels, predictions))
        # l2 regularization on weights
        if self.l2_reg > 0:
            loss += tf.contrib.layers.l2_regularizer(
                self.l2_reg)(weights.concat_projection)
            if self.use_deep:
                for i in xrange(len(self.deep_layers)):
                    loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(weights["layer_%d" % i])
        return loss

    def _count_parameters(self):
        return 0

    def _create_optimizer_subgraph(self, loss):
        if self.optimizer_type == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                               epsilon=1e-8).minimize(loss)
        elif self.optimizer_type == "adagrad":
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                  initial_accumulator_value=1e-8).minimize(loss)
        elif self.optimizer_type == "gd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)
        elif self.optimizer_type == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                loss)
        return optimizer

    def _create_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)
            self.placeholders = self._create_placeholders()
            weights = self._create_weights()

            feat_unary, feat_embeddings = self._prepare_embeddings_subgraph(weights.unary,
                                                                            weights.embeddings,
                                                                            self.placeholders.features)
            fm_unary, fm_quad = self._create_fm_subgraph(feat_unary, feat_embeddings)
            y_deep = self._create_deep_subgraph(weights, feat_embeddings, self.deep_layers)
            if self.use_fm and self.use_deep:
                concat_outputs = tf.concat([fm_unary, fm_quad, y_deep], axis=1)
            elif self.use_fm:
                concat_outputs = tf.concat([fm_unary, fm_quad], axis=1)
            elif self.use_deep:
                concat_outputs = y_deep
            out = tf.add(tf.matmul(concat_outputs, weights.concat_projection), weights.concat_bias)

            loss = self._create_loss_subgraph(self.placeholders.labels, out, weights)
            self.optimizer = self._create_optimizer_subgraph(loss)

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._create_session()
            self.sess.run(init)
            self._count_parameters()

    def _create_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def _fit_on_batch(self, batch):
        X, y = batch
        feed_dict = {self.placeholders.features: X,
                     self.placeholders.labels: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}

        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss, opt

    def fit(self, data_gen):
        counter = 0
        for epoch in xrange(self.num_epochs):
            data_gen.reset()
            for batch in data_gen:
                start_time = time()
                loss, opt = self._fit_on_batch(batch)
                dt = time.time() - start_time
                speed = len(batch) / dt
                counter += 1
                if counter % self.debug_period_batches:
                    print loss



class DSSMModel(Model):
    pass

class WideDeepModel(Model):
    pass
