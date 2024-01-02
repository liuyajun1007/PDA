
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from mine_function import read_txt1, get_lapl_matrix, get_syn_sim, get_initial_weights_by_manifold

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 piRNA = np.zeros([4350, 4350]),
                 sim_d = np.zeros([21, 21]),
                 piRNA2 = np.zeros([4350, 4350]),
                 sim_d2 = np.zeros([21, 21]),
                 piRNA3=np.zeros([4350, 4350]),
                 sim_d3=np.zeros([21, 21]),
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        #！！！！
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg
        self.piRNA = piRNA
        self.sim_d = sim_d
        self.piRNA2 = piRNA2
        self.sim_d2 = sim_d2
        self.piRNA3 = piRNA3
        self.sim_d3 = sim_d3
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                                 name="feat_index")  # None * F

            self.feat_index2 = tf.placeholder(tf.int32, shape=[None, None],
                                             name="feat_index2")  # None * (3*F)



            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                                 name="feat_value")  # None * F

            self.feat_value2 = tf.placeholder(tf.float32, shape=[None, None],
                                             name="feat_value2")  # None * (3*F)



            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # model get the embedding weight
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                             self.feat_index2)  # None * (3*F) * K

            #get feature value
            feat_value = tf.reshape(self.feat_value, shape=[-1, 2, 1])
            #feat_value = tf.Print(feat_value, [feat_value.shape, feat_value], message="print feat_value value:")

            #1021扩增feat_value为3倍长
            feat_value2 = tf.reshape(self.feat_value2, shape=[-1, self.field_size, 1])
            #feat_value2 = tf.Print(feat_value2, [feat_value2.shape, feat_value2], message="print feat_value2 value:")

            # get the embeding of each node
            self.embeddings = tf.multiply(self.embeddings, feat_value2)
            #self.embeddings = tf.Print(self.embeddings, [self.embeddings.shape, self.embeddings], message="print embeddings value:")

            # ---------- first order term ----------
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index) # None * F * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
            self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0]) # None * F
            print("self.y_first_order:",self.y_first_order.shape[0],self.y_first_order.shape[1] )
            # ---------- second order term ---------------
            #get final embedding
            # embeddings拆分成6个矩阵，然后第一、三、五个叠加，第二、四、六个叠加
            self.embeddings_res = tf.split(self.embeddings,6,1 )
            print("self.embeddings:", self.embeddings.shape[0], self.embeddings.shape[1])
            self.embeddings_res1 = tf.add(self.embeddings_res[0],self.embeddings_res[2])
            self.embeddings_res1 = tf.add(self.embeddings_res1,self.embeddings_res[4])
            self.embeddings_res2 = tf.add(self.embeddings_res[1],self.embeddings_res[3])
            self.embeddings_res2 = tf.add(self.embeddings_res2,self.embeddings_res[5])

            self.embeddings_final=tf.concat([self.embeddings_res1, self.embeddings_res2], axis=1)
            print("self.embeddings_final:", self.embeddings_final.shape[0], self.embeddings_final.shape[1])

            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings_final, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K


            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings_final)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # None * K

            print("self.y_second_order:", self.y_second_order.shape[0], self.y_second_order.shape[1])

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.embeddings_final, shape=[-1, 2 * self.embedding_size]) # None * (F*K)， flatten the

            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

            print("self.y_deep:", self.y_deep.shape[0], self.y_deep.shape[1])

            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            print("concat_input", concat_input[0], concat_input[1])
            print("concat_projection:", self.weights["concat_projection"].shape[0], self.weights["concat_projection"].shape[1])
            #print("concat_bias:", self.weights["concat_bias"].shape[0], self.weights["concat_bias"].shape[1])

            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])


            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))


            # add the laplacian normalization
            # laplacian！！！
            lap_matx_m, lap_Dm = get_lapl_matrix(self.piRNA)
            lap_matx_d, lap_Dd = get_lapl_matrix(self.sim_d)

            lap_matx_m2, lap_Dm2 = get_lapl_matrix(self.piRNA2)
            lap_matx_d2, lap_Dd2 = get_lapl_matrix(self.sim_d2)

            lap_matx_m3, lap_Dm3 = get_lapl_matrix(self.piRNA3)
            lap_matx_d3, lap_Dd3 = get_lapl_matrix(self.sim_d3)

            lap_matx_m_1 = tf.constant(lap_matx_m, tf.float32)
            lap_matx_d_1 = tf.constant(lap_matx_d, tf.float32)

            lap_matx_m_2 = tf.constant(lap_matx_m2, tf.float32)
            lap_matx_d_2 = tf.constant(lap_matx_d2, tf.float32)

            lap_matx_m_3 = tf.constant(lap_matx_m3, tf.float32)
            lap_matx_d_3 = tf.constant(lap_matx_d3, tf.float32)

            ##feature_embeddings
            weight_emb = self.weights["feature_embeddings"]
            m_n = lap_matx_m.shape[0]
            d_n = lap_matx_d.shape[0]
            weight_emb_m = weight_emb[0:m_n,:] #piRNA number rows and embeding_size colums
            weight_emb_d = weight_emb[m_n:m_n+d_n, :]
            weight_emb_m2 = weight_emb[m_n+d_n: 2*m_n+d_n, :]
            weight_emb_d2 = weight_emb[2*m_n+d_n: 2*m_n + 2*d_n, :]
            weight_emb_m3 = weight_emb[2*m_n+2*d_n: 3 * m_n + 2*d_n, :]
            weight_emb_d3 = weight_emb[3 * m_n + 2*d_n: 3 * m_n + 3 * d_n, :]

            self.m_lap = tf.matmul(tf.matmul(tf.transpose(weight_emb_m), lap_matx_m_1), weight_emb_m)
            self.d_lap = tf.matmul(tf.matmul(tf.transpose(weight_emb_d), lap_matx_d_1), weight_emb_d)
            self.m_lap2 = tf.matmul(tf.matmul(tf.transpose(weight_emb_m2), lap_matx_m_2), weight_emb_m2)
            self.d_lap2 = tf.matmul(tf.matmul(tf.transpose(weight_emb_d2), lap_matx_d_2), weight_emb_d2)
            self.m_lap3 = tf.matmul(tf.matmul(tf.transpose(weight_emb_m3), lap_matx_m_3), weight_emb_m3)
            self.d_lap3 = tf.matmul(tf.matmul(tf.transpose(weight_emb_d3), lap_matx_d_3), weight_emb_d3)

            self.m_loss = tf.trace(self.m_lap)
            self.d_loss = tf.trace(self.d_lap)
            self.m_loss2 = tf.trace(self.m_lap2)
            self.d_loss2 = tf.trace(self.d_lap2)
            self.m_loss3 = tf.trace(self.m_lap3)
            self.d_loss3 = tf.trace(self.d_lap3)

            self.l_loss = tf.add(tf.multiply(self.m_loss, tf.constant([0.01], tf.float32)),tf.multiply(self.d_loss, tf.constant([0.01], tf.float32)))
            self.l_loss = tf.add(self.l_loss,
                                 tf.multiply(self.m_loss2, tf.constant([0.01], tf.float32)))
            self.l_loss = tf.add(self.l_loss,
                                tf.multiply(self.d_loss2, tf.constant([0.01], tf.float32)))
            self.l_loss = tf.add(self.l_loss,
                                 tf.multiply(self.m_loss3, tf.constant([0.01], tf.float32)))
            self.l_loss = tf.add(self.l_loss,
                                 tf.multiply(self.d_loss3, tf.constant([0.01], tf.float32)))
            self.loss = tf.add(self.loss, self.l_loss)



            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d"%i])

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            elif self.optimizer_type == "yellowfin":
                self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(
                    self.loss)

            # init
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


    def _initialize_weights(self):#initizing all the weights parameters
        weights = dict()

        # embeddings

        ##embedding 先对所有similarity matrix进行laplacian regularization，得到的结果进行加权
        #获得feature_embeddings weights
        weights["feature_embeddings"] = tf.Variable(
            #tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            (get_initial_weights_by_manifold(self.embedding_size, self.piRNA, self.sim_d, self.piRNA2, self.sim_d2,self.piRNA3, self.sim_d3)).astype(np.float32),
            name="feature_embeddings")  # feature_size * K

        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        # deep layers
        num_layer = len(self.deep_layers)
        #input_size = self.field_size * self.embedding_size
        input_size = 2 * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        # if self.use_fm and self.use_deep:
        #     input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        # elif self.use_fm:
        #     input_size = self.field_size + self.embedding_size
        # elif self.use_deep:
        #     input_size = self.deep_layers[-1]
        if self.use_fm and self.use_deep:
            input_size = 2 + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = 2 + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]


        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),dtype=np.float32)# layers[i-1]*layers[i]
        #print("weight", weights["concat_projection"])
                        #np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        #dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)
        #weights["concat_bias"] = tf.Variable(tf.constant(0), dtype=np.float32)
        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index+1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]


    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)


    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_index2: np.hstack((Xi,Xi,Xi)),
                     self.feat_value: Xv,
                     self.feat_value2: np.hstack((Xv,Xv,Xv)),
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        # loss,loss_l, loss_m, loss_d, opt = self.sess.run((self.loss,self.l_loss, self.m_loss, self.d_loss, self.optimizer), feed_dict=feed_dict)
        loss, opt = self.sess.run(
            (self.loss, self.optimizer), feed_dict=feed_dict)
        # return loss, loss_l, loss_m, loss_d
        return loss


    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        AUC = []
        AUPR = []
        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            #print("total_batch",  total_batch)

            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
            # evaluate training and validation datasets
            train_result_auc, train_AUPR = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_result_auc)

            if has_valid:
                valid_result_auc, valid_AUPR = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_result_auc)
                # AUC.append(valid_result_auc)
                # AUPR.append(valid_AUPR)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f, valid-AUPR=%.4f, [%.1f s]"
                        % (epoch + 1, train_result_auc, valid_result_auc, valid_AUPR, time() - t1))

                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                        % (epoch + 1, train_result_auc, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,
                                                                self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # check

                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                valid_result = self.evaluate(Xi_valid,Xv_valid,y_valid)
                #if abs(train_result - best_train_score) < 0.001 or \
                if abs(valid_result - best_train_score) < 0.001 or \
                        (self.greater_is_better and train_result > best_train_score) or \
                    ((not self.greater_is_better) and train_result < best_train_score):
                    break
        print("AUC", AUC)
        return AUC, AUPR



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


    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_index2: np.hstack((Xi_batch,Xi_batch,Xi_batch)),
                         self.feat_value: Xv_batch,
                         self.feat_value2: np.hstack((Xv_batch,Xv_batch,Xv_batch)),
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}

            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred


    def evaluate(self, Xi, Xv, y):
        y_pred = self.predict(Xi, Xv)
        #print("y_pred", y_pred)
        ROC = self.eval_metric(y, y_pred)
        precision, recall, _thresholds = precision_recall_curve(y, y_pred)
        AUPR = auc(recall, precision)
        return ROC, AUPR

