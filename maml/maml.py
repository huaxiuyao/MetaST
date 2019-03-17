import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


class BasicModel:
    def __init__(self, dim_input, dim_output, seq_length,
                 filter_num, dim_cnn_flatten, dim_fc, dim_lstm_hidden,
                 update_lr, meta_lr, meta_batch_size, update_batch_size,
                 test_num_updates):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.channels = dim_output
        self.img_size = int(np.sqrt(self.dim_input / self.channels))

        self.dim_output = dim_output
        self.seq_length = seq_length
        self.filter_num = filter_num
        self.dim_cnn_flatten = dim_cnn_flatten
        self.dim_fc = dim_fc
        self.dim_lstm_hidden = dim_lstm_hidden

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.update_batch_size = update_batch_size
        self.test_num_updates = test_num_updates

        self.meta_batch_size = meta_batch_size

        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

    def update(self, loss, weights):
        grads = tf.gradients(loss, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        new_weights = dict(
            zip(weights.keys(), [weights[key] - self.update_lr * gradients[key] for key in weights.keys()]))
        return new_weights

    def construct_convlstm(self):
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_conv1'] = tf.Variable(tf.zeros([self.filter_num]))

        weights['conv2'] = tf.get_variable('conv2', [k, k, self.filter_num, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_conv2'] = tf.Variable(tf.zeros([self.filter_num]))

        weights['conv3'] = tf.get_variable('conv3', [k, k, self.filter_num, self.filter_num],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b_conv3'] = tf.Variable(tf.zeros([self.filter_num]))

        weights['fc1'] = tf.Variable(tf.random_normal([self.dim_cnn_flatten, self.dim_fc]), name='fc1')
        weights['b_fc1'] = tf.Variable(tf.zeros([self.dim_fc]))

        weights['kernel_lstm'] = tf.get_variable('kernel_lstm', [self.dim_fc + self.dim_lstm_hidden,
                                                                 4 * self.dim_lstm_hidden])
        weights['b_lstm'] = tf.Variable(tf.zeros([4 * self.dim_lstm_hidden]))

        weights['b_fc2'] = tf.Variable(tf.zeros([self.dim_output]))

        return weights

    def cnn(self, inp, weights):
        def conv_block(cinp, cweight, bweight, activation):
            """ Perform, conv, batch norm, nonlinearity, and max pool """
            stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
            conv_output = tf.nn.conv2d(cinp, cweight, no_stride, 'SAME') + bweight
            return activation(conv_output)

        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b_conv1'], tf.nn.relu)
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b_conv2'], tf.nn.relu)
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b_conv3'], tf.nn.relu)
        hidden4 = tf.reshape(hidden3, [-1, np.prod([int(dim) for dim in hidden3.get_shape()[1:]])])
        return tf.matmul(hidden4, weights['fc1']) + weights['b_fc1']

    def lstm(self, inp, weights):
        def lstm_block(linp, pre_state, kweight, bweight, activation):
            sigmoid = math_ops.sigmoid
            one = constant_op.constant(1, dtype=dtypes.int32)
            c, h = pre_state

            gate_inputs = math_ops.matmul(
                array_ops.concat([linp, h], 1), kweight)
            gate_inputs = nn_ops.bias_add(gate_inputs, bweight)

            i, j, f, o = array_ops.split(
                value=gate_inputs, num_or_size_splits=4, axis=one)

            forget_bias_tensor = constant_op.constant(1.0, dtype=f.dtype)

            add = math_ops.add
            multiply = math_ops.multiply
            new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                        multiply(sigmoid(i), activation(j)))
            new_h = multiply(activation(new_c), sigmoid(o))

            new_state = [new_c, new_h]
            return new_h, new_state

        inp = tf.unstack(tf.transpose(inp, perm=[1, 0, 2]))
        state = [tf.zeros([self.update_batch_size, self.dim_lstm_hidden]),
                 tf.zeros([self.update_batch_size, self.dim_lstm_hidden])]
        output = None
        for t in range(len(inp)):
            output, state = lstm_block(inp[t], state,
                                       weights['kernel_lstm'], weights['b_lstm'],
                                       tf.nn.tanh)
        return output

    def forward_convlstm(self, inp, weights):
        inp = tf.reshape(inp, [-1, self.dim_input])

        cnn_outputs = self.cnn(inp, weights)
        cnn_outputs = tf.reshape(cnn_outputs, [-1, self.seq_length, self.dim_fc])

        lstm_outputs = self.lstm(cnn_outputs, weights)
        return lstm_outputs


class STDN(BasicModel):
    def __init__(self, dim_input, dim_output, seq_length,
                 filter_num, dim_cnn_flatten, dim_fc, dim_lstm_hidden,
                 update_lr, meta_lr, meta_batch_size, update_batch_size,
                 test_num_updates):
        print("Initializing STDN...")
        BasicModel.__init__(self, dim_input, dim_output, seq_length,
                            filter_num, dim_cnn_flatten, dim_fc, dim_lstm_hidden,
                            update_lr, meta_lr, meta_batch_size, update_batch_size,
                            test_num_updates)

    def loss_func(self, pred, label):
        pred = tf.reshape(pred, [-1])
        label = tf.reshape(label, [-1])
        return tf.reduce_mean(tf.square(pred - label))

    def construct_model(self):
        with tf.variable_scope('model', reuse=None):
            with tf.variable_scope('maml', reuse=None):
                self.weights = weights = self.construct_convlstm()
                weights['fc2'] = tf.Variable(tf.random_normal(
                    [self.dim_lstm_hidden, self.dim_output]), name='fc6')   # output layer

            num_updates = self.test_num_updates

            def task_metalearn(inp):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                task_outputa = self.forward(inputa, weights)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                fast_weights = self.update(task_lossa, weights)

                output = self.forward(inputb, fast_weights)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights), labela)
                    fast_weights = self.update(loss, fast_weights)

                    output = self.forward(inputb, fast_weights)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                return task_output

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]

            inputs = (self.inputa, self.inputb, self.labela, self.labelb)
            result = tf.map_fn(task_metalearn,
                               elems=inputs,
                               dtype=out_dtype,
                               parallel_iterations=self.meta_batch_size)
            outputas, outputbs, lossesa, lossesb = result

        # Performance & Optimization
        self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size)
                                              for j in range(num_updates)]
        self.total_rmse1 = tf.sqrt(lossesa)
        self.total_rmse2 = [tf.sqrt(total_losses2[j]) for j in range(num_updates)]

        self.outputas, self.outputbs = outputas, outputbs
        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
        self.metatrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_losses2[num_updates-1])

        maml_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/maml")
        self.finetune_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1, var_list=maml_vars)

    def forward(self, inp, weights):
        convlstm_outputs = self.forward_convlstm(inp, weights)
        preds = tf.nn.tanh(tf.matmul(convlstm_outputs, weights['fc2']) + weights['b_fc2'])
        return preds


class AttentionSTDN(BasicModel):
    def __init__(self, dim_input, dim_output, seq_length,
                 filter_num, dim_cnn_flatten, dim_fc, dim_lstm_hidden,
                 update_lr, meta_lr, meta_batch_size, update_batch_size,
                 test_num_updates, cluster_num, memory_dim, cluster_loss_weight):
        print("Initializing attention STDN...")
        BasicModel.__init__(self, dim_input, dim_output, seq_length,
                            filter_num, dim_cnn_flatten, dim_fc, dim_lstm_hidden,
                            update_lr, meta_lr, meta_batch_size, update_batch_size,
                            test_num_updates)

        self.cluster_num = cluster_num
        self.memory_dim = memory_dim
        self.cluster_loss_weight = cluster_loss_weight

        self.clustera = tf.placeholder(tf.int32)
        self.clusterb = tf.placeholder(tf.int32)
        self.cluster_weighta = tf.placeholder(tf.float32)
        self.cluster_weightb = tf.placeholder(tf.float32)

    def loss_func(self, pred, label, att_val, cluster, cluster_weight):
        pred = tf.reshape(pred, [-1])
        label = tf.reshape(label, [-1])
        pred_loss = tf.reduce_mean(tf.square(pred - label))

        one_hot_cluster = tf.one_hot(cluster, self.cluster_num)
        cluster_loss = tf.reduce_mean(-cluster_weight * tf.reduce_sum(one_hot_cluster * tf.log(att_val), axis=1))
        return pred_loss, cluster_loss

    def construct_memory_weights(self):
        memory_weights = {}
        dtype = tf.float32

        init = tf.glorot_normal_initializer()
        memory_weights['M'] = tf.get_variable('mem', [self.cluster_num, self.memory_dim],
                                              initializer=init, dtype=dtype)
        memory_weights['Wa'] = tf.get_variable('att', [self.dim_lstm_hidden, self.memory_dim],
                                               initializer=init, dtype=dtype)
        memory_weights['fc'] = tf.get_variable('mem_fc', [self.memory_dim, self.memory_dim],
                                               initializer=init, dtype=dtype)
        return memory_weights

    def attention(self, inp, weights):
        score = tf.matmul(tf.matmul(inp, weights['Wa']), tf.transpose(weights['M']))
        return tf.nn.softmax(score)

    def forward(self, inp, weights, memory_weights):
        convlstm_outputs = self.forward_convlstm(inp, weights)

        attention_vals = self.attention(convlstm_outputs, memory_weights)
        attentive_cluster_reps = tf.matmul(
            tf.matmul(attention_vals, memory_weights['M']), memory_weights['fc'])

        final_outputs = tf.concat([convlstm_outputs, attentive_cluster_reps], axis=1)

        preds = tf.nn.tanh(tf.matmul(final_outputs, weights['fc2']) + weights['b_fc2'])
        return preds, attention_vals

    def combine_loss(self, loss, closs):
        return loss + self.cluster_loss_weight * closs

    def task_metalearn(self, inp):
        """ Perform gradient descent for one task in the meta-batch. """
        weights, memory_weights = self.weights, self.memory_weights

        inputa, inputb, labela, labelb, clustera, clusterb, cluster_weighta, cluster_weightb = inp
        task_outputbs, task_att_valbs, task_lossesb, task_clossesb = [], [], [], []

        task_outputa, task_att_vala = self.forward(inputa, weights, memory_weights)
        task_lossa, task_clossa = self.loss_func(task_outputa, labela, task_att_vala, clustera, cluster_weighta)

        fast_weights = self.update(
            self.combine_loss(task_lossa, task_clossa), weights)

        output, att_val = self.forward(inputb, fast_weights, memory_weights)
        task_outputbs.append(output)
        task_att_valbs.append(att_val)
        task_lossb, task_clossb = self.loss_func(output, labelb, att_val, clusterb, cluster_weightb)
        task_lossesb.append(task_lossb)
        task_clossesb.append(task_clossb)

        for j in range(self.test_num_updates - 1):
            outputa, att_vala = self.forward(inputa, fast_weights, memory_weights)
            loss, closs = self.loss_func(outputa, labela, att_vala, clustera, cluster_weighta)
            fast_weights = self.update(self.combine_loss(loss, closs), fast_weights)

            output, att_val = self.forward(inputb, fast_weights, memory_weights)
            task_outputbs.append(output)
            task_att_valbs.append(att_val)
            task_lossb, task_clossb = self.loss_func(output, labelb, att_val, clusterb, cluster_weightb)
            task_lossesb.append(task_lossb)
            task_clossesb.append(task_clossb)

        task_output = [task_outputa, task_outputbs,
                       task_att_vala, task_att_valbs,
                       task_lossa, task_lossesb,
                       task_clossa, task_clossesb]
        return task_output

    def construct_model(self):
        with tf.variable_scope('model', reuse=None):
            with tf.variable_scope('maml', reuse=None):
                self.weights = weights = self.construct_convlstm()
                weights['fc2'] = tf.Variable(tf.random_normal(
                    [self.dim_lstm_hidden + self.memory_dim, self.dim_output]), name='fc6')   # output layer
            with tf.variable_scope('memory', reuse=None):
                self.memory_weights = self.construct_memory_weights()

            num_updates = self.test_num_updates

            out_dtype = [tf.float32, [tf.float32]*num_updates,
                         tf.float32, [tf.float32]*num_updates,
                         tf.float32, [tf.float32]*num_updates,
                         tf.float32, [tf.float32]*num_updates]

            inputs = (self.inputa, self.inputb, self.labela, self.labelb, self.clustera, self.clusterb,
                      self.cluster_weighta, self.cluster_weightb)
            result = tf.map_fn(self.task_metalearn,
                               elems=inputs,
                               dtype=out_dtype,
                               parallel_iterations=self.meta_batch_size)
            outputas, outputbs, att_vala, att_valbs, lossesa, lossesb, clossesa, clossesb = result

        self.all_att_val1 = att_vala
        self.all_att_vals2 = att_valbs

        # Performance & Optimization
        self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size)
                                              for j in range(num_updates)]
        self.total_rmse1 = tf.sqrt(total_loss1)
        self.total_rmse2 = [tf.sqrt(total_losses2[j]) for j in range(num_updates)]

        self.total_closs1 = total_closs1 = tf.reduce_sum(clossesa) / tf.to_float(self.meta_batch_size)
        self.total_closses2 = total_closses2 = [tf.reduce_sum(clossesb[j]) / tf.to_float(self.meta_batch_size)
                                              for j in range(num_updates)]
        self.total_cluster_prob1 = 1 / tf.exp(total_closs1)
        self.total_cluster_prob2 = [1 / tf.exp(total_closses2[j]) for j in range(num_updates)]

        self.outputas, self.outputbs = outputas, outputbs
        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(
            self.combine_loss(total_loss1, total_closs1))

        self.metatrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(
            self.combine_loss(total_losses2[num_updates-1], total_closses2[num_updates-1]))

        maml_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/maml")
        self.finetune_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1, var_list=maml_vars)

