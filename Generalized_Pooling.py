# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers

def get_attention_weights(inputs, num_da, scope, scope_reuse=False, regularizer=None):
    with tf.variable_scope(scope, reuse=scope_reuse):
        input_size = inputs.get_shape()[-1]

        W_1 = tf.get_variable("W_1", shape=[input_size, num_da],
                              dtype=tf.float32, initializer=tf.truncated_normal_initializer(),
                              regularizer=regularizer)
        b_1 = tf.get_variable("b_1", shape=[num_da],
                              dtype=tf.float32, initializer=tf.constant_initializer(0.1),
                              regularizer=regularizer)
        W_2 = tf.get_variable("W_2", shape=[num_da, input_size],
                              dtype=tf.float32, initializer=tf.truncated_normal_initializer(),
                              regularizer=regularizer)
        b_2 = tf.get_variable("b_2", shape=[input_size],
                              dtype=tf.float32, initializer=tf.constant_initializer(0.1),
                              regularizer=regularizer)

        temp = tf.nn.relu(tf.einsum('abc,cd->abd', inputs, W_1) + b_1)
        A = tf.einsum('abc,cd->abd', temp, W_2) + b_2

    return A, W_1

def get_masked_weights(inputs, seq_len, max_len):
    seq_mask = tf.sequence_mask(seq_len, max_len, dtype=tf.float32)  # [batch_size, max_len]
    seq_mask = tf.expand_dims(seq_mask, -1)  # [batch_size, max_len, 1]
    outputs = inputs*seq_mask + (seq_mask - 1) * 1e9
    outputs = tf.nn.softmax(outputs, axis=1)
    return outputs

def fc_layer(inputs, output_size, dropout_keep_prob, scope, scope_reuse=False, regularizer=None):
    with tf.variable_scope(scope, reuse=scope_reuse):
        input_size = inputs.get_shape()[-1].value
        W = tf.get_variable("W_fc", shape=[input_size, output_size], initializer=tf.orthogonal_initializer(),
                            regularizer=regularizer)
        b = tf.get_variable("b_fc", shape=[output_size], initializer=tf.zeros_initializer(),
                            regularizer=regularizer)
        outputs = tf.nn.relu(tf.matmul(inputs, W) + b)
        outputs = tf.nn.dropout(outputs, keep_prob=dropout_keep_prob)
    return outputs


class GeneralizedPooling(object):
    def __init__(self, max_len_left, max_len_right, vocab_size,
                embedding_size, num_stack, rnn_size, num_heads, num_da,
                 num_hidden, penalty_type=0, mu=1e-2, l2_reg_lambda=0.0):

        regularizer = layers.l2_regularizer(l2_reg_lambda)
        # placeholder for input data
        self.input_left = tf.placeholder(tf.int32, shape=[None, max_len_left],
                                         name="input_left")
        self.input_right = tf.placeholder(tf.int32, shape=[None, max_len_right],
                                          name="input_right")
        self.input_y = tf.placeholder(tf.float32, shape=[None, 2],
                                      name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            W_weight = tf.get_variable(name='embedding_weights', shape=[vocab_size, embedding_size],
                                        dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
            self.embedding_weight = tf.concat([tf.zeros([1, embedding_size]), W_weight[1:, :]],
                                              axis=0)

            self.emb_left  = tf.nn.embedding_lookup(self.embedding_weight, self.input_left, name="emb_left")
            self.emb_right = tf.nn.embedding_lookup(self.embedding_weight, self.input_right, name="emb_right")


        with tf.name_scope("sequence_encoder"):
            self.length_left = self.get_length(self.input_left)
            self.length_right = self.get_length(self.input_right)

            bilstm_stacked_left  = [self.emb_left]
            bilstm_stacked_right = [self.emb_right]
            for i in range(num_stack):
                bilstm_left  = tf.concat(bilstm_stacked_left, 2)
                bilstm_right = tf.concat(bilstm_stacked_right, 2)

                cell_fw = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True, name="cell_fw_{}".format(i))
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
                cell_bw = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True, name="cell_bw_{}".format(i))
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)

                output_left, states_left   = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                            cell_bw,
                                                                            bilstm_left,
                                                                            dtype=tf.float32,
                                                                            sequence_length=self.length_left)

                output_right, states_right = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                           cell_bw,
                                                                           bilstm_right,
                                                                           dtype=tf.float32,
                                                                           sequence_length=self.length_right)

                H_left  = tf.concat(output_left, 2)
                H_right = tf.concat(output_right, 2)

                bilstm_stacked_left  = [self.emb_left, H_left]
                bilstm_stacked_right = [self.emb_right, H_right]

            self.H_left  = bilstm_stacked_left[-1]
            self.H_right = bilstm_stacked_right[-1]


        with tf.name_scope('generalized_pooling'):
            v_left  = []
            v_right = []
            W_1 = []
            for i in range(num_heads):
                # consider mask the padding token
                A_i_left, w_1  = get_attention_weights(self.H_left, num_da, "head_{}".format(i), scope_reuse=False, regularizer=regularizer)
                A_i_left = get_masked_weights(A_i_left, self.length_left, max_len_left)

                A_i_right, _   = get_attention_weights(self.H_right, num_da, "head_{}".format(i), scope_reuse=True, regularizer=regularizer)
                A_i_right = get_masked_weights(A_i_right, self.length_right, max_len_right)

                v_i_left  = tf.reduce_sum(tf.multiply(A_i_left, self.H_left), axis=1)
                v_i_right = tf.reduce_sum(tf.multiply(A_i_right, self.H_right), axis=1)

                v_left.append(v_i_left)
                v_right.append(v_i_right)
                W_1.append(w_1)

            self.V_left  = tf.concat(v_left, axis=-1)
            self.V_right = tf.concat(v_right, axis=-1)

        with tf.name_scope('penalization'):
            self.penalty = 0.0
            if penalty_type == 0:
                for i in range(num_heads):
                    for j in range(i+1, num_heads):
                        self.penalty += tf.nn.relu(1 - tf.square(tf.norm(W_1[i]-W_1[j], ord='fro', axis=[0,1])))

        with tf.name_scope('mlp_layer'):
            self.V = tf.concat([self.V_left, self.V_right, tf.abs(self.V_left-self.V_right),
                                tf.multiply(self.V_left, self.V_right)], axis=-1)

            output = fc_layer(self.V, num_hidden, self.dropout_keep_prob, 'fc_1', scope_reuse=False, regularizer=regularizer)
            # has a shortcut connection
            self.full_out = fc_layer(tf.concat([self.V, output], axis=-1), num_hidden, self.dropout_keep_prob, 'fc_2', scope_reuse=False,
                                     regularizer=regularizer)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_hidden, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b_output")
            self.scores = tf.nn.xw_plus_b(self.full_out, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + mu * self.penalty + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")


    @staticmethod
    def get_length(x):
        x_sign = tf.sign(tf.abs(x))
        length = tf.reduce_sum(x_sign, axis=1)
        return tf.cast(length, tf.int32)


