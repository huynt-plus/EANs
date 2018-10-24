import tensorflow as tf
import time, os
from tensorflow.contrib import layers
from utils import get_batch_index
from tensorflow.contrib import rnn
from tensorflow.python.ops import math_ops
# Aspect embedding and Context embedding are treated separately

class AN_model(object):
    def __init__(self, config, sess):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout

        self.word2id = config.word2id
        self.max_aspect_len = config.max_aspect_len
        self.max_context_len = config.max_context_len
        self.word2vec = config.word2vec
        self.sess = sess

    def _attention_layer(self, inputs, name):
        with tf.variable_scope(name):
            u_context = tf.Variable(tf.truncated_normal([self.n_hidden * 2]), name='u_context')
            h = layers.fully_connected(inputs, self.n_hidden * 2, activation_fn=tf.nn.tanh)
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            att_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return att_output

    def _bidirectional_GRU_encoder(self, inputs, name):
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.n_hidden)
            GRU_cell_bw = rnn.GRUCell(self.n_hidden)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=self._length(inputs),
                                                                                 dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def _bidirectional_LSTM_encoder(self, inputs, name):
        with tf.variable_scope(name):
            LSTM_cell_fw = rnn.LSTMCell(self.n_hidden)
            LSTM_cell_bw = rnn.LSTMCell(self.n_hidden)
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=LSTM_cell_fw,
                                                                                 cell_bw=LSTM_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=self._length(inputs),
                                                                                 dtype=tf.float32)
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def _length(self, sequences):
        used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
        seq_len = tf.reduce_sum(used, reduction_indices=1)
        return tf.cast(seq_len, tf.int32)

    def build_model(self):

        with tf.name_scope('inputs'):
            self.aspects = tf.placeholder(tf.int32, [None, self.max_aspect_len])
            self.contexts = tf.placeholder(tf.int32, [None, self.max_context_len])
            self.labels = tf.placeholder(tf.int32, [None, self.n_class])
            self.aspect_lens = tf.placeholder(tf.int32, None)
            self.context_lens = tf.placeholder(tf.int32, None)
            self.dropout_keep_prob = tf.placeholder(tf.float32)

            aspect_inputs = tf.nn.embedding_lookup(self.word2vec, self.aspects)
            aspect_inputs = tf.cast(aspect_inputs, tf.float32)
            aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=self.dropout_keep_prob)

            context_inputs = tf.nn.embedding_lookup(self.word2vec, self.contexts)
            context_inputs = tf.cast(context_inputs, tf.float32)
            context_inputs = tf.nn.dropout(context_inputs, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('weights'):
            weights = {
                'aspect_score': tf.get_variable(
                    name='W_a',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='W_c',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.n_hidden * 2, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('biases'):
            biases = {
                'aspect_score': tf.get_variable(
                    name='B_a',
                    shape=[self.max_aspect_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='B_c',
                    shape=[self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('LSTM'):
            aspect_outputs, aspect_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=aspect_inputs,
                sequence_length=self.aspect_lens,
                dtype=tf.float32,
                scope='aspect_lstm'
            )
            # aspect_outputs = self._bidirectional_LSTM_encoder(aspect_inputs, 'aspect_lstm')

            batch_size = tf.shape(aspect_outputs)[0]

            context_outputs, context_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=context_inputs,
                sequence_length=self.context_lens,
                dtype=tf.float32,
                scope='context_lstm'
            )
            # context_outputs = self._bidirectional_LSTM_encoder(a_til_concat, 'context_lstm')

            # Aspect attention vector and Context embeddings
            aspect_att = self._attention_layer(aspect_outputs, 'aspect_att')
            context_att_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_att_outputs_iter = context_att_outputs_iter.unstack(context_outputs)
            aspect_att_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_att_iter = aspect_att_iter.unstack(aspect_att)
            context_att_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            context_att = tf.TensorArray(size=batch_size, dtype=tf.float32)
            context_lens_iter = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
            context_lens_iter = context_lens_iter.unstack(self.context_lens)
            def _condition(i, context_att_rep, context_att):
                return i < batch_size

            def _body(i, context_att_rep, context_att):
                a = context_att_outputs_iter.read(i)
                b = aspect_att_iter.read(i)
                l = math_ops.to_int32(context_lens_iter.read(i))
                context_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a, weights['context_score']),
                              tf.reshape(b, [-1, 1])) + biases['context_score']), [1, -1])
                context_att_temp = tf.concat([tf.nn.softmax(tf.slice(context_score, [0, 0], [1, l])),
                                              tf.zeros([1, self.max_context_len - l])], 1)
                # context_att_temp = tf.nn.softmax(context_score)
                context_att = context_att.write(i, context_att_temp)
                context_att_rep = context_att_rep.write(i, tf.matmul(context_att_temp, a))
                return (i + 1, context_att_rep, context_att)

            _, context_att_rep_final, context_att_final = tf.while_loop(cond=_condition, body=_body,
                                                                    loop_vars=(0, context_att_rep, context_att))
            self.context_atts = tf.reshape(context_att_final.stack(), [-1, self.max_context_len])
            self.context_att_reps = tf.reshape(context_att_rep_final.stack(), [-1, self.n_hidden])

            aspect_lstm_avg = tf.reduce_mean(aspect_outputs, 1)
            self.reps = tf.concat([aspect_lstm_avg, self.context_att_reps], 1)
            self.scores = tf.matmul(self.reps, weights['softmax']) + biases['softmax']

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.labels))
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                               global_step=self.global_step)

        with tf.name_scope('accuracy'):
            self.correct_predictions = tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_predictions, tf.int32))

    def train_step(self, data):
        aspects, contexts, labels, aspect_lens, context_lens = data
        cost, cnt = 0.0, 0

        for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, self.batch_size,
                                               True, self.dropout):
            _, loss, step, summary = self.sess.run([self.optimizer, self.loss, self.global_step, self.train_summary_op],
                                                   feed_dict=sample)
            self.train_summary_writer.add_summary(summary, step)
            cost += loss * num
            cnt += num

        _, train_acc = self.eval_step(data)

        return float(cost) / cnt, train_acc

    def eval_step(self, data):
        aspects, contexts, labels, aspect_lens, context_lens = data
        cost, acc, cnt = 0.0, 0, 0

        for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(data), False,
                                               1.0):
            loss, accuracy, step, summary = self.sess.run(
                [self.loss, self.accuracy, self.global_step, self.test_summary_op], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num
            self.test_summary_writer.add_summary(summary, step)

        return float(cost) / cnt, float(acc) / cnt

    def get_batch_data(self, aspects, contexts, labels, aspect_lens, context_lens, batch_size, is_shuffle, keep_prob):
        for index in get_batch_index(len(aspects), batch_size, is_shuffle):
            feed_dict = {
                self.aspects: aspects[index],
                self.contexts: contexts[index],
                self.labels: labels[index],
                self.aspect_lens: aspect_lens[index],
                self.context_lens: context_lens[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)

    def train(self, train_data, test_data):
        # Keep track of gradient values (optional)
        summary_loss = tf.summary.scalar('loss', self.loss)
        summary_acc = tf.summary.scalar('acc', self.accuracy)

        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])

        timestamp = str(int(time.time()))
        out_dir = 'logs/' + str(timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(
            self.l2_reg)
        self.train_summary_writer = tf.summary.FileWriter(out_dir + '/train', self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(out_dir + '/test', self.sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        print('Training...')
        self.sess.run(tf.global_variables_initializer())

        max_acc, step = 0., -1
        for i in range(self.n_epoch):
            train_loss, train_acc = self.train_step(train_data)
            test_loss, test_acc = self.eval_step(test_data)
            if test_acc > max_acc:
                max_acc = test_acc
                step = i
                saver.save(self.sess, checkpoint_prefix, global_step=step)
            print('epoch %s: train-loss=%.6f; train-acc=%.6f; train-loss=%.6f; test-acc=%.6f; best-acc=%.6f' % (
                str(i), train_loss, train_acc, test_loss, test_acc, max_acc))
        saver.save(self.sess, checkpoint_prefix)
        print('The max accuracy of testing results is %s of step %s' % (max_acc, step))