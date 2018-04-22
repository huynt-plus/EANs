import tensorflow as tf
import time, os
from tensorflow.contrib import layers
from utils import get_batch_index
from tensorflow.contrib import rnn
import numpy as np
import pickle

# Aspect lexicon embedding is concatenated with Aspect embedding
# Context lexicon embedding is concatenated with Context embedding
# Aspect embedding is concatenated with Context embedding

class ALAN_model(object):
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
        self.lex_dim = config.lex_dim
        self.sess = sess
        self.seed = config.seed

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
        print "Building model..."
        # with tf.name_scope('inputs'):
        self.aspects = tf.placeholder(tf.int32, [None, self.max_aspect_len], name="input_aspects")
        self.contexts = tf.placeholder(tf.int32, [None, self.max_context_len], name="input_contexts")
        self.context_lex_embedding = tf.placeholder(tf.float32, [None, self.max_context_len, self.lex_dim], name="input_context_lex")
        self.aspect_lex_embedding = tf.placeholder(tf.float32, [None, self.max_aspect_len, self.lex_dim], name="input_aspect_lex")
        self.labels = tf.placeholder(tf.int32, [None, self.n_class], name="input_labels")
        self.aspect_lens = tf.placeholder(tf.int32, None, name="input_aspect_lens")
        self.context_lens = tf.placeholder(tf.int32, None, name="input_context_lens")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        aspect_inputs = tf.nn.embedding_lookup(self.word2vec, self.aspects)
        aspect_inputs = tf.cast(aspect_inputs, tf.float32)
        # aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=self.dropout_keep_prob)

        context_inputs = tf.nn.embedding_lookup(self.word2vec, self.contexts)
        context_inputs = tf.cast(context_inputs, tf.float32)
        # context_inputs = tf.nn.dropout(context_inputs, keep_prob=self.dropout_keep_prob)

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

        aspect_lex_concat = tf.concat(axis=2, values=[aspect_inputs, self.aspect_lex_embedding])
        aspect_lex_concat = tf.nn.dropout(aspect_lex_concat, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('LSTM'):
            aspect_outputs, aspect_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=aspect_lex_concat,
                sequence_length=self.aspect_lens,
                dtype=tf.float32,
                scope='aspect_lstm'
            )
            # aspect_outputs = self._bidirectional_LSTM_encoder(aspect_inputs, 'aspect_lstm')
            batch_size = tf.shape(aspect_outputs)[0]


            aspect_avg_inputs = tf.reduce_mean(aspect_inputs, 1)
            til_hid = tf.tile(aspect_avg_inputs, [1, self.max_context_len])
            til_hid3dim = tf.reshape(til_hid, [-1, self.max_context_len, self.embedding_dim])
            a_til_concat = tf.concat(axis=2, values=[context_inputs, til_hid3dim])
            a_til_concat_lex = tf.concat(axis=2, values=[a_til_concat, self.context_lex_embedding])
            a_til_concat_lex = tf.nn.dropout(a_til_concat_lex, keep_prob=self.dropout_keep_prob)

            context_outputs, context_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=a_til_concat_lex,
                sequence_length=self.context_lens,
                dtype=tf.float32,
                scope='context_lstm'
            )
            # context_outputs = self._bidirectional_LSTM_encoder(a_til_concat, 'context_lstm')

            # Aspect attention vector and Context embeddings
            self.aspect_att = self._attention_layer(aspect_outputs, 'aspect_att')
            context_att_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_att_outputs_iter = context_att_outputs_iter.unstack(context_outputs)
            aspect_att_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_att_iter = aspect_att_iter.unstack(self.aspect_att)
            context_att_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            context_att = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def _condition(i, context_att_rep, context_att):
                return i < batch_size

            def _body(i, context_att_rep, context_att):
                a = context_att_outputs_iter.read(i)
                b = aspect_att_iter.read(i)
                context_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a, weights['context_score']),
                              tf.reshape(b, [-1, 1])) + biases['context_score']), [1, -1])

                context_att_temp = tf.nn.softmax(context_score)
                context_att = context_att.write(i, context_att_temp)
                context_att_rep = context_att_rep.write(i, tf.matmul(context_att_temp, a))
                return (i + 1, context_att_rep, context_att)

            _, context_att_rep_final, context_att_final = tf.while_loop(cond=_condition, body=_body,
                                                                    loop_vars=(0, context_att_rep, context_att))
            self.context_atts = tf.reshape(context_att_final.stack(), [-1, self.max_context_len], name="att_w")
            self.context_att_reps = tf.reshape(context_att_rep_final.stack(), [-1, self.n_hidden], name="att_rep")

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

        with tf.name_scope('tracking'):
            # Keep track of gradient values (optional)
            summary_loss = tf.summary.scalar('loss', self.loss)
            summary_acc = tf.summary.scalar('acc', self.accuracy)

            self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
            self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])

            timestamp = str(int(time.time()))
            self.out_dir = 'logs/' + str(timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_h' + \
                           str(self.n_hidden) + '_e' + str(self.n_epoch)

            print self.out_dir

            self.train_summary_writer = tf.summary.FileWriter(self.out_dir + '/train', self.sess.graph)
            self.test_summary_writer = tf.summary.FileWriter(self.out_dir + '/test', self.sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
            self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

    def train_step(self, data):
        aspects, contexts, labels, aspect_lens, context_lens, aspect_lex, context_lex = zip(*data)
        cost, cnt = 0.0, 0

        for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, aspect_lex, context_lex, self.batch_size,
                                               True, self.dropout):
            _, loss, step, summary = self.sess.run([self.optimizer, self.loss, self.global_step, self.train_summary_op],
                                                   feed_dict=sample)
            self.train_summary_writer.add_summary(summary, step)
            cost += loss * num
            cnt += num

        _, train_acc = self.eval_step(data)

        return float(cost) / cnt, train_acc

    def eval_step(self, data):
        aspects, contexts, labels, aspect_lens, context_lens, aspect_lex, context_lex = zip(*data)
        cost, acc, cnt = 0.0, 0, 0

        for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, aspect_lex, context_lex, len(data), False,
                                               1.0):
            loss, accuracy, step, summary = self.sess.run(
                [self.loss, self.accuracy, self.global_step, self.test_summary_op], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num
            self.test_summary_writer.add_summary(summary, step)

        return float(cost) / cnt, float(acc) / cnt

    def get_batch_data(self, aspects, contexts, labels, aspect_lens, context_lens, aspect_lex, context_lex, batch_size, is_shuffle,
                       keep_prob):
        aspects = np.array(aspects)
        contexts = np.array(contexts)
        labels = np.array(labels)
        aspect_lens = np.array(aspect_lens)
        context_lens = np.array(context_lens)
        context_lex = np.array(context_lex)
        aspect_lex = np.array(aspect_lex)
        for index in get_batch_index(len(aspects), batch_size, is_shuffle):
            feed_dict = {
                self.aspects: aspects[index],
                self.contexts: contexts[index],
                self.labels: labels[index],
                self.aspect_lens: aspect_lens[index],
                self.context_lens: context_lens[index],
                self.context_lex_embedding: context_lex[index],
                self.aspect_lex_embedding: aspect_lex[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)

    def analysis(self, train_data, test_data):
        timestamp = str(int(time.time()))
        print "Analyzing_" + timestamp
        aspects, contexts, labels, aspect_lens, context_lens, aspect_lex, context_lex = zip(*train_data)
        with open('analysis/train_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens,
                                                   aspect_lex, context_lex, len(train_data), False, 1.0):
                aspect_atts, context_atts, correct_pred = self.sess.run([self.aspect_att, self.context_atts,
                                                                         self.correct_predictions], feed_dict=sample)
                for a, b, c in zip(aspect_atts, context_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))
        print('Finishing analyzing training data')

        aspects, contexts, labels, aspect_lens, context_lens, aspect_lex, context_lex = zip(*test_data)
        with open('analysis/test_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens,
                                                   aspect_lex, context_lex, len(test_data), False, 1.0):
                aspect_atts, context_atts, correct_pred = self.sess.run([self.aspect_att, self.context_atts,
                                                                         self.correct_predictions], feed_dict=sample)
                for a, b, c in zip(aspect_atts, context_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))
        print('Finishing analyzing testing data')

    def train(self, sess, train_data, test_data):

        if self.seed is not None:
            np.random.seed(self.seed)

        saver = tf.train.Saver(tf.global_variables())

        print('Training...')
        self.sess.run(tf.global_variables_initializer())

        max_acc, current_step = 0., -1
        for i in range(self.n_epoch):
            train_loss, train_acc = self.train_step(train_data)
            test_loss, test_acc = self.eval_step(test_data)
            if test_acc > max_acc:
                max_acc = test_acc
                current_step = tf.train.global_step(sess, self.global_step)
                saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
            print('epoch %s: train-loss=%.6f; train-acc=%.6f; train-loss=%.6f; test-acc=%.6f; best-acc=%.6f' % (
                str(i), train_loss, train_acc, test_loss, test_acc, max_acc))
        # saver.save(self.sess, self.checkpoint_prefix)
        print('The max accuracy of testing results is %s of step %s' % (max_acc, current_step))

        print('Analyzing ...')
        saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir))
        self.analysis(train_data, test_data)


def get_atts(test_data, checkpoint_dir):
    aspects, contexts, labels, aspect_lens, context_lens, aspect_lex, context_lex = zip(*test_data)
    # y_test = np.argmax(labels, axis=1)
    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    # checkpoint_file = checkpoint_dir
    print("checkpoint_file: ", checkpoint_file)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_aspects = graph.get_operation_by_name("input_aspects").outputs[0]
            input_contexts = graph.get_operation_by_name("input_contexts").outputs[0]
            input_labels = graph.get_operation_by_name("input_labels").outputs[0]
            input_aspect_lens = graph.get_operation_by_name("input_aspect_lens").outputs[0]
            input_context_lens = graph.get_operation_by_name("input_context_lens").outputs[0]
            input_context_lex = graph.get_operation_by_name("input_context_lex").outputs[0]
            input_aspect_lex = graph.get_operation_by_name("input_aspect_lex").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to get attention weights
            attention_weights = graph.get_operation_by_name("LSTM/att_w").outputs[0]

            all_attentions = []
            def get_batch(aspects, contexts, labels, aspect_lens, context_lens, aspect_lex, context_lex,
                          batch_size, is_shuffle, keep_prob):
                aspects = np.array(aspects)
                contexts = np.array(contexts)
                labels = np.array(labels)
                aspect_lens = np.array(aspect_lens)
                context_lens = np.array(context_lens)
                context_lex = np.array(context_lex)
                aspect_lex = np.array(aspect_lex)
                for index in get_batch_index(len(aspects), batch_size, is_shuffle):
                    feed_dict = {
                        input_aspects: aspects[index],
                        input_contexts: contexts[index],
                        input_labels: labels[index],
                        input_aspect_lens: aspect_lens[index],
                        input_context_lens: context_lens[index],
                        input_context_lex: context_lex[index],
                        input_aspect_lex: aspect_lex[index],
                        dropout_keep_prob: keep_prob
                    }
                    yield feed_dict, len(index)
            for sample, num in get_batch(aspects, contexts, labels, aspect_lens, context_lens, aspect_lex,
                                         context_lex, len(test_data), False, 1.0):
                attentions = sess.run(attention_weights, feed_dict=sample)
                all_attentions.append(attentions)

    with open('atts.pickle', 'wb') as g:
        pickle.dump(all_attentions, g)