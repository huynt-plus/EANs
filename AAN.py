import tensorflow as tf
from utils import get_data_info, read_data, load_word_embeddings
from AAN_model import AAN_model

if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
    tf.app.flags.DEFINE_integer('batch_size', 100, 'number of example per batch')
    tf.app.flags.DEFINE_integer('n_epoch', 300, 'number of epoch')
    tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
    tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
    tf.app.flags.DEFINE_integer('pre_processed', 0, 'Whether the data is pre-processed')
    tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
    tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

    tf.app.flags.DEFINE_string('embedding_fname', './vec/glove.42B.300d.txt', 'embedding file name')
    tf.app.flags.DEFINE_string('embedding', 'glove', 'oov')
    tf.app.flags.DEFINE_string('train_fname', './data/laptop/train.txt', 'training file name')
    tf.app.flags.DEFINE_string('test_fname', './data/laptop/test.txt', 'testing file name')
    tf.app.flags.DEFINE_string('data_info', './data/data_info.txt', 'the file saving data information')
    tf.app.flags.DEFINE_string('train_data', './data/train_data.txt', 'the file saving training data')
    tf.app.flags.DEFINE_string('test_data', './data/test_data.txt', 'the file saving testing data')

    print('Loading data info ...')
    FLAGS.word2id, FLAGS.max_aspect_len, FLAGS.max_context_len = get_data_info(FLAGS.train_fname, FLAGS.test_fname,
                                                                               FLAGS.data_info, FLAGS.pre_processed)

    print('Loading training data and testing data ...')
    train_data = read_data(FLAGS.train_fname, FLAGS.word2id, FLAGS.max_aspect_len, FLAGS.max_context_len,
                           FLAGS.train_data, FLAGS.pre_processed)
    test_data = read_data(FLAGS.test_fname, FLAGS.word2id, FLAGS.max_aspect_len, FLAGS.max_context_len, FLAGS.test_data,
                          FLAGS.pre_processed)

    print('Loading pre-trained word vectors ...')
    FLAGS.word2vec = load_word_embeddings(FLAGS.embedding_fname, FLAGS.embedding_dim, FLAGS.word2id)

    with tf.Session() as sess:
        model = AAN_model(FLAGS, sess)
        model.build_model()
        model.train(train_data, test_data)

    print "model=AAN, embedding=%s, batch-size=%s, n_epoch=%s, n_hidden=%s, data=%s" % (
        FLAGS.embedding, FLAGS.batch_size, FLAGS.n_epoch, FLAGS.n_hidden, FLAGS.train_fname)