import tensorflow as tf
from utils import get_data_info, load_data, load_word_embeddings, get_lex_file_list, load_bin_vec
from ALAN_model import ALAN_model
from loader.lex_helper import LexHelper

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
    tf.app.flags.DEFINE_string('embedding', 'glove', 'embedding file name')
    tf.app.flags.DEFINE_string('train_fname', './data/restaurant/train.txt', 'training file name')
    tf.app.flags.DEFINE_string('test_fname', './data/restaurant/test.txt', 'testing file name')
    tf.app.flags.DEFINE_string('data_info', './data/data_info.txt', 'the file saving data information')
    tf.app.flags.DEFINE_string('train_data', './data/train_data.txt', 'the file saving training data')
    tf.app.flags.DEFINE_string('test_data', './data/test_data.txt', 'the file saving testing data')
    tf.app.flags.DEFINE_string('lex_path', './resrc/lexicons/lex_config.txt', 'the file saving testing data')
    tf.app.flags.DEFINE_integer('seed', 12345, 'The random seed')

    print('Loading data info ...')
    FLAGS.word2id, FLAGS.max_aspect_len, FLAGS.max_context_len = get_data_info(FLAGS.train_fname, FLAGS.test_fname,
                                                                               FLAGS.data_info, FLAGS.pre_processed)

    print('Loading training data and testing data ...')

    train_aspects, train_contexts, train_labels, train_aspect_lens, \
    train_context_lens, train_aspect_texts, train_context_texts = load_data(FLAGS.train_fname, FLAGS.word2id,
                                                                            FLAGS.max_aspect_len, FLAGS.max_context_len,
                                                                            FLAGS.train_data, FLAGS.pre_processed)


    test_aspects, test_contexts, test_labels, test_aspect_lens, \
    test_context_lens, test_aspect_texts, test_context_texts = load_data(FLAGS.test_fname, FLAGS.word2id,
                                                                         FLAGS.max_aspect_len, FLAGS.max_context_len,
                                                                         FLAGS.test_data, FLAGS.pre_processed)

    print('Loading pre-trained word vectors ...')
    if FLAGS.embedding == 'glove':
        FLAGS.word2vec = load_word_embeddings(FLAGS.embedding_fname, FLAGS.embedding_dim, FLAGS.word2id)
    else:
        FLAGS.word2vec = load_bin_vec(FLAGS.embedding_fname, FLAGS.embedding_dim, FLAGS.word2id)

    # Building lexicon embedding
    lex_list = get_lex_file_list(FLAGS.lex_path)
    train = zip(train_aspect_texts, train_context_texts)
    test = zip(test_aspect_texts, test_context_texts)
    lex = LexHelper(lex_list, train, test,
                    max_aspect_len=FLAGS.max_aspect_len, max_context_len=FLAGS.max_context_len)
    train_context_lex, train_aspect_lex, test_context_lex, test_aspect_lex, FLAGS.lex_dim = lex.build_lex_embeddings()

    train_data = zip(train_aspects, train_contexts, train_labels, train_aspect_lens,
                     train_context_lens, train_aspect_lex, train_context_lex)
    test_data = zip(test_aspects, test_contexts, test_labels, test_aspect_lens,
                    test_context_lens, test_aspect_lex, test_context_lex)

    with tf.Session() as sess:
        model = ALAN_model(FLAGS, sess)
        model.build_model()
        model.train(sess, train_data, test_data)
    print ("model=ALAN, model_path=%s, embedding=%s, batch-size=%s, n_epoch=%s, n_hidden=%s, data=%s" % (
        model.out_dir, FLAGS.embedding, FLAGS.batch_size, FLAGS.n_epoch, FLAGS.n_hidden, FLAGS.train_fname))