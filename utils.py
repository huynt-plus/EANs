import os
import ast
import spacy
import numpy as np
from errno import ENOENT
from collections import Counter
from nltk.corpus import stopwords

nlp = spacy.load("en")

def get_data_info(train_fname, test_fname, save_fname, pre_processed):
    word2id, max_aspect_len, max_context_len = {}, 0, 0
    word2id[0] = '<pad>'
    stop_words = ["'", ",", ".", "!", "?"]
    stop = stopwords.words('english')
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        with open(save_fname, 'r') as f:
            for line in f:
                content = line.strip().split()
                if len(content) == 3:
                    max_aspect_len = int(content[1])
                    max_context_len = int(content[2])
                else:
                    word2id[content[0]] = int(content[1])
    else:
        if not os.path.isfile(train_fname):
            raise IOError(ENOENT, 'Not a file', train_fname)
        if not os.path.isfile(test_fname):
            raise IOError(ENOENT, 'Not a file', test_fname)

        words = []

        lines = open(train_fname, 'r').readlines()
        for i in range(0, len(lines), 3):
            sptoks = nlp(lines[i].decode('utf8'))
            words.extend([sp.text.lower() for sp in sptoks if sp.text.lower() not in stop])
            if len(sptoks) > max_context_len:
                max_context_len = len(sptoks)
            sptoks = nlp(lines[i + 1].decode('utf8'))
            if len(sptoks) > max_aspect_len:
                max_aspect_len = len(sptoks)
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word and '\n' not in word and 'aspect_term' not in word and word not in stop_words:
                word2id[word] = len(word2id)

        lines = open(test_fname, 'r').readlines()
        for i in range(0, len(lines), 3):
            sptoks = nlp(lines[i].decode('utf8'))
            words.extend([sp.text.lower() for sp in sptoks if sp.text.lower() not in stop])
            if len(sptoks) > max_context_len:
                max_context_len = len(sptoks)
            sptoks = nlp(lines[i + 1].decode('utf8'))
            if len(sptoks) > max_aspect_len:
                max_aspect_len = len(sptoks)
        word_count = Counter(words).most_common()
        for word, _ in word_count:
            if word not in word2id and ' ' not in word and '\n' not in word and 'aspect_term' not in word and word not in stop_words:
                word2id[word] = len(word2id)

        with open(save_fname, 'w') as f:
            f.write('length %s %s\n' % (max_aspect_len, max_context_len))
            for key, value in word2id.items():
                f.write('%s %s\n' % (key, value))

    print('There are %s words in the dataset, the max length of aspect is %s, and the max length of context is %s' % (
    len(word2id), max_aspect_len, max_context_len))
    return word2id, max_aspect_len, max_context_len


def read_data(fname, word2id, max_aspect_len, max_context_len, save_fname, pre_processed):
    aspects, contexts, labels, aspect_lens, context_lens = list(), list(), list(), list(), list()
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        lines = open(save_fname, 'r').readlines()
        for i in range(0, len(lines), 5):
            aspects.append(ast.literal_eval(lines[i]))
            contexts.append(ast.literal_eval(lines[i + 1]))
            labels.append(ast.literal_eval(lines[i + 2]))
            aspect_lens.append(ast.literal_eval(lines[i + 3]))
            context_lens.append(ast.literal_eval(lines[i + 4]))
    else:
        if not os.path.isfile(fname):
            raise IOError(ENOENT, 'Not a file', fname)

        lines = open(fname, 'r').readlines()
        with open(save_fname, 'w') as f:
            for i in range(0, len(lines), 3):
                polarity = lines[i + 2].split()[0]
                if polarity == 'conflict':
                    continue

                context_sptoks = nlp(lines[i].decode('utf8'))
                context = []
                for sptok in context_sptoks:
                    if sptok.text.lower() in word2id:
                        context.append(word2id[sptok.text.lower()])

                aspect_sptoks = nlp(lines[i + 1].decode('utf8'))
                aspect = []
                for aspect_sptok in aspect_sptoks:
                    if aspect_sptok.text.lower() in word2id:
                        aspect.append(word2id[aspect_sptok.text.lower()])

                aspects.append(aspect + [0] * (max_aspect_len - len(aspect)))
                f.write("%s\n" % aspects[-1])
                contexts.append(context + [0] * (max_context_len - len(context)))
                f.write("%s\n" % contexts[-1])
                if polarity == 'negative':
                    labels.append([1, 0, 0])
                elif polarity == 'neutral':
                    labels.append([0, 1, 0])
                elif polarity == 'positive':
                    labels.append([0, 0, 1])
                f.write("%s\n" % labels[-1])
                aspect_lens.append(len(aspect_sptoks))
                f.write("%s\n" % aspect_lens[-1])
                context_lens.append(len(context_sptoks))
                f.write("%s\n" % context_lens[-1])

    print("Read %s examples from %s" % (len(aspects), fname))
    return np.asarray(aspects), np.asarray(contexts), np.asarray(labels), np.asarray(aspect_lens), np.asarray(
        context_lens)


def load_word_embeddings(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)
    print ('Loading Glove...')
    word2vec = np.random.uniform(-0.01, 0.01, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(fname, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            content = line.strip().split()
            if content[0] in word2id:
                try:
                    word2vec[word2id[content[0]]] = np.array(list(map(float, content[1:])))
                    oov = oov - 1
                except:
                    pass
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec
def load_bin_vec(fname, embedding_dim, word2id):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)
    print ('Loading GoogleW2V...')
    word2vec = np.random.uniform(-0.01, 0.01, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in word2id:
                word2vec[word2id[word]] = np.fromstring(f.read(binary_len), dtype='float32')
                oov = oov - 1
            else:
                f.read(binary_len)
    print('There are %s words in vocabulary and %s words out of vocabulary' % (len(word2id) - oov, oov))
    return word2vec

def get_batch_index(length, batch_size, is_shuffle=True):
    index = list(range(length))
    if is_shuffle:
        np.random.shuffle(index)
    for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
        yield index[i * batch_size:(i + 1) * batch_size]

def load_data(fname, word2id, max_aspect_len, max_context_len, save_fname, pre_processed):
    aspects, contexts, labels, aspect_lens, context_lens = list(), list(), list(), list(), list()
    aspect_texts, context_texts = list(), list()
    if pre_processed:
        if not os.path.isfile(save_fname):
            raise IOError(ENOENT, 'Not a file', save_fname)
        lines = open(save_fname, 'r').readlines()
        for i in range(0, len(lines), 5):
            aspects.append(ast.literal_eval(lines[i]))
            contexts.append(ast.literal_eval(lines[i + 1]))
            labels.append(ast.literal_eval(lines[i + 2]))
            aspect_lens.append(ast.literal_eval(lines[i + 3]))
            context_lens.append(ast.literal_eval(lines[i + 4]))
    else:
        if not os.path.isfile(fname):
            raise IOError(ENOENT, 'Not a file', fname)

        lines = open(fname, 'r').readlines()
        with open(save_fname, 'w') as f:
            for i in range(0, len(lines), 3):
                polarity = lines[i + 2].split()[0]
                if polarity == 'conflict':
                    continue

                context_sptoks = nlp(lines[i].decode('utf8'))
                context = []
                context_words = []
                for sptok in context_sptoks:
                    if sptok.text.lower() in word2id:
                        context.append(word2id[sptok.text.lower()])
                        context_words.append(sptok.text.lower())

                aspect_sptoks = nlp(lines[i + 1].decode('utf8'))
                aspect = []
                aspect_words = []
                for aspect_sptok in aspect_sptoks:
                    if aspect_sptok.text.lower() in word2id:
                        aspect.append(word2id[aspect_sptok.text.lower()])
                        aspect_words.append(aspect_sptok.text.lower())

                aspects.append(aspect + [0] * (max_aspect_len - len(aspect)))
                aspect_texts.append(aspect_words)
                f.write("%s\n" % aspects[-1])
                contexts.append(context + [0] * (max_context_len - len(context)))
                context_texts.append(context_words)
                f.write("%s\n" % contexts[-1])
                if polarity == 'negative':
                    labels.append([1, 0, 0])
                elif polarity == 'neutral':
                    labels.append([0, 1, 0])
                elif polarity == 'positive':
                    labels.append([0, 0, 1])
                f.write("%s\n" % labels[-1])
                aspect_lens.append(len(aspect_sptoks))
                f.write("%s\n" % aspect_lens[-1])
                context_lens.append(len(context_sptoks))
                f.write("%s\n" % context_lens[-1])

    print("Read %s examples from %s" % (len(aspects), fname))
    return np.asarray(aspects), np.asarray(contexts), np.asarray(labels), np.asarray(aspect_lens), np.asarray(
        context_lens), np.asarray(aspect_texts), np.asarray(context_texts)

def get_lex_file_list(lex_file_path):
    lex_file_list = []
    with open(lex_file_path, 'rt') as handle:
        for line in handle.readlines():
            path = line.strip()

            if os.path.isfile(path):
                lex_file_list.append(path)
            else:
                print ('wrong file name(s) in the lex_config.txt\n%s' % path)
                return None

    return lex_file_list