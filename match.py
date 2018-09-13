import pickle as pk

import re

import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from util import load_word_re, load_type_re, load_word_pair, word_replace, flat_read
from util import map_path, map_model


seq_len = 30

path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_word_pair(path_homo)
syno_dict = load_word_pair(path_syno)

path_train = 'data/train.csv'
path_sent = 'feat/sent_train.pkl'
path_label = 'feat/label_train.pkl'
path_word2ind = 'model/word2ind.pkl'
texts = flat_read(path_train, 'text')
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}

models = {'dnn': load_model(map_path('dnn', paths)),
          'cnn': load_model(map_path('cnn', paths)),
          'rnn': load_model(map_path('rnn', paths))}


def predict(text, name):
    text = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    text = word_replace(text, syno_dict)
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_model(name, models)
    pad_mat = np.repeat(pad_seq, len(sents), axis=0)
    probs = model.predict([pad_mat, sents])
    probs = np.reshape(probs, (1, -1))[0]
    max_probs = sorted(probs, reverse=True)[:3]
    max_inds = np.argsort(-probs)[:3]
    max_preds = [labels[ind] for ind in max_inds]
    max_texts = [texts[ind] for ind in max_inds]
    formats = list()
    for pred, prob, text in zip(max_preds, max_probs, max_texts):
        formats.append('{} {:.3f} {}'.format(pred, prob, text))
    return ', '.join(formats)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('dnn: %s' % predict(text, 'dnn'))
        print('cnn: %s' % predict(text, 'cnn'))
        print('rnn: %s' % predict(text, 'rnn'))
