import pickle as pk

import numpy as np

from collections import Counter

from keras.models import Model
from keras.layers import Input

from keras.preprocessing.sequence import pad_sequences

from preprocess import clean

from nn_arch import match

from encode import load_encode

from util import map_item


def define_match(encode_len):
    input1 = Input(shape=(encode_len,))
    input2 = Input(shape=(encode_len,))
    output = match(input1, input2)
    return Model([input1, input2], output)


def load_match(name, encode_len):
    feat_len = encode_len * 2 if name == 'rnn' else encode_len
    model = define_match(feat_len)
    model.load_weights(map_item(name, paths), by_name=True)
    return model


def load_cache(path_cache):
    with open(path_cache, 'rb') as f:
        core_sents = pk.load(f)
    return core_sents


seq_len = 30
encode_len = 200

path_embed = 'feat/embed.pkl'
path_word2ind = 'model/word2ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5',
         'dnn_cache': 'cache/dnn.pkl',
         'cnn_cache': 'cache/cnn.pkl',
         'rnn_cache': 'cache/rnn.pkl'}

caches = {'dnn': load_cache(map_item('dnn_cache', paths)),
          'cnn': load_cache(map_item('cnn_cache', paths)),
          'rnn': load_cache(map_item('rnn_cache', paths))}

models = {'dnn_encode': load_encode('dnn', embed_mat, seq_len),
          'cnn_encode': load_encode('cnn', embed_mat, seq_len),
          'rnn_encode': load_encode('rnn', embed_mat, seq_len),
          'dnn_match': load_match('dnn', encode_len),
          'cnn_match': load_match('cnn', encode_len),
          'rnn_match': load_match('rnn', encode_len)}


def predict(text, name, vote):
    text = clean(text)
    core_sents, core_labels = map_item(name, caches)
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    encode = map_item(name + '_encode', models)
    encode_seq = encode.predict([pad_seq])
    encode_mat = np.repeat(encode_seq, len(core_sents), axis=0)
    model = map_item(name + '_match', models)
    probs = model.predict([encode_mat, core_sents])
    probs = np.reshape(probs, (1, -1))[0]
    max_probs = sorted(probs, reverse=True)[:vote]
    max_inds = np.argsort(-probs)[:vote]
    max_preds = [core_labels[ind] for ind in max_inds]
    if __name__ == '__main__':
        formats = list()
        for pred, prob in zip(max_preds, max_probs):
            formats.append('{} {:.3f}'.format(pred, prob))
        return ', '.join(formats)
    else:
        pairs = Counter(max_preds)
        return pairs.most_common()[0][0]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('dnn: %s' % predict(text, 'dnn', vote=5))
        print('cnn: %s' % predict(text, 'cnn', vote=5))
        print('rnn: %s' % predict(text, 'rnn', vote=5))
