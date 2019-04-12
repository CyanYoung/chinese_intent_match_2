import pickle as pk

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding
from keras.utils import plot_model

from sklearn.ensemble import IsolationForest

from sklearn.cluster import KMeans

from nn_arch import dnn_encode, cnn_encode, rnn_encode

from util import flat_read, map_item


def define_encode(name, embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len, name='embed')
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    output = func(embed_input)
    model = Model(input, output)
    if __name__ == '__main__':
        plot_model(model, map_item(name + '_plot', paths), show_shapes=True)
    return model


def load_encode(name, embed_mat, seq_len):
    model = define_encode(name, embed_mat, seq_len)
    model.load_weights(map_item(name, paths), by_name=True)
    return model


seq_len = 30
max_core = 5

path_embed = 'feat/embed.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)

path_sent = 'feat/sent_train.pkl'
path_train = 'data/train.csv'
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
labels = flat_read(path_train, 'label')

funcs = {'dnn': dnn_encode,
         'cnn': cnn_encode,
         'rnn': rnn_encode}

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5',
         'dnn_cache': 'cache/dnn.pkl',
         'cnn_cache': 'cache/cnn.pkl',
         'rnn_cache': 'cache/rnn.pkl',
         'dnn_plot': 'model/plot/dnn_encode.png',
         'cnn_plot': 'model/plot/cnn_encode.png',
         'rnn_plot': 'model/plot/rnn_encode.png'}

models = {'dnn': load_encode('dnn', embed_mat, seq_len),
          'cnn': load_encode('cnn', embed_mat, seq_len),
          'rnn': load_encode('rnn', embed_mat, seq_len)}


def split(sents, labels):
    label_set = sorted(list(set(labels)))
    labels = np.array(labels)
    sent_mat, label_mat = list(), list()
    for match_label in label_set:
        match_inds = np.where(labels == match_label)
        match_sents = sents[match_inds]
        sent_mat.append(match_sents)
        match_labels = [match_label] * len(match_sents)
        label_mat.append(match_labels)
    return sent_mat, label_mat


def clean(encode_mat, label_mat):
    for i in range(len(encode_mat)):
        model = IsolationForest(n_estimators=100, contamination=0.1)
        model.fit(encode_mat[i])
        flags = model.predict(encode_mat[i])
        count = np.sum(flags > 0)
        if count > max_core:
            inds = np.where(flags < 0)
            encode_mat[i] = np.delete(encode_mat[i], inds, axis=0)
    return encode_mat, label_mat


def merge(encode_mat, label_mat):
    core_sents, core_labels = list(), list()
    for sents, labels in zip(encode_mat, label_mat):
        core_num = min(len(sents), max_core)
        model = KMeans(n_clusters=core_num, n_init=10, max_iter=100)
        model.fit(sents)
        core_sents.extend(model.cluster_centers_.tolist())
        core_labels.extend([labels[0]] * core_num)
    return np.array(core_sents), np.array(core_labels)


def cache(sents, labels):
    sent_mat, label_mat = split(sents, labels)
    for name, model in models.items():
        encode_mat = list()
        for sents in sent_mat:
            encode_mat.append(model.predict(sents))
        encode_mat, label_mat = clean(encode_mat, label_mat)
        core_sents, core_labels = merge(encode_mat, label_mat)
        path_cache = map_item(name + '_cache', paths)
        with open(path_cache, 'wb') as f:
            pk.dump((core_sents, core_labels), f)


if __name__ == '__main__':
    cache(sents, labels)
