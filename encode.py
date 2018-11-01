import pickle as pk

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding
from keras.utils import plot_model

from sklearn.cluster import KMeans

from nn_arch import dnn_cache, cnn_cache, rnn_cache

from util import flat_read, map_item


def define_model(name, embed_mat, seq_len):
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


def load_model(name, embed_mat, seq_len):
    model = define_model(name, embed_mat, seq_len)
    model.load_weights(map_item(name, paths), by_name=True)
    return model


seq_len = 30
max_core = 5

path_embed = 'feat/embed.pkl'
path_sent = 'feat/sent_train.pkl'
path_train = 'data/train.csv'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
labels = flat_read(path_train, 'label')

funcs = {'dnn': dnn_cache,
         'cnn': cnn_cache,
         'rnn': rnn_cache}

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5',
         'dnn_cache': 'cache/dnn.h5',
         'cnn_cache': 'cache/cnn.h5',
         'rnn_cache': 'cache/rnn.h5',
         'dnn_plot': 'model/plot/dnn_cache.png',
         'cnn_plot': 'model/plot/cnn_cache.png',
         'rnn_plot': 'model/plot/rnn_cache.png'}

models = {'dnn': load_model('dnn', embed_mat, seq_len),
          'cnn': load_model('cnn', embed_mat, seq_len),
          'rnn': load_model('rnn', embed_mat, seq_len)}


def label2ind(labels, path_label_ind):
    label_set = sorted(list(set(labels)))
    label_inds = dict()
    for i in range(len(label_set)):
        label_inds[label_set[i]] = i
    inds = [label_inds[label] for label in labels]
    with open(path_label_ind, 'wb') as f:
        pk.dump(label_inds, f)
    return label_inds, np.array(inds)


def cluster(sents, inds, ind_labels):
    label_num = len(ind_labels)
    cores = list()
    labels = list()
    for i in range(label_num):
        ind_args = np.where(inds == i)
        match_sents = sents[ind_args]
        core_num = min(len(match_sents), max_core)
        model = KMeans(n_clusters=core_num, n_init=10, max_iter=100)
        model.fit(match_sents)
        cores.extend(model.cluster_centers_.tolist())
        labels.extend([ind_labels[i]] * core_num)
    return np.array(cores), labels


def cache(sents, labels, path_label_ind):
    label_inds, inds = label2ind(labels, path_label_ind)
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    for name, model in models.items():
        path_cache = map_item(name + '_cache', paths)
        encode_sents = model.predict(sents)
        cores_labels = cluster(encode_sents, inds, ind_labels)
        with open(path_cache, 'wb') as f:
            pk.dump(cores_labels, f)


if __name__ == '__main__':
    path_label_ind = 'feat/label_ind.pkl'
    cache(sents, labels, path_label_ind)
