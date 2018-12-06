import pickle as pk

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding
from keras.utils import plot_model

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


def split(sents, labels, path_label):
    label_set = sorted(list(set(labels)))
    labels = np.array(labels)
    sent_mat, core_labels, core_nums = list(), list(), list()
    for match_label in label_set:
        match_inds = np.where(labels == match_label)
        match_sents = sents[match_inds]
        sent_mat.append(match_sents)
        core_num = min(len(match_sents), max_core)
        core_nums.append(core_num)
        core_labels.extend([match_label] * core_num)
    with open(path_label, 'wb') as f:
        pk.dump(core_labels, f)
    return sent_mat, core_nums


def cluster(encode_mat, core_nums):
    core_sents = list()
    for sents, core_num in zip(encode_mat, core_nums):
        model = KMeans(n_clusters=core_num, n_init=10, max_iter=100)
        model.fit(sents)
        core_sents.extend(model.cluster_centers_.tolist())
    return np.array(core_sents)


def cache(path_sent, path_train, path_label):
    with open(path_sent, 'rb') as f:
        sents = pk.load(f)
    labels = flat_read(path_train, 'label')
    sent_mat, core_nums = split(sents, labels, path_label)
    for name, model in models.items():
        encode_mat = list()
        for sents in sent_mat:
            encode_mat.append(model.predict(sents))
        core_sents = cluster(encode_mat, core_nums)
        path_cache = map_item(name + '_cache', paths)
        with open(path_cache, 'wb') as f:
            pk.dump(core_sents, f)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    path_sent = 'feat/sent_train.pkl'
    path_label = 'cache/label.pkl'
    cache(path_sent, path_train, path_label)
