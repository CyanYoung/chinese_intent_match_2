import pickle as pk

from keras.models import Model
from keras.layers import Input, Embedding
from keras.utils import plot_model

from nn_arch import dnn_cache, cnn_cache, rnn_cache

from util import map_item


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

path_embed = 'feat/embed.pkl'
path_sent = 'feat/sent_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_sent, 'rb') as f:
    sents = pk.load(f)

funcs = {'dnn': dnn_cache,
         'cnn': cnn_cache,
         'rnn': rnn_cache}

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5',
         'dnn_plot': 'model/plot/dnn_cache.png',
         'cnn_plot': 'model/plot/cnn_cache.png',
         'rnn_plot': 'model/plot/rnn_cache.png'}

models = {'dnn': load_model('dnn', embed_mat, seq_len),
          'cnn': load_model('cnn', embed_mat, seq_len),
          'rnn': load_model('rnn', embed_mat, seq_len)}


def cache(name, sents, path_cache):
    model = map_item(name, models)
    encode_sents = model.predict(sents)
    with open(path_cache, 'wb') as f:
        pk.dump(encode_sents, f)


if __name__ == '__main__':
    path_cache = 'cache/dnn.pkl'
    cache('dnn', sents, path_cache)
    path_cache = 'cache/cnn.pkl'
    cache('cnn', sents, path_cache)
    path_cache = 'cache/rnn.pkl'
    cache('rnn', sents, path_cache)
