from keras.layers import Dense, SeparableConv1D, LSTM
from keras.layers import Dropout, GlobalMaxPooling1D, BatchNormalization, Masking
from keras.layers import Lambda, Concatenate, Subtract, Multiply

import keras.backend as K


def dnn(embed_input1, embed_input2):
    mean = Lambda(lambda a: K.mean(a, axis=1))
    da1 = Dense(200, activation='relu')
    da2 = Dense(200, activation='relu')
    da3 = Dense(1, activation='sigmoid')
    x = mean(embed_input1)
    x = da1(x)
    x = da2(x)
    y = mean(embed_input2)
    y = da1(y)
    y = da2(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da3(z)


def cnn(embed_input1, embed_input2):
    ca1 = SeparableConv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = SeparableConv1D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    bn = BatchNormalization()
    mp = GlobalMaxPooling1D()
    concat1 = Concatenate()
    concat2 = Concatenate()
    da = Dense(1, activation='sigmoid')
    x1 = ca1(embed_input1)
    x1 = bn(x1)
    x1 = mp(x1)
    x2 = ca2(embed_input1)
    x2 = bn(x2)
    x2 = mp(x2)
    x3 = ca3(embed_input1)
    x3 = bn(x3)
    x3 = mp(x3)
    x = concat1([x1, x2, x3])
    y1 = ca1(embed_input2)
    y1 = bn(y1)
    y1 = mp(y1)
    y2 = ca2(embed_input2)
    y2 = bn(y2)
    y2 = mp(y2)
    y3 = ca3(embed_input2)
    y3 = bn(y3)
    y3 = mp(y3)
    y = concat1([y1, y2, y3])
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = concat2([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)


def rnn(embed_input1, embed_input2):
    mask = Masking()
    ra = LSTM(200, activation='tanh')
    da = Dense(1, activation='sigmoid')
    x = mask(embed_input1)
    x = ra(x)
    y = mask(embed_input2)
    y = ra(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)
