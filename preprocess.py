import os

import re

from random import shuffle, sample

from util import load_word_re, load_type_re, load_word_pair, word_replace


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_word_pair(path_homo)
syno_dict = load_word_pair(path_syno)


def save_pair(path, pairs):
    head = 'text1,text2,flag'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text1, text2, flag in pairs:
            f.write(text1 + ',' + text2 + ',' + str(flag) + '\n')


def make_pair(path_univ_dir, path_train_pair, path_test_pair):
    labels = list()
    label_texts = dict()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        labels.append(label)
        label_texts[label] = list()
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                label_texts[label].append(line.strip())
    pairs = list()
    fold = 2  # neg / pos
    for i in range(len(labels)):
        texts = label_texts[labels[i]]
        rest_texts = list()
        for j in range(len(labels)):
            if j != i:
                rest_texts.extend(label_texts[labels[j]])
        for j in range(len(texts) - 1):
            pairs.append((texts[j], texts[j - 1], 1))
            neg_texts = sample(rest_texts, fold)
            for neg_text in neg_texts:
                pairs.append((texts[j], neg_text, 0))
    shuffle(pairs)
    bound = int(len(pairs) * 0.9)
    save_pair(path_train_pair, pairs[:bound])
    save_pair(path_test_pair, pairs[bound:])


def save(path, texts, labels):
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(texts, labels):
            f.write(text + ',' + label + '\n')


def merge(path_univ_dir, path_train, path_test):
    texts = list()
    labels = list()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                texts.append(line.strip())
                labels.append(label)
    texts_labels = list(zip(texts, labels))
    shuffle(texts_labels)
    texts, labels = zip(*texts_labels)
    bound = int(len(texts) * 0.9)
    save(path_train, texts[:bound], labels[:bound])
    save(path_test, texts[bound:], labels[bound:])


def prepare(path_univ_dir):
    files = os.listdir(path_univ_dir)
    for file in files:
        text_set = set()
        texts = list()
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                text = re.sub(stop_word_re, '', line.strip())
                for word_type, word_re in word_type_re.items():
                    text = re.sub(word_re, word_type, text)
                text = word_replace(text, homo_dict)
                text = word_replace(text, syno_dict)
                if text not in text_set:
                    text_set.add(text)
                    texts.append(text)
        shuffle(texts)
        with open(os.path.join(path_univ_dir, file), 'w') as f:
            for text in texts:
                f.write(text + '\n')


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    prepare(path_univ_dir)
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    merge(path_univ_dir, path_train, path_test)
    path_train_pair = 'data/train_pair.csv'
    path_test_pair = 'data/test_pair.csv'
    make_pair(path_univ_dir, path_train_pair, path_test_pair)
