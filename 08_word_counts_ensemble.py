# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 08_word_counts_ensemble.py
# @Time    : 2018/9/4 17:12
# 词频->词性->长度


import os
from collections import Counter
import pandas as pd
from tqdm import tqdm
from jieba import posseg
import pickle
import jieba
jieba.load_userdict('data/custom_dict.txt')  # 设置词库

test_data_path='data/test_word_counts.pkl'
train_data_path='data/train_word_counts.pkl'

test_data = pd.read_csv('data/test_docs.csv')
train_data = pd.read_csv('data/new_train_docs.csv')

allow_pos = {'nr': 12, 'nz': 11, 'ns': 10, 'nt': 9, 'eng': 8, 'l': 7,
             'i': 6, 'a': 5, 'nrt': 4, 'n': 3, 'v': 2, 't': 1}
stop_words = open('data/stop_words.txt', 'r', encoding='utf-8').read().split('\n')


def generate_name(word_tags):
    """
    解决分词缺陷：杰森·斯坦森
    :param word_tags:
    :return:
    """
    name_pos = ['ns', 'n', 'vn', 'nr', 'nt', 'eng', 'nrt']
    for word_tag in word_tags:
        if word_tag[0] == '·':
            index = word_tags.index(word_tag)
            if (index+1)<len(word_tags):
                prefix = word_tags[index - 1]
                suffix = word_tags[index + 1]
                if prefix[1] in name_pos and suffix[1] in name_pos:
                    name = prefix[0] + word_tags[index][0] + suffix[0]
                    word_tags = word_tags[index + 2:]
                    word_tags.insert(0, (name, 'nr'))
    return word_tags


def keyword_counts(data_path,df_data,stop_words=(),allow_pos=()):
    """
    标题和文章分句 重要词性组成
    :return:
    """
    ids, titles, docs = df_data['id'], df_data['title'], df_data['doc']
    print("generate docs..")
    all_data = []
    for data in zip(titles, docs):
        candidate_keywords=[]
        doc = data[0] + '。' + data[1]
        word_tags = []
        for word, pos in posseg.cut(doc):
            if word not in stop_words and pos in allow_pos:
                if len(word) > 1:
                    word_tags.append((word, pos))

        if '·' in word_tags:
            word_tags=generate_name(word_tags)

        # print(len(word_tags),len(set(word_tags)),word_tags)
        for key,value in Counter(word_tags).items():
            candidate_keywords.append((key[0],key[1],value))
        # 排序
        candidate_keywords=sorted(candidate_keywords,key=lambda x:(x[2],allow_pos[x[1]],len(x[0])),reverse=True)
        print(candidate_keywords)
        all_data.append(candidate_keywords)




    with open(data_path, 'wb') as out_data:
        pickle.dump(all_data, out_data, pickle.HIGHEST_PROTOCOL)
    return all_data


# keyword_counts(train_data_path,train_data,stop_words,allow_pos)
# keyword_counts(test_data_path,test_data,stop_words,allow_pos)


def evaluate(data_path, df_data, stop_words=(), allow_pos=()):
    """
    标题和文章分句 重要词性组成
    :return:
    """
    ids, titles, docs = df_data['id'], df_data['title'], df_data['doc']

    true_keywords = train_data['keyword'].apply(lambda x: x.split(','))
    score = 0

    all_data = []
    for data in zip(titles, docs,true_keywords):

        candidate_keywords = []
        doc = data[0] + '。' + data[1]
        word_tags = []
        for word, pos in posseg.cut(doc):
            if word not in stop_words and pos in allow_pos:
                if len(word) > 1:
                    word_tags.append((word, pos))

        if '·' in word_tags:
            word_tags = generate_name(word_tags)

        # print(len(word_tags),len(set(word_tags)),word_tags)
        for key, value in Counter(word_tags).items():

            candidate_keywords.append((key[0], key[1], value))
        # 排序
        candidate_keywords = sorted(candidate_keywords, key=lambda x: (x[2], allow_pos[x[1]], len(x[0])), reverse=True)
        print(data[2],candidate_keywords)
        all_data.append(candidate_keywords)

        # 评价
        # true_keys = data[2]
        # key_1 = candidate_keywords[0][0]
        # key_2 = candidate_keywords[1][0]
        #
        # if key_1 not in true_keys or key_2 not in true_keys:
        #     print((key_1, key_2), '--', true_keys)
        #
        # if key_1 in true_keys:
        #     score += 0.5
        # if key_2 in true_keys:
        #     score += 0.5
    print(score)
    with open(data_path, 'wb') as out_data:
        pickle.dump(all_data, out_data, pickle.HIGHEST_PROTOCOL)
    return all_data


evaluate(train_data_path,train_data,stop_words,allow_pos)