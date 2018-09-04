# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : utils.py
# @Time    : 2018/8/31 15:23

# 统计下文档的长度
# import os
# import pickle
# def load_docs():
#     """
#     标题和文章分句 重要词性组成
#     :return:
#     """
#     data_path='data/all_doc_pos.pkl'
#     if os.path.exists(data_path):
#         with open(data_path,'rb') as in_data:
#             all_docs=pickle.load(in_data)
#         return all_docs
#
#
#
#
# docs=load_docs()
# for doc in docs:
#     print(doc)
#     print('============')
# import pandas as pd
# df_data=pd.DataFrame(data=docs,columns=['doc'])
# df_data['doc_len']=df_data['doc'].apply(lambda x:len(x.split(' ')))
# print(df_data['doc_len'].describe())
#
# """
# count    108295.000000
# mean        259.534309
# std         231.359050
# min           3.000000
# 25%         127.000000
# 50%         193.000000
# 75%         310.000000
# max        5973.000000
# Name: doc_len, dtype: float64
# """
#
# # 统计下训练集的词性(pyltp)
# import pandas as pd
# import os
# from pyltp import Segmentor,Postagger
# LTP_DATA_DIR='E:\Project\Python\pyltp\ltp'
#
# pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
# postagger = Postagger() # 初始化实例
# postagger.load(pos_model_path)  # 加载模型
#
# train_data=pd.read_csv('data/new_train_docs.csv')
# keywords=train_data['keyword'].apply(lambda x:x.split(',')).tolist()
# # keywords=[word  for keyword in keywords for word in keyword]
# from collections import Counter
# pos=[]
# for keyword in keywords:
#     print(keyword," ".join(postagger.postag(keyword)))
#     for word in keyword:
#         pos.append("".join(postagger.postag([word])))
# pos_dict=Counter(pos)
# print(pos_dict)
#
#
# # 全部词性
# # for line in open('data/lexicon.txt','r',encoding='utf-8'):
# #     print(line.strip()," ".join(postagger.postag([line.strip()])))
#
# import jieba.posseg
# jieba.load_userdict('data/custom_dict.txt')
#
# jieba_pos=[]
# for keyword in keywords:
#     for word in keyword:
#         for word,tag in jieba.posseg.cut(word):
#             if tag=='ng':
#                 print(word)
#             jieba_pos.append(tag)
# jieba_pos_dict=Counter(jieba_pos)
# print(jieba_pos_dict)


import pandas as pd
from tqdm import tqdm
from jieba import posseg
import pickle
train_data = pd.read_csv('data/new_train_docs.csv')
train_data_path='data/new_train_docs.pkl'

allow_pos = {'nr': 1, 'nz': 2, 'ns': 3, 'nt': 4, 'eng': 5, 'l': 6, 'i': 7, 'a': 8, 'nrt': 9, 'n': 10, 'v': 11, 't': 12}
stop_words = open('data/stop_words.txt', 'r', encoding='utf-8').read().split('\n')


def generate_docs(data_path,df_data):
    """
    标题和文章分句 重要词性组成
    :return:
    """
    ids, titles, docs = df_data['id'], df_data['title'], df_data['doc']
    print("generate docs..")
    all_docs = []
    for data in tqdm(zip(titles, docs)):
        doc = data[0] + '。' + data[1]
        word_tags = []
        for word, pos in posseg.cut(doc):
            if word not in stop_words and pos in allow_pos:
                if len(word) > 1:
                    word_tags.append((word, pos))
        new_doc = " ".join([word_tag[0] for word_tag in word_tags])
        # print(new_doc)
        all_docs.append(new_doc)
    with open(data_path, 'wb') as out_data:
        pickle.dump(all_docs, out_data, pickle.HIGHEST_PROTOCOL)
    return all_docs

train=generate_docs(train_data_path,train_data)
for doc in train:
    print(doc)