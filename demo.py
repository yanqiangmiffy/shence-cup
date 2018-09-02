# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : demo.py
# @Time    : 2018/8/31 10:12

# from snownlp import SnowNLP
# import pandas as pd
# text="《命运速递》主题曲MV曝光吕晓霖片中虐恋触人心弦"
# snow=SnowNLP(text)
# keyword=snow.keywords(limit=5)
# print(keyword[:2])
# print(snow.words)
#
# ids,labels_1,labels_2=[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]]
# data={'id':ids,
#       'label1':labels_1,
#       'label2':labels_2}
# df_data=pd.DataFrame(data,columns=['id','label1','label2'])
# df_data.to_csv('result/01_textrank1.csv',index=False)

# coding=utf-8
# 分句
from pyltp import SentenceSplitter
sents = SentenceSplitter.split('游戏讲述的是一群富有勇气又有一丝小坏的人们的传奇，他们正试着逃离惠灵顿威尔士单调古板的生活。')  # 分句
print('\n'.join(sents))



# from itertools import chain
# a=chain([1,2,4],['a','c',1])
# for data in a:
#     print(data)

import jieba.posseg
# text="华为新机皇P30pro曝光"
# text="【创新菜】法式红酒烩鸡肉,"
# text="【菜谱】夏日美味小点心——绿豆冰糕"
# text="老公做了一桌菜，色香味俱全，婆婆一回来，看见就不高兴"
# text='"巨舰,航母",我海军一款4万吨巨舰作用比肩航母未来至少需要12艘'
# text='亚丁湾上练兵忙'
text='孕5月一小动作险致流产，孕期别碰这部位，99%孕妇不知'
for word,tag in jieba.posseg.cut(text):
    print(word,tag)

# sen='UPDATE staff_table SET dept="Market" WHERE where dept="IT"' # 提取引号中的内容
# import re
# mth=re.findall('"(.*?)"',sen)
# for m in mth:
#     print(m)
#
# print('我\001爱你')

documents = ['我 爱 北京 天安门，天安门 很 壮观',
             '我 经常 在 广场 拍照']
from sklearn.feature_extraction.text import TfidfVectorizer
global_tfidf_vecc = TfidfVectorizer()
global_count_data = global_tfidf_vecc.fit_transform(documents)
print(global_count_data, global_count_data.shape, type(global_count_data))
# count_array = count_data.toarray()
# print(count_array, count_array.shape, type(count_data))
# print('词汇表为：\n', tfidf_vecc.vocabulary_)

# 统计词频

# from sklearn.feature_extraction.text import CountVectorizer
# # cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
# cv=CountVectorizer(max_df=0.85)
# word_count_vector=cv.fit_transform(documents)
#
# # 计算tfidf
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(word_count_vector)
#
# # 提取关键词
# def sort_coo(coo_matrix):
#     tuples = zip(coo_matrix.col, coo_matrix.data)
#     return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
# def extract_topn_from_vector(feature_names, sorted_items, topn=10):
#     """get the feature names and tf-idf score of top n items"""
#
#     # use only topn items from vector
#     sorted_items = sorted_items[:topn]
#
#     score_vals = []
#     feature_vals = []
#
#     for idx, score in sorted_items:
#         fname = feature_names[idx]
#
#         # keep track of feature name and its corresponding score
#         score_vals.append(round(score, 3))
#         feature_vals.append(feature_names[idx])
#
#     # create a tuples of feature,score
#     # results = zip(feature_vals,score_vals)
#     results = {}
#     for idx in range(len(feature_vals)):
#         results[feature_vals[idx]] = score_vals[idx]
#
#     return results
#
# # you only needs to do this once
# feature_names=cv.get_feature_names()
#
# # get the document that we want to extract keywords from
# doc=documents[1]
#
# #generate tf-idf for the given document
# tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
#
# #sort the tf-idf vectors by descending order of scores
# sorted_items=sort_coo(tf_idf_vector.tocoo())
#
# #extract only the top n; n here is 10
# keywords=extract_topn_from_vector(feature_names,sorted_items,10)
#
# # now print the results
# print("\n===Keywords===")
# for k in keywords:
#     print(k,keywords[k])


data={'a':1,'b':2}
print(list(data.keys()))