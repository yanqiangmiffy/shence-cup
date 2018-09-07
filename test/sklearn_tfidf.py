# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : demo.py
# @Time    : 2018/9/7 11:43
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import pickle

# 加载语料
with open('../data/train_docs.pkl','rb') as in_data:
    train_docs=pickle.load(in_data)

cv = CountVectorizer(min_df=1, max_df=1.0)


word_count_vector=cv.fit_transform(train_docs)
feature_names = cv.get_feature_names()

print(feature_names)
