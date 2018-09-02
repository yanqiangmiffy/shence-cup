# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 23:11
# @Author  : quincyqiang
# @File    : 07_sklearn_tfidf.py
# @Software: PyCharm


import os
# 基于sklearn tfidf
import pandas as pd
from tqdm import tqdm
from jieba.analyse import extract_tags # tf-idf
from jieba import posseg
import pickle
import jieba
jieba.load_userdict('data/custom_dict.txt') # 设置词库
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
'''
 nr 人名
 nz 其他专名
 ns 地名
 nt 机构团体
 n 名词
 l 习用语
 i 成语
 a 形容词
 nrt
 v 动词
 t 时间词
 vn 动名词
'''

all_docs_file=open('data/all_docs.txt','r',encoding='utf-8')
allow_pos={'nr':1,'nz':2,'ns':3,'nt':4,'eng':5,'n':6,'l':7,'i':8,'a':9,'nrt':10,'v':11,'t':12,'vn':13}
stop_words=open('data/stop_words.txt','r',encoding='utf-8').read().split('\n')


def generate_docs():
    """
    标题和文章分句 重要词性组成
    :return:
    """
    data_path='data/all_doc_pos.pkl'
    if os.path.exists(data_path):
        with open(data_path,'rb') as in_data:
            all_docs=pickle.load(in_data)
        return all_docs

    print("generate docs..")
    all_docs=[]
    for line in tqdm(all_docs_file):
        data = line.strip().split('\001')
        doc=data[1]+'。'+data[2]
        word_tags=[]
        for word, pos in posseg.cut(doc):
            if word not in stop_words and pos in allow_pos:
                word_tags.append((word, pos))
        new_doc=" ".join([word_tag[0] for word_tag in word_tags])
        # print(new_doc)
        all_docs.append(new_doc)
    with open(data_path,'wb') as out_data:
        pickle.dump(all_docs,out_data,pickle.HIGHEST_PROTOCOL)
    return all_docs


docs=generate_docs()


def sort_coo(coo_matrix):
    """
    排序
    :param coo_matrix:
    :return:
    """
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """
    获取关键字
    :param feature_names:
    :param sorted_items:
    :param topn:
    :return:
    """
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:

        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def train(documents):
    """
    训练并返回模型
    :return:
    """
    model_path = 'model/cv_tfidf.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as in_data:
            cv = pickle.load(in_data)
            tfidf_transformer = pickle.load(in_data)
        return cv,tfidf_transformer


    print("train....")
    cv=CountVectorizer(max_df=0.85,min_df=1,stop_words=stop_words)
    word_count_vector = cv.fit_transform(documents)

    # 计算tfidf
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    with open(model_path,'wb') as out_data:
        pickle.dump(cv,out_data,pickle.HIGHEST_PROTOCOL)
        pickle.dump(tfidf_transformer,out_data,pickle.HIGHEST_PROTOCOL)
    return cv,tfidf_transformer


cv,tfidf_transformer=train(docs)


def predict(test_docs):
    """
    预测
    :return:
    """
    print("predicting...")
    ids = []
    labels_1 = []
    labels_2 = []

    for line in tqdm(all_docs_file):
        data = line.strip().split('\001')
        ids.append(data[0])

    feature_names=cv.get_feature_names()
    for doc in tqdm(test_docs):
        doc=" ".join(doc.split()[:500]) # 关键词在文章的前半部分
        tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

        sorted_items = sort_coo(tf_idf_vector.tocoo())

        keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
        # for k in keywords:
        #     print(k, keywords[k])
        keywords=list(keywords.keys())
        if len(keywords) >=2:
            labels_1.append(keywords[0])
            labels_2.append(keywords[1])
        else:
            print(test_docs.index(doc))
            labels_1.append(keywords[0])
            labels_2.append(' ')

    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}
    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/07_sklearn_tfidf.csv', index=False)

    # 关闭文件
    all_docs_file.close()


predict(docs)