# -*- coding: utf-8 -*-
# @Time    : 2018/9/2 21:01
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm
import os
# 基于sklearn tfidf
import pandas as pd
from tqdm import tqdm
from jieba.analyse import extract_tags  # tf-idf
from jieba import posseg
import pickle
import jieba
jieba.load_userdict('data/custom_dict.txt')  # 设置词库
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

test_data = pd.read_csv('data/test_docs.csv')
train_data = pd.read_csv('data/new_train_docs.csv')

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
    for data in tqdm(titles, docs):
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


def load_docs(data_path,df_data):
    """
    加载 分词后的文档
    :return:
    """
    if os.path.exists(data_path):
        with open(data_path, 'rb') as in_data:
            all_docs = pickle.load(in_data)
        return all_docs

    all_docs=generate_docs(data_path,df_data)
    return all_docs


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


def train(model_path,data_path,df_data):
    """
    训练并返回模型
    :return:
    """
    documents = load_docs(data_path,df_data)
    print("train....")
    cv = CountVectorizer(max_df=0.85, min_df=1, stop_words=stop_words)
    word_count_vector = cv.fit_transform(documents)

    # 计算tfidf
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    with open(model_path, 'wb') as out_data:
        pickle.dump(cv, out_data, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tfidf_transformer, out_data, pickle.HIGHEST_PROTOCOL)
    return cv, tfidf_transformer


def load_model(model_path,data_path,df_data):
    """
    加载训练好的模型
    :param model_path:
    :return:
    """
    if os.path.exists(model_path):
        with open(model_path, 'rb') as in_data:
            cv = pickle.load(in_data)
            tfidf_transformer = pickle.load(in_data)
        return cv, tfidf_transformer

    cv, tfidf_transformer = train(model_path,data_path,df_data)
    return cv, tfidf_transformer


def get_keywords(cv, tfidf_transformer, doc, feature_names):
    tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

    sorted_items = sort_coo(tf_idf_vector.tocoo())

    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)

    # print(keywords)
    # for k in keywords:
    #     print(k, keywords[k])

    keywords = list(keywords.keys())

    return keywords
