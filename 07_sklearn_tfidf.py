# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 23:11
# @Author  : quincyqiang
# @File    : 07_sklearn_tfidf.py
# @Software: PyCharm


# 基于sklearn tfidf
import pandas as pd
from tqdm import tqdm
from utils import load_docs,get_keywords,load_model

test_data_path='data/all_doc_pos.pkl'
test_data=pd.read_csv('data/test_docs.csv')
test_docs=load_docs(test_data_path,test_data)


train_data_path='data/new_train_docs.pkl'
train_data=pd.read_csv('data/new_train_docs.csv')
train_docs=load_docs(train_data_path,train_data)

submit_result='result/07_sklearn_tfidf.csv'
train_result='result/07_train.csv'


def predict(docs):
    """
    预测
    :return:
    """
    print("predicting...")
    labels_1 = []
    labels_2 = []
    model_path = 'model/cv_tfidf.pkl'
    cv, tfidf_transformer = load_model(model_path, test_data_path, test_data)
    feature_names = cv.get_feature_names()
    for doc in tqdm(docs):
        doc = " ".join(doc.split()[:500])  # 关键词在文章的前半部分

        keywords = get_keywords(cv, tfidf_transformer, doc, feature_names)
        if len(keywords) >= 2:
            labels_1.append(keywords[0])
            labels_2.append(keywords[1])
        else:
            labels_1.append(keywords[0])
            labels_2.append(' ')

    return labels_1,labels_2


def extract_keyword_sklearn(documents,result_file):
    """
    生成提交结果
    :param documents
    :param result_file:
    :return:
    """

    ids = test_data['id']
    labels_1, labels_2=predict(documents)
    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}
    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv(result_file, index=False)


extract_keyword_sklearn(test_docs,submit_result)


def evaluate():
    pass