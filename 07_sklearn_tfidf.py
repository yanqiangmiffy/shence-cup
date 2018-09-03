# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 23:11
# @Author  : quincyqiang
# @File    : 07_sklearn_tfidf.py
# @Software: PyCharm


# 基于sklearn tfidf
import pandas as pd
from tqdm import tqdm
from utils import load_docs,get_keywords,load_model
test_data=pd.read_csv('data/test_docs.csv')
train_data=pd.read_csv('data/new_train_docs.csv')

docs=load_docs(data_path = 'data/all_doc_pos.pkl')


def predict(test_docs):
    """
    预测
    :return:
    """
    print("predicting...")
    ids= test_data['id']

    labels_1 = []
    labels_2 = []

    cv, tfidf_transformer = load_model(model_path='model/cv_tfidf.pkl')
    feature_names=cv.get_feature_names()
    for doc in tqdm(test_docs):
        doc=" ".join(doc.split()[:500]) # 关键词在文章的前半部分

        keywords=get_keywords(cv,tfidf_transformer,doc,feature_names)
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


predict(docs[100000:])