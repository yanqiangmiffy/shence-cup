# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 02_jieba_tfidf.py
# @Time    : 2018/8/30 18:24
import pandas as pd
from jieba.analyse import *
import jieba.analyse
from tqdm import tqdm
jieba.analyse.set_stop_words('data/stop_words.txt')
jieba.load_userdict('data/custom_dict.txt')
# 加载数据
# extract_keyword_by_tfidf
test_data=pd.read_csv('data/test_docs.csv')
train_data=pd.read_csv('data/new_train_docs.csv')
allow_pos={'nr':1,'nz':2,'ns':3,'nt':4,'eng':5,'l':6,'i':7,'n':8,'a':9,'nrt':10,'v':11,'t':12}


def extract_keyword_by_tfidf(test_data):

    ids,titles,docs=test_data['id'],test_data['title'],test_data['doc']

    labels_1 = []
    labels_2 = []
    for data in tqdm(zip(titles,docs)):
        temp_keywords = [keyword for keyword, weight in extract_tags(data[0] + str(data[1]), withWeight=True, topK=5)]
        # print("tfidf:",temp_keywords)
        labels_1.append(temp_keywords[0])
        labels_2.append(temp_keywords[1])

    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}

    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/02_jieba_tfidf.csv', index=False)

if __name__ == '__main__':
    extract_keyword_by_tfidf(test_data)
