# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 02_jieba_tfidf.py
# @Time    : 2018/8/30 18:24
import pickle
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

all_pos=['ns', 'n', 'vn', 'v','nr','nt','eng','l','i','a','nrt']

def extract_keyword_by_tfidf(test_data):

    ids,titles,docs=test_data['id'],test_data['title'],test_data['doc']
    with open('data/all_doc_pos.pkl','rb') as in_data:
        test_docs=pickle.load(in_data)
    labels_1 = []
    labels_2 = []
    for title,doc in tqdm(zip(titles,test_docs)):
        temp_keywords = [keyword for keyword in
                         extract_tags(title + doc[:100]+doc[:-100],topK=3)]

        print("tfidf:",temp_keywords)
        if len(temp_keywords)>2:
            labels_1.append(temp_keywords[0])
            labels_2.append(temp_keywords[1])
        else:
            labels_1.append(temp_keywords[0])
            labels_2.append(' ')

    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}

    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/02_jieba_tfidf.csv', index=False)


def evaluate(df_data=None):
    ids, titles, docs = df_data['id'], df_data['title'], df_data['doc']
    true_keywords=df_data['keyword'].apply(lambda x:x.split(','))
    labels_1 = []
    labels_2 = []
    empty = 0
    score=0
    import pickle
    with open('data/new_train_docs.pkl', 'rb') as in_data:
        docs = pickle.load(in_data)
    for title,doc,true_keyword in tqdm(zip(titles, docs,true_keywords)):

        temp_keywords = [keyword for keyword in
                         extract_tags(title*2+doc[:100]+doc[:-100],topK=2)]
        # print("tfidf:",temp_keywords)
        labels_1.append(temp_keywords[0])
        labels_2.append(temp_keywords[1])
        print(temp_keywords[0], temp_keywords[1], true_keyword)
        if temp_keywords[0] in true_keyword:
            score += 0.5
        if temp_keywords[1] in true_keyword:
            score += 0.5

    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}
    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/06_train.csv', index=False)
    print("使用tf-idf提取的次数：", empty)
    print("最终得分为：",score)
if __name__ == '__main__':
    # extract_keyword_by_tfidf(test_data)
    evaluate(train_data)