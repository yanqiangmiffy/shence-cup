# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 02_jieba_tfidf.py
# @Time    : 2018/8/30 18:24
import pandas as pd
from jieba.analyse import *
import jieba.analyse
from tqdm import tqdm
jieba.analyse.set_stop_words('data/stop_words.txt')

# 加载数据
all_docs_file=open('data/all_docs.txt','r',encoding='utf-8')

def extract_keyword_by_tfidf():

    ids=[]
    labels_1=[]
    labels_2=[]
    for line in tqdm(all_docs_file):
        data=line.strip().split('')
        ids.append(data[0])
        text=data[1]+data[2]
        keywords=[keyword for keyword, weight in extract_tags(text, withWeight=True,topK=5)]
        if len(keywords)>=2:
            labels_1.append(keywords[0])
            labels_2.append(keywords[1])
        else:
            print(text)
            labels_1.append(keywords[0])
            labels_2.append('')

    import csv
    with open('result/02_snow_tfidf.csv','w',encoding='utf-8',newline='') as out_data:
        csv_writer=csv.writer(out_data)
        csv_writer.writerow(('id','label1','label2'))
        for id,label1,label2 in zip(ids,labels_1,labels_2):
            csv_writer.writerow((id,label1,label2))

    df_data=pd.DataFrame([ids,labels_1,labels_2],columns=['id','label1','label2'])
    df_data.to_csv('result/02_jieba_tfidf.csv',index=False)


def extract_keyword_by_textrank():

    ids=[]
    labels_1=[]
    labels_2=[]
    for line in tqdm(all_docs_file):
        data=line.strip().split('')
        ids.append(data[0])
        text=data[1]+data[2]
        keywords=[keyword for keyword, weight in textrank(text, withWeight=True,topK=5)]
        if len(keywords)>=2:
            labels_1.append(keywords[0])
            labels_2.append(keywords[1])
        else:
            print(text)
            labels_1.append(keywords[0])
            labels_2.append('')

    import csv
    with open('result/02_snow_tfidf.csv','w',encoding='utf-8',newline='') as out_data:
        csv_writer=csv.writer(out_data)
        csv_writer.writerow(('id','label1','label2'))
        for id,label1,label2 in zip(ids,labels_1,labels_2):
            csv_writer.writerow((id,label1,label2))

    df_data=pd.DataFrame([ids,labels_1,labels_2],columns=['id','label1','label2'])
    df_data.to_csv('result/02_jieba_tfidf.csv',index=False)