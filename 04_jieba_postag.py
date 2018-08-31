# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 02_jieba_tfidf.py
# @Time    : 2018/8/30 18:24
import pandas as pd
from tqdm import tqdm
from jieba import posseg
import random
import jieba
'''
 nr 人名
 nz 其他专名
 ns 地名
 n 名词
'''

all_docs_file=open('data/all_docs.txt','r',encoding='utf-8')

# 基于jieba的textrank提取关键词
def extract_keyword_by_possag():
    ids=[]
    labels_1=[]
    labels_2=[]
    for line in tqdm(all_docs_file):
        keywords=[]
        data=line.strip().split('')

        word_tags=[(word,pos) for word,pos in posseg.cut(data[1])] # 标题
        for word_tag in word_tags:
            if word_tag[1] == 'nr':
                keywords.append(word_tag[0])
            for word_tag in word_tags:
                if word_tag[1] == 'ns':
                    keywords.append(word_tag[0])
            for word_tag in word_tags:
                if word_tag[1] == 'nz':
                    keywords.append(word_tag[0])
            for word_tag in word_tags:
                if word_tag[1] == 'ns':
                    keywords.append(word_tag[0])
            for word_tag in word_tags:
                if word_tag[1] == 'n':
                    keywords.append(word_tag[0])

        ids.append(data[0])
        if len(keywords)<2:
            words = [word for word in jieba.cut(data[1])]
            labels_1.append(random.choice(words))
            labels_2.append(random.choice(words))
        else:
            labels_1.append(keywords[0])
            labels_2.append(keywords[1])

    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}

    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/04_jieba_postag.csv', index=False)

extract_keyword_by_possag()

all_docs_file.close()