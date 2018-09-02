# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 02_jieba_tfidf.py
# @Time    : 2018/8/30 18:24
import pandas as pd
from jieba.analyse import *
import jieba.analyse
from tqdm import tqdm
jieba.load_userdict('data/custom_dict.txt')
jieba.analyse.set_stop_words('data/stop_words.txt')
import random
# 加载数据
all_docs_file=open('data/all_docs.txt','r',encoding='utf-8')


# 基于jieba的textrank提取关键词
def extract_keyword_by_textrank():
    ids=[]
    labels_1=[]
    labels_2=[]
    for line in tqdm(all_docs_file):
        data=line.strip().split('')
        ids.append(data[0])
        text=data[1]+data[2] #标题加文章
        # text=data[1] #标题
        keywords=[keyword for keyword, weight in textrank(text, withWeight=True,topK=5)]
        if len(keywords)>=2:
            labels_1.append(keywords[0])
            labels_2.append(keywords[1])
        elif len(keywords)==1:
            labels_1.append(keywords[0])
            labels_2.append('')
        else:
            words=[word for word in jieba.cut(data[1])]
            labels_1.append(words[0])
            labels_2.append(random.choice(words))

    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}
    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/03_jieba_textrank.csv', index=False)

    # 关闭文件
    all_docs_file.close()

extract_keyword_by_textrank()