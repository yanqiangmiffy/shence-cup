# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 21:57
# @Author  : quincyqiang
# @File    : 06_jieba_ensemble.py
# @Software: PyCharm

import pandas as pd
from tqdm import tqdm
from jieba.analyse import extract_tags # tf-idf
from jieba import posseg
import random
import jieba
jieba.analyse.set_stop_words('data/stop_words.txt') # 去除停用词
jieba.load_userdict('data/custom_dict.txt') # 设置词库

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

'''

all_docs_file=open('data/all_docs.txt','r',encoding='utf-8')
allow_pos={'nr':1,'nz':2,'ns':3,'nt':4,'eng':5,'n':6,'l':7,'i':8,'a':9,'nrt':10,'v':11,'t':12}
train_docs_file=open('data/train_docs_keywords.txt','r',encoding='utf-8')


def extract_keyword_ensemble():
    ids = []
    labels_1 = []
    labels_2 = []
    empty=0
    for line in tqdm(all_docs_file):
        keywords = []
        data = line.strip().split('\001')
        ids.append(data[0])

        word_tags=[(word,pos) for word,pos in posseg.cut(data[1])] # 标题
        for word_pos in word_tags:
            if word_pos[1] in allow_pos:
                keywords.append(word_pos)

        keywords = sorted(keywords, reverse=False, key=lambda x: allow_pos[x[1]])

        if len(keywords) <2:
            # 使用tf-idf
            empty+=1
            temp_keywords = [keyword for keyword, weight in extract_tags(data[1]+data[2], withWeight=True, topK=5)]
            labels_1.append(temp_keywords[0])
            labels_2.append(temp_keywords[1])
        else:
            labels_1.append(keywords[0][0])
            labels_2.append(keywords[1][0])
    print(empty)

    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}

    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/06_jieba_ensemble.csv', index=False)
    print(empty)
    # 关闭文件
    all_docs_file.close()
    train_docs_file.close()


if __name__ == '__main__':
    extract_keyword_ensemble()