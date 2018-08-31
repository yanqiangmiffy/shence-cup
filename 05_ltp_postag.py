# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 02_jieba_tfidf.py
# @Time    : 2018/8/30 18:24
import pandas as pd
from tqdm import tqdm
import random
import os
from pyltp import Segmentor,Postagger
LTP_DATA_DIR='D:\Data\ltp_data_v3.4.0'

cws_model_path=os.path.join(LTP_DATA_DIR,'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

segmentor=Segmentor()
segmentor.load(cws_model_path)
words=segmentor.segment('游戏讲述的是一群富有勇气又有一丝小坏的人们的传奇，他们正试着逃离惠灵顿威尔士单调古板的生活。')

postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型
words = ['元芳', '你', '怎么', '看']  # 分词结果
postags = postagger.postag(words)  # 词性标注





'''
 词性：
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


    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}

    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/04_jieba_postag.csv', index=False)

extract_keyword_by_possag()

all_docs_file.close()