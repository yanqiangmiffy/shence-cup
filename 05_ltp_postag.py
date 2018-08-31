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

postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型





'''
 词性：
 nh	person name	杜甫, 汤姆
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
        words=segmentor.segment(data[1])
        postags=postagger.postag(words)
        print("|".join(words)," ".join(postags))
    # data = {'id': ids,
    #         'label1': labels_1,
    #         'label2': labels_2}
    #
    # df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    # df_data.to_csv('result/04_jieba_postag.csv', index=False)

extract_keyword_by_possag()

all_docs_file.close()