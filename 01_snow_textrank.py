# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 01_snow_textrank.py
# @Time    : 2018/8/30 17:17

import pandas as pd
from tqdm import tqdm
from snownlp import SnowNLP
import random
# 加载数据
all_docs_file=open('data/all_docs.txt','r',encoding='utf-8')

ids=[]
labels_1=[]
labels_2=[]
for line in tqdm(all_docs_file):
    data=line.strip().split('')
    ids.append(data[0])
    # text=data[1] + data[2].replace('\xa0', '').replace('\u3000', '')
    text=data[1]+data[2]
    snow=SnowNLP(text)
    keyword=snow.keywords(limit=5)
    print(keyword)
    if len(keyword)>=2:
        labels_1.append(keyword[0])
        labels_2.append(keyword[1])
    if len(keyword)==1:
        labels_1.append(keyword[0])
        labels_2.append(snow.words[1])
    if len(keyword)<1:
        if len(snow.words)>1:
            labels_1.append(snow.words[0])
            labels_2.append(snow.words[1])
        else:
            print(data[1])
            labels_1.append(snow.words[0])
            labels_2.append('')


data={'id':ids,
      'label1':labels_1,
      'label2':labels_2}
df_data=pd.DataFrame(data,columns=['id','label1','label2'])
df_data.to_csv('result/01_textrank1.csv',index=False)

