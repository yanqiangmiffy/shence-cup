# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : demo.py
# @Time    : 2018/8/31 10:12
from snownlp import SnowNLP
import pandas as pd
text="《命运速递》主题曲MV曝光吕晓霖片中虐恋触人心弦"
snow=SnowNLP(text)
keyword=snow.keywords(limit=5)
print(keyword[:2])
print(snow.words)

ids,labels_1,labels_2=[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]]
data={'id':ids,
      'label1':labels_1,
      'label2':labels_2}
df_data=pd.DataFrame(data,columns=['id','label1','label2'])
df_data.to_csv('result/01_textrank1.csv',index=False)