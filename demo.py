# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : demo.py
# @Time    : 2018/8/31 10:12

# from snownlp import SnowNLP
# import pandas as pd
# text="《命运速递》主题曲MV曝光吕晓霖片中虐恋触人心弦"
# snow=SnowNLP(text)
# keyword=snow.keywords(limit=5)
# print(keyword[:2])
# print(snow.words)
#
# ids,labels_1,labels_2=[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]]
# data={'id':ids,
#       'label1':labels_1,
#       'label2':labels_2}
# df_data=pd.DataFrame(data,columns=['id','label1','label2'])
# df_data.to_csv('result/01_textrank1.csv',index=False)

# coding=utf-8
# 分句
from pyltp import SentenceSplitter
sents = SentenceSplitter.split('游戏讲述的是一群富有勇气又有一丝小坏的人们的传奇，他们正试着逃离惠灵顿威尔士单调古板的生活。')  # 分句
print('\n'.join(sents))

# 分词
import os
from pyltp import Segmentor
LTP_DATA_DIR='D:\Data\ltp_data_v3.4.0'
cws_model_path=os.path.join(LTP_DATA_DIR,'cws.model')
segmentor=Segmentor()
segmentor.load(cws_model_path)
words=segmentor.segment('游戏讲述的是一群富有勇气又有一丝小坏的人们的传奇，他们正试着逃离惠灵顿威尔士单调古板的生活。')
print(type(words))
print('\t'.join(words))
segmentor.release()


# 自定义词典
import os
LTP_DATA_DIR='D:\Data\ltp_data_v3.4.0'  # ltp模型目录的路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

from pyltp import Segmentor
segmentor = Segmentor()  # 初始化实例
segmentor.load_with_lexicon(cws_model_path, 'lexicon') # 加载模型，第二个参数是您的外部词典文件路径
words = segmentor.segment('亚硝酸盐是一种化学物质')
print('\t'.join(words))
segmentor.release()

# 词性标注
# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR='D:\Data\ltp_data_v3.4.0'
# ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

from pyltp import Postagger
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型

words = ['元芳', '你', '怎么', '看']  # 分词结果
postags = postagger.postag(words)  # 词性标注

print('\t'.join(postags))
postagger.release()  # 释放模型

# 命名实体识别
import os
LTP_DATA_DIR='D:\Data\ltp_data_v3.4.0'  # ltp模型目录的路径
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`

from pyltp import NamedEntityRecognizer
recognizer = NamedEntityRecognizer() # 初始化实例
recognizer.load(ner_model_path)  # 加载模型

words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
netags = recognizer.recognize(words, postags)  # 命名实体识别

print('\t'.join(netags))
recognizer.release()  # 释放模型


