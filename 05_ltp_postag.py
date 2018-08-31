# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 02_jieba_tfidf.py
# @Time    : 2018/8/30 18:24
# 使用pyltp的分词和词性标注模块，另外使用train_docs里面的结果

import pandas as pd
from tqdm import tqdm
import random
import os
from pyltp import Segmentor,Postagger
LTP_DATA_DIR='E:\Project\Python\pyltp\ltp'

cws_model_path=os.path.join(LTP_DATA_DIR,'cws.model')
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`

# 加载自定义词典
segmentor=Segmentor()
segmentor.load_with_lexicon(cws_model_path,'data/lexicon')
# segmentor.load(cws_model_path)
postagger = Postagger() # 初始化实例
postagger.load(pos_model_path)  # 加载模型




'''
 词性：
 nh	person name	杜甫, 汤姆
 nz 其他专名：如 诺贝尔奖
 ns 地名
 ws 英文名称
 n 名词
 j 简称
 ni 机构名称
 r 代词：后来的我们
 i 习语,方言
 nt 时间
 a 形容词
 m 数位词
 v 动词
'''

allow_pos={'nh':1,'nz':2,'ns':3,'ws':4,'n':5,',r':6,'j':7,'i':8,'nt':9,'a':10,'m':11,'v':12}
all_docs_file=open('data/all_docs.txt','r',encoding='utf-8')
train_docs_file=open('data/train_docs_keywords.txt','r',encoding='utf-8')

def get_keyword_by_id(train_ids,keywords,doc_id):
    """
    根据doc的id提取从训练集返回正确的关键词
    :return:
    """

    # print(data[1].strip().split(',')," ".join(postagger.postag(data[1].strip().split(','))))
    dict_word=[word for keyword in keywords for word in keyword ]
    with open('data/lexicon','w',encoding='utf-8') as dict_file:
        for word in dict_word:
            dict_file.write(word+'\n')


    if doc_id in train_ids:
        # print(doc_id,train_ids.index(doc_id))
        return keywords[train_ids.index(doc_id)]
    return None



# 基于jieba的textrank提取关键词
def extract_keyword_by_ltppossag():
    ids=[]
    labels_1=[]
    labels_2=[]

    # 训练集中的ids,keywords
    train_ids = []
    train_keywords = []
    for line in train_docs_file:
        data = line.split('\t')
        train_ids.append(data[0].strip())
        train_keywords.append(data[1].strip().split(','))
    print(len(train_ids),len(train_keywords))

    for line in tqdm(all_docs_file):
        all_temp_keywords=[]
        data=line.strip().split('')
        ids.append(data[0].strip())
        words=segmentor.segment(data[1])
        postags=postagger.postag(words)
        # print("|".join(words)," ".join(postags))
        train_keyword=get_keyword_by_id(train_ids,train_keywords,doc_id=data[0].strip())

        if train_keyword:
            if len(train_keyword)>=2:
                labels_1.append(train_keyword[0])
                labels_2.append(train_keyword[1])
            else:
                labels_1.append(train_keyword[0])
                labels_2.append('')
        else:

            for word_pos in zip(words,postags):
                if word_pos[1] in allow_pos:
                    all_temp_keywords.append(word_pos)
            all_temp_keywords=sorted(all_temp_keywords, reverse=False,key=lambda x:allow_pos[x[1]])

            if len(all_temp_keywords)<2:
                labels_1.append(all_temp_keywords[0][0])
                labels_2.append('')
            else:
                labels_1.append(all_temp_keywords[0][0])
                labels_2.append(all_temp_keywords[1][0])
    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}

    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/05_ltp_postag.csv', index=False)

    # 关闭文件
    all_docs_file.close()
    train_docs_file.close()

extract_keyword_by_ltppossag()

# keyword=get_keyword_by_id(doc_id='D012650')
# print(keyword)


