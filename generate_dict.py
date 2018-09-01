# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 10:37
# @Author  : quincyqiang
# @File    : generate_dict.py
# @Software: PyCharm
# 构建jieba自定义词典

"""
 nr 人名
 nz 其他专名
 ns 地名
 nt 机构团体
 n 名词
"""
import pandas as pd
import random
import re
from tqdm import tqdm
custom_dict_file=open('data/custom_dict.txt','a',encoding='utf-8')


def get_keyword():
    """
    从new_train_docs.csv（只需要keyword）关键字，标注为nz(不一定合理....) 词频随机设置（10-20）
    :return:
    """
    train_data=pd.read_csv('data/new_train_docs.csv')
    keywords=train_data['keyword'].apply(lambda x:x.split(',')).tolist()
    keywords=[word  for keyword in keywords for word in keyword]
    for keyword in keywords:
        custom_dict_file.write('{0} {1} nz\n'.format(keyword,str(random.randint(10,20))))

def get_tag_word():
    """
    提取《》、【】,“”中的专有名词：test_docs.csv
    :return:
    """
    test_data=pd.read_csv('data/test_docs.csv')
    titles=test_data['title'].tolist()
    pattern=re.compile(r'《(.+)》|【(.+)】|“(.+)”')
    for title in tqdm(titles):
        keywords=re.findall(pattern,title)
        if keywords:
            print(keywords)
            print(title)


if __name__ == '__main__':
    # get_keyword()
    get_tag_word()
    custom_dict_file.close()