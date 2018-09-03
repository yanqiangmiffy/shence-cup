# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 9:37
# @Author  : quincyqiang
# @File    : preprocess.py
# @Software: PyCharm
# 预处理：1、从all_docs去除训练集中的数据 2、从补全train_doc的数据
import pandas as pd


def processed():
    all_data=pd.read_csv('data/all_docs.txt',sep='\001',header=None)
    all_data.columns=['id','title','doc']
    train_data=pd.read_csv('data/train_docs_keywords.txt',sep='\t',header=None)
    train_data.columns=['id','keyword']

    # 从all_docs筛选出train_docs的标题和文章
    train_id_list=list(train_data['id'].unique())
    train_title_doc=all_data[all_data['id'].isin(train_id_list)]
    new_train_data=pd.merge(train_data,train_title_doc,on=['id'])
    # 这里按id排下序，将通一种类的文章聚合在一起
    new_train_data=new_train_data.sort_values(by='id',ascending=True)
    new_train_data.to_csv('data/new_train_docs.csv',index=False)

    # 从all_docs去除train_docs的数据：应为对评分没有用
    test_title_doc = all_data[~all_data['id'].isin(train_id_list)]
    print(test_title_doc.shape[0])
    test_title_doc.to_csv('data/test_docs.csv',index=False)

if __name__ == '__main__':
    processed()