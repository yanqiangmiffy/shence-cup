# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 9:37
# @Author  : quincyqiang
# @File    : preprocess.py
# @Software: PyCharm
# 预处理：1、从all_docs去除训练集中的数据 2、从补全train_doc的数据
import os
from collections import Counter
import pandas as pd
from tqdm import tqdm
from jieba import posseg
import pickle
import jieba
jieba.load_userdict('data/custom_dict.txt')  # 设置词库


def txt2csv():
    all_data=pd.read_csv('data/all_docs.txt',sep='\001',header=None)
    all_data.columns=['id','title','doc']
    all_data.to_csv('data/all_docs.csv',index=False)
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


def generate_name(word_tags):
    """
    解决分词缺陷：杰森·斯坦森
    :param word_tags:
    :return:
    """
    name_pos = ['ns', 'n', 'vn', 'nr', 'nt', 'eng', 'nrt']
    for word_tag in word_tags:
        if word_tag[0] == '·' or word_tag[0] == '！':
            index = word_tags.index(word_tag)
            if (index+1)<len(word_tags):
                prefix = word_tags[index - 1]
                suffix = word_tags[index + 1]
                if prefix[1] in name_pos and suffix[1] in name_pos:
                    name = prefix[0] + word_tags[index][0] + suffix[0]
                    word_tags = word_tags[index + 2:]
                    word_tags.insert(0, (name, 'nr'))
    return word_tags


def generate_tokenized_doc(data_path,df_data,stop_words=(),allow_pos=()):
    """
        标题和文章分句 重要词性组成
        :return:
        """
    ids, titles, docs = df_data['id'], df_data['title'], df_data['doc']
    print("generate docs..",data_path)
    all_docs = []
    txt_file= open(data_path + '.txt', 'w', encoding='utf-8')
    for title,doc in tqdm(zip(titles, docs)):

        doc = str(title) + '。' + str(doc)
        word_tags = []
        for word, pos in posseg.cut(doc):
            if word not in stop_words and pos in allow_pos:
                if len(word) > 1 and len(word)<10:
                    word_tags.append((word, pos))
        # 提取特殊名字
        # if '·' in title or '！' in title:
        #     word_tags = generate_name(word_tags)
        #     print(word_tags)
        new_doc = " ".join([word_tag[0] for word_tag in word_tags])
        # print(new_doc)
        # 保存分词好的数据
        all_docs.append(new_doc)
        txt_file.write(new_doc+'\n')

    with open(data_path, 'wb') as out_data:
        pickle.dump(all_docs, out_data, pickle.HIGHEST_PROTOCOL)
    return all_docs

if __name__ == '__main__':
    # txt2csv()
    test_data_path = 'data/test_doc.pkl'
    train_data_path = 'data/train_docs.pkl'
    test_data = pd.read_csv('data/test_docs.csv')
    train_data = pd.read_csv('data/new_train_docs.csv')
    allow_pos = {'nr': 12, 'nz': 11, 'ns': 10, 'nt': 9, 'eng': 8, 'l': 7,
                 'i': 6, 'a': 5, 'nrt': 4, 'n': 3, 'v': 2, 't': 1}
    stop_words = open('data/stop_words.txt', 'r', encoding='utf-8').read().split('\n')

    x = generate_tokenized_doc(train_data_path, train_data, stop_words, allow_pos)
    x1 = generate_tokenized_doc(test_data_path, test_data, stop_words, allow_pos)