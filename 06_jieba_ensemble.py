# -*- coding: utf-8 -*-
# @Time    : 2018/9/1 21:57
# @Author  : quincyqiang
# @File    : 06_jieba_ensemble.py
# @Software: PyCharm

import pandas as pd
from tqdm import tqdm
from jieba.analyse import extract_tags,textrank # tf-idf
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

test_data=pd.read_csv('data/test_docs.csv')
train_data=pd.read_csv('data/new_train_docs.csv')
allow_pos={'nr':1,'nz':2,'ns':3,'nt':4,'eng':5,'n':6,'l':7,'i':8,'a':9,'nrt':10,'v':11,'t':12}
tf_pos = ['ns', 'n', 'vn', 'nr', 'nt', 'eng', 'nrt']

def generate_name(word_tags):
    name_pos = ['ns', 'n', 'vn', 'nr', 'nt', 'eng', 'nrt']
    for word_tag in word_tags:
        if word_tag[0] == '·':
            index = word_tags.index(word_tag)
            if (index+1)<len(word_tags):
                prefix = word_tags[index - 1]
                suffix = word_tags[index + 1]
                if prefix[1] in name_pos and suffix[1] in name_pos:
                    name = prefix[0] + word_tags[index][0] + suffix[0]
                    word_tags = word_tags[index + 2:]
                    word_tags.insert(0, (name, 'nr'))
    return word_tags



def extract_keyword_ensemble(test_data):

    ids,titles,docs=test_data['id'],test_data['title'],test_data['doc']

    labels_1 = []
    labels_2 = []
    empty=0
    for title,doc in tqdm(zip(titles,docs)):
        keywords = []
        word_tags=[(word,pos) for word,pos in posseg.cut(title)] # 标题

        # 判断是否存在特殊符号
        if '·' in title:
            word_tags = generate_name(word_tags)

        for word_pos in word_tags:
            if word_pos[1] in allow_pos:
                keywords.append(word_pos)

        # 根据看关键词筛选
        keywords=[keyword for keyword in keywords if len(keyword[0])>1]

        # 先按词性排序，再按长度排序
        keywords = sorted(keywords, reverse=False, key=lambda x: (allow_pos[x[1]],-len(x[0])))
        # print(keywords)

        if len(keywords) <2:
            # 使用tf-idf
            empty+=1
            temp_keywords = [keyword for keyword in extract_tags(title+str(doc)[:100]+str(doc)[:-50],topK=5)]
            # print("tfidf:",temp_keywords)
            labels_1.append(temp_keywords[0])
            labels_2.append(temp_keywords[1])
        else:
            labels_1.append(keywords[0][0])
            labels_2.append(keywords[1][0])

    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}

    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/06_jieba_ensemble.csv', index=False)
    print("使用tf-idf提取的次数：",empty)



def evaluate():
    ids, titles, docs = train_data['id'], train_data['title'], train_data['doc']
    true_keywords=train_data['keyword'].apply(lambda x:x.split(','))
    labels_1 = []
    labels_2 = []

    use_idf,part_wrong= 0,0

    score=0
    for data in tqdm(zip(titles, docs,true_keywords)):
        title=data[0]
        doc=data[1]

        keywords = []
        word_tags = [(word, pos) for word, pos in posseg.cut(title)]  # 标题
        # 判断是否存在特殊符号
        if '·' in title:
            word_tags=generate_name(word_tags)

        for word_pos in word_tags:
            if word_pos[1] in allow_pos:
                keywords.append(word_pos)

        keywords = [keyword for keyword in keywords if len(keyword[0]) > 1]
        keywords = sorted(keywords, reverse=False, key=lambda x: (allow_pos[x[1]], -len(x[0])))

        true_keys=data[2]
        if len(keywords) < 2:
            # 使用tf-idf
            use_idf += 1
            temp_keywords = [keyword for keyword in
                             extract_tags(title + str(doc)[:100]+str(doc)[:-80],topK=2)]
            # print("tfidf:",temp_keywords)
            labels_1.append(temp_keywords[0])
            labels_2.append(temp_keywords[1])
            # print(temp_keywords[0],temp_keywords[1],data[2])
            if temp_keywords[0] in true_keys:
                score+=0.5
            if temp_keywords[1] in true_keys:
                score+=0.5
        else:
            key_1 = keywords[0][0]
            key_2 = keywords[1][0]

            if key_1 not in true_keys or key_2 not in true_keys:
                part_wrong+=1
                # print("prediction--true keys--title--candidate keys")
                temp_keywords = [keyword for keyword in
                                 extract_tags(title + str(doc),topK=2)]

                print((key_1,key_2),'--',temp_keywords,'--',true_keys,'--',title,'--',keywords)

            if key_1 in true_keys:
                score+=0.5
            if key_2 in true_keys:
                score+=0.5
            labels_1.append(key_1)
            labels_2.append(key_2)

    data = {'id': ids,
            'label1': labels_1,
            'label2': labels_2}
    df_data = pd.DataFrame(data, columns=['id', 'label1', 'label2'])
    df_data.to_csv('result/06_train.csv', index=False)
    print("使用tf-idf提取的次数：", use_idf)
    print("预测出错的次数：",part_wrong)
    print("最终得分为：",score)


if __name__ == '__main__':
    # extract_keyword_ensemble(test_data)
    evaluate()