# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 07_w2v_rf.py
# @Time    : 2018/9/6 15:51

"""
1.用jieba切词，去除停用词，去除非中文的词语
2.用word2vec将每篇文本中的所有词语转化为词向量
3.对于每篇文本，生成其词向量矩阵和词语类标矩阵（两类，关键词1，非关键词0）
4.将词向量矩阵和类标矩阵传入分类器进行二分类（这里选用随机森林决策树），提取出关键词（预测类标为1），作为预测关键词
"""
import pickle
import codecs
from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from jieba import posseg
from sklearn import cross_validation
import collections
import math
import jieba
from tqdm import tqdm
import pandas as pd

jieba.load_userdict('data/custom_dict.txt')  # 设置词库
stop_words = open('data/stop_words.txt', 'r', encoding='utf-8').read().split('\n')
allow_pos={'nr':1,'nz':2,'ns':3,'nt':4,'eng':5,'n':6,'l':7,'i':8,'a':9,'nrt':10,'v':11,'t':12}

train_data=pd.read_csv('data/new_train_docs.csv',sep=',')
true_keywords=train_data['keyword'].apply(lambda x:x.split(',')).tolist()
titles=train_data['title'].tolist()
docs=train_data['doc'].tolist()

all_docs=[title+" "+doc for title,doc in zip(titles,docs)]

# jieba 分词
def cut_words(text):
    words=[word for word,tag in posseg.cut(text) if word not in stop_words]
    return words

    # words = []
    # for word, tag in posseg.cut(text):
    #     if word not in stop_words and tag in allow_pos:
    #         words.append(word)
    # return words
# 随机森林决策树
def rf():
    clf=RandomForestClassifier()
    return clf


def get_tfidf():
    tf_idf=[]

    doc_num=len(all_docs)
    doc_count=0

    for doc in tqdm(all_docs):
        doc_count+=1
        doc_words_list=cut_words(doc)
        words_num=len(doc_words_list)
        temp_dict=collections.Counter(doc_words_list)
        words_list=list(temp_dict.keys())

        for i in range(len(words_list)):
            temp=[]
            word=words_list[i]
            # 计算一个词的tf
            word_tf=float(temp_dict[word])/float(words_num)
            # 计算一个词的idf
            text_contain_num=0
            for doc in all_docs:
                if word in doc:
                    text_contain_num+=1
            word_idf=math.log(float(doc_num)/float(text_contain_num+1))

            temp.append(doc_count)
            temp.append(word)
            temp.append(word_tf*word_idf)
            tf_idf.append(temp)
    return tf_idf


# 自动提取关键词
def get_txt_keywords():
    # 获取整个语料库所有文档中每个词语的tfidf权重
    word_tf_idf = get_tfidf()
    # 数据结构化
    # 将文本及其对应的关键词写入本地内存
    # txtList是一维数组，用于存储文本
    # kwList是二维数组，用于存储关键词，其内每个小数组都是一篇文本的关键词，与txtList一一对应
    txt_list, kw_list = [], true_keywords
    whole_txt = ''  # 语料库所有文本合在一起
    for doc in all_docs:
        whole_txt += doc.strip()
        txt_list.append(doc.strip())

    # 读取整个语料库中所有文本，切词，做word2vec，生成词向量
    whole_words = cut_words(whole_txt)  # 语料库所有词语
    # 建模
    model = Word2Vec([whole_words], min_count=1)
    # 存储语料库中每个词语对应的词向量
    word_vecd = {}
    uniq_words = list(frozenset(whole_words))
    for i in range(len(uniq_words)):
        word = uniq_words[i]
        word_vecd[word] = model[word]
    re_list, accu_list = [], []  # 用于存储每篇文本的召回率和准确率
    clf = rf()
    # 依次遍历每一篇文本
    score=0
    for i in range(len(txt_list)):
        now_doc_kw = kw_list[i]  # 当前文本的原始关键词列表
        now_doc_words = cut_words(txt_list[i])  # 当前文本的所有词语
        doc_word_vec, docWordL = [], []
        for j in range(len(now_doc_words)):
            docWord = now_doc_words[j]
            # 生成每篇文本的词向量矩阵，其中一行为一个词向量
            doc_word_vec.append(list(word_vecd[docWord]))
            # 生成每篇文本的类标向量，其中一个元素对应一个词语的类标，关键词类标1，非关键词类标0
            if docWord in now_doc_kw:
                docWordL.append(1)
            else:
                docWordL.append(0)
        docWordM = np.array(doc_word_vec)
        # 对当前文本词语做关键词和非关键词的二分类（k折交叉验证）
        preLabel = cross_validation.cross_val_predict(clf, docWordM, docWordL, cv=5)  # 预测样本类别
        sKW = set()  # 用于存储当前文本预测出来的关键词并去重
        for r in range(len(preLabel)):
            if preLabel[r] == 1:
                sKW.add(now_doc_words[r])
        predictKw = list(sKW)  # 预测的关键词
        # 计算原始标记关键词和预测关键词之间的交集，即预测中多少个
        countKw = 0
        for k1 in range(len(now_doc_kw)):
            for k2 in range(len(predictKw)):
                if now_doc_kw[k1] == predictKw[k2]:
                    countKw += 1
                    score+=0.5
                    break
        # 计算召回率
        recallRate = float(countKw) / float(len(now_doc_kw))
        # 计算准确率
        accuracyRate = float(countKw) / float(len(predictKw) + 1)
        re_list.append(recallRate)
        accu_list.append(accuracyRate)
        # 获取预测关键词的tfidf权重，根据权重降序排列输出预测关键词
        preKwTFIDF = []
        for p in range(len(predictKw)):
            preWord = predictKw[p]
            for t in range(len(word_tf_idf)):
                if word_tf_idf[t][0] == i + 1 and word_tf_idf[t][1] == preWord:
                    preKwTFIDF.append(word_tf_idf[t][2])
                    break
        wordTFIDFD = {}
        for ww in range(len(preKwTFIDF)):
            wordTFIDFD[preKwTFIDF[ww]] = predictKw[ww]
        preKwTFIDF.sort(reverse=True)
        print('---------第' + str(i + 1) + '篇文本结果---------')
        print('【一】原始标记的关键词')
        print(now_doc_kw)
        print('【二】预测关键词按照tfidf权重降序排列')
        for www in range(len(preKwTFIDF)):
            print(wordTFIDFD[preKwTFIDF[www]], preKwTFIDF[www])
        print('【三】预测中几个关键词')
        print(countKw)
    # 输出评估结果
    print('=======================================================================')
    print('Mean Recall Rate:', float(sum(re_list)) / float(len(re_list)))
    print('Mean Accuracy Rate:', float(sum(accu_list)) / float(len(accu_list)))
    print("最终得分为：",score)

if __name__ == '__main__':
    get_txt_keywords()
