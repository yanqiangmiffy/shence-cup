# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : lda_topic.py
# @Time    : 2018/9/7 16:51

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import pickle
import pandas as pd

from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary

train_data=pd.read_csv('../data/new_train_docs.csv')
true_keywords=train_data['keyword'].apply(lambda x:x.split(',')).tolist()
titles=train_data['title'].tolist()

# 1 加载语料
with open('../data/train_docs.pkl','rb') as in_data:
    train_docs=pickle.load(in_data)

train_docs=[[word for word in doc.split(' ')] for doc in train_docs]


dictionary = corpora.Dictionary(train_docs)
corpus = [dictionary.doc2bow(text) for text in train_docs]
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=6)

topic_list = lda.print_topics(20)
# print(type(lda.print_topics(20)))


for topic in topic_list:
    print(topic)
print("第一主题",lda.print_topic(1))
print('给定一个新文档，输出其主题分布')

# test_doc = list(new_doc) #新文档进行分词
test_doc = train_docs[2]  # 查看训练集中第三个样本的主题分布
print(test_doc)
doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
doc_lda = lda[doc_bow]  # 得到新文档的主题分布
# 输出新文档的主题分布
print(doc_lda)

for topic in doc_lda:
    print("%s\t%f\n" % (lda.print_topic(topic[0]), topic[1]))
