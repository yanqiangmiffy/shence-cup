# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : lda_topic.py
# @Time    : 2018/9/7 16:51

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import pickle
import pandas as pd
import gensim
from gensim import corpora,models
train_data=pd.read_csv('../data/new_train_docs.csv')
true_keywords=train_data['keyword'].apply(lambda x:x.split(',')).tolist()
titles=train_data['title'].tolist()

# 1 加载语料
with open('../data/train_docs.pkl','rb') as in_data:
    train_docs=pickle.load(in_data)

train_docs=[[word for word in doc.split(' ')] for doc in train_docs]
print(train_docs[0])
# 判断关键字的相似度
model = gensim.models.Word2Vec(train_docs, size=1000)
print(model.most_similar('水木年华', topn=10))

dictionary = corpora.Dictionary(train_docs)
print(dictionary)
corpus = [dictionary.doc2bow(text) for text in train_docs]
print(corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# 根据结果使用lsi做主题分类效果会比较好
print('#############' * 4)
lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(2)
for doc in corpus_lsi:
    print(doc)

print('#############' * 4)
lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, update_every=0, passes=1)
corpus_lda = lda[corpus_tfidf]
lda.print_topics(2)
for doc in corpus_lda:
    print(doc)