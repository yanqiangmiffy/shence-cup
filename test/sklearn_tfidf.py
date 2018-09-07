# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : sklearn_tfidf.py
# @Time    : 2018/9/7 11:43
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import pickle
import pandas as pd

train_data=pd.read_csv('../data/new_train_docs.csv')
true_keywords=train_data['keyword'].apply(lambda x:x.split(',')).tolist()
titles=train_data['title'].tolist()

# 1 加载语料
with open('../data/train_docs.pkl','rb') as in_data:
    train_docs=pickle.load(in_data)

def train(docs):
    # 2 将语料转换为词袋向量
    cv = CountVectorizer(min_df=1, max_df=1.0)

    word_count_vector=cv.fit_transform(docs)

    # print(feature_names)

    # 3 根据词袋向量统计TF-IDF
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)
    # for idx, word in enumerate(feature_names):
    #   print("{}\t{}".format(word, tfidf_transformer.idf_[idx]))
    return cv,tfidf_transformer


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)

    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):

    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []

    # idx 为单词索引 score为对应的tf-idf值
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results



def predict(doc,cv,tfidf_transformer,feature_names):
    # 4 加载一篇文档的tf-idf 增加标题和摘要部分权重
    title_text=" ".join(doc.split(' ')[:10])*2
    text=doc+title_text
    tf_idf_vector=tfidf_transformer.transform(cv.transform([text]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
    tfidf_key=list(keywords.keys())
    key_1,key_2=tfidf_key[0],tfidf_key[1]
    return key_1,key_2,keywords


cv,tfidf_transformer=train(train_docs)
feature_names = cv.get_feature_names()


score=0
not_right=0
for doc,true_key in zip(train_docs,true_keywords):
    key_1, key_2,keywords = predict(doc, cv, tfidf_transformer, feature_names)


    if key_1 in true_key:
        score+=0.5
    if key_2 in true_key:
        score+=0.5
    if key_1 not in true_key or key_2 not in true_key:
        print((key_1,key_2),true_key,keywords)
        not_right  +=1
print("最终得分：{}".format(score))
print("不正确：{}".format(not_right))