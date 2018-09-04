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



# from itertools import chain
# a=chain([1,2,4],['a','c',1])
# for data in a:
#     print(data)

import jieba.posseg
# text="华为新机皇P30pro曝光"
# text="【创新菜】法式红酒烩鸡肉,"
# text="【菜谱】夏日美味小点心——绿豆冰糕"
# text="老公做了一桌菜，色香味俱全，婆婆一回来，看见就不高兴"
# text='"巨舰,航母",我海军一款4万吨巨舰作用比肩航母未来至少需要12艘'
# text='亚丁湾上练兵忙'
text='孕5月一小动作险致流产，孕期别碰这部位，99%孕妇不知'
for word,tag in jieba.posseg.cut(text):
    print(word,tag)

# sen='UPDATE staff_table SET dept="Market" WHERE where dept="IT"' # 提取引号中的内容
# import re
# mth=re.findall('"(.*?)"',sen)
# for m in mth:
#     print(m)
#
# print('我\001爱你')

documents = ['孕妇 慎用 塑料容器 香水 损害 胎儿 大脑 发育 环球网 英国 邮报 日报 美国 人员 得出结论 孕期 塑料容器 香水 胎儿 大脑 发育 损害 专家建议 孕妇 类型 塑料制品 香水 伊利诺伊大学 神经科学 心理学 教授 尼斯 拉斯卡 JaniceJuraska 博士 耗时 激素 人类 大脑 发育 苏珊 桑茨 SusanSchantz 博士 致力于 产品 含有 化学物质 内分泌 干扰 已知 邻苯二甲酸 含有 化学物质 荷尔蒙 分泌 拉斯卡 博士 团队 实验 含有 物质 饼干 怀孕 小白鼠 孕期 分娩 喂食 饼干 观察 观察 发现 物质 母体 老鼠 后代 相当于 认知 能力 暴露 化学物质 同龄 迟缓 孕期 塑料容器 香水 胎儿 大脑 发育 损害 拉斯卡 博士 计算 小鼠 前额 皮层 神经元 突触 注意力 计划 协调 控制 大脑 区域 计算结果 显示 胚胎 暴露 化学物质 老鼠 神经元 突触 减少 拉斯卡 博士 桑茨 博士 致力于 邻苯二甲酸 额叶 皮层 损害 成熟期 大鼠 注射 化学物质 化学物质 神经 发育 拉斯卡 博士 禁止 化学品 孕妇 怀孕 哺乳 含有 化学物质 东西 塑料容器 香水 程度 减少 母体 胎儿 暴露 有害物质 可能性 实习 编译 审稿 刘洋',
             '我 经常 北京1 天安门1 在 广场 拍照']
from sklearn.feature_extraction.text import TfidfVectorizer
global_tfidf_vecc = TfidfVectorizer()
global_count_data = global_tfidf_vecc.fit_transform(documents)
print(global_count_data, global_count_data.shape, type(global_count_data))
# count_array = count_data.toarray()
# print(count_array, count_array.shape, type(count_data))
# print('词汇表为：\n', tfidf_vecc.vocabulary_)

# 统计词频

from sklearn.feature_extraction.text import CountVectorizer
# cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
cv=CountVectorizer(max_df=0.85,ngram_range=(2,2))
word_count_vector=cv.fit_transform(documents)

# 计算tfidf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

# 提取关键词
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

# you only needs to do this once
feature_names=cv.get_feature_names()

# get the document that we want to extract keywords from
doc=documents[0]

#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)

# now print the results
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])


# data={'a':1,'b':2}
# print(list(data.keys()))
jieba.load_userdict('data/custom_dict.txt')
print("-".join(jieba.cut('039B潜舰近照曝光!克服「AIP系统问题」停产再量产')))
# for word ,tag in jieba.posseg.cut('黄渤托举高秋梓'):
#     print(word,tag)

# allow_pos={'nr':1,'nz':2,'ns':3,'nt':4,'eng':5,'n':6,'l':7,'i':8,'a':9,'nrt':10,'v':11,'t':12}
#
# keywords=[('陈烈', 'nr'), ('导演', 'nz'), ('电影', 'nz'), ('爱是永恒', 'nz'), ('院线', 'n'), ('著名', 'a'), ('上映', 'v')]
# keywords = sorted(keywords, reverse=False, key=lambda x: (allow_pos[x[1]],-len(x[0])))
# print(keywords)

# all_pos=['ns', 'n', 'vn','nr','nt','eng','l','i','a','nrt']
#
# # word_tags=[('谢琳', 'nr'), ('·', 'x'), ('伍德蕾', 'nr'), ('新片', 'nz'), ('确定', 'v'), ('编剧', 'n'), ('曾', 'd'), ('合作', 'vn'), ('《', 'x'), ('青春密语', 'nz'), ('》', 'x')]
# word_tags=[('哈里斯', 'nrt'), ('·', 'x'), ('迪金森', 'nr'), ('加盟', 'nz'), ('《', 'x'), ('沉睡魔咒2', 'nz'), ('》', 'x'), ('饰演', 'n'), ('王子', 'nr')]
# for word_tag in word_tags:
#
#     if word_tag[0]=='·':
#         index=word_tags.index(word_tag)
#         prefix=word_tags[index-1]
#         suffix=word_tags[index+1]
#         if prefix[1] in all_pos and suffix[1] in all_pos:
#             name=prefix[0]+word_tags[index][0]+suffix[0]
#             word_tags=word_tags[index+2:]
#             word_tags.insert(0,(name,'nr'))
#
# print(word_tags)
