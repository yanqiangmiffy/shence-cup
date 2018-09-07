# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : demo.py
# @Time    : 2018/9/7 14:20
import jieba
jieba.load_userdict('data/custom_dict.txt')

for word in jieba.cut('哈里斯·迪金森加盟《沉睡魔咒2》饰演王子'):
    print(word)