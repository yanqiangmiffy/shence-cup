# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : demo.py
# @Time    : 2018/9/5 13:26
import jieba.posseg
jieba.load_userdict('data/custom_dict.txt')
text='《神奇女侠2》正式开机！盖尔·加朵与派派现身片场有说有笑 '
for word,tag in jieba.posseg.cut(text):
    print(word,tag)