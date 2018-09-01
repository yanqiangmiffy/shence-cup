# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : utils.py
# @Time    : 2018/8/31 15:23

import jieba.posseg
# text="华为新机皇P30pro曝光"
# text="【创新菜】法式红酒烩鸡肉,"
# text="【菜谱】夏日美味小点心——绿豆冰糕"
# text="老公做了一桌菜，色香味俱全，婆婆一回来，看见就不高兴"
text='"巨舰,航母",我海军一款4万吨巨舰作用比肩航母未来至少需要12艘'
for word,tag in jieba.posseg.cut(text):
    print(word,tag)

# sen='UPDATE staff_table SET dept="Market" WHERE where dept="IT"' # 提取引号中的内容
# import re
# mth=re.findall('"(.*?)"',sen)
# for m in mth:
#     print(m)
#
# print('我\001爱你')