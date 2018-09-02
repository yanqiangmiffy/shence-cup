# -*- coding: utf-8 -*-
# @Time    : 2018/9/2 21:01
# @Author  : quincyqiang
# @File    : utils.py
# @Software: PyCharm
custom_dict_file=open('data/custom_dict.txt','r',encoding='utf-8')
def process():
    """
    去除长度为1的word
    :return:
    """
    with open('data/perfect.txt','w',encoding='utf-8') as new_file:
        for line in custom_dict_file:
            data=line.strip().split(' ')
            new_file.write(" ".join(data[:2])+"\n")
if __name__ == '__main__':
    process()