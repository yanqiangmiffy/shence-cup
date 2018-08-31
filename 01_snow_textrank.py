# -*- coding: utf-8 -*-
# @Author  : quincyqiang
# @File    : 01_snow_textrank.py
# @Time    : 2018/8/30 17:17

import pandas as pd
from tqdm import tqdm
from snownlp import SnowNLP
import random
# 加载数据
all_docs_file=open('data/all_docs.txt','r',encoding='utf-8')

ids=[]
labels_1=[]
labels_2=[]
for line in tqdm(all_docs_file):
    data=line.strip().split('')
    ids.append(data[0])
    # text=data[1] + data[2].replace('\xa0', '').replace('\u3000', '')
    text=data[1]
    snow=SnowNLP(text)
    keyword=snow.keywords(limit=5)
    if len(keyword)>2:
        labels_1.append(keyword[0])
        labels_2.append(keyword[1])
    if len(keyword)==1:
        labels_1.append(keyword[0])
        labels_2.append(snow.words[1])
    if len(keyword)<1:
        if len(snow.words)>1:
            labels_1.append(snow.words[0])
            labels_2.append(snow.words[1])

        else:
            print(data[1])
            labels_1.append(snow.words[0])
            labels_2.append('')
# import csv
# with open('result/01_snow_textrnak.csv','w',encoding='utf-8',newline='') as out_data:
#     csv_writer=csv.writer(out_data)
#     for id,label1,label2 in zip(ids,labels_1,labels_2):
#         csv_writer.writerow((id,label1,label2))

data={'id':ids,
      'label1':labels_1,
      'label2':labels_2}
df_data=pd.DataFrame(data,columns=['id','label1','label2'])
df_data.to_csv('result/01_textrank1.csv',index=False)

# text="《三生三世》里凤九东华最有“夫妻相”，这一模一样的动作是证明。电视剧《三生三世十里桃花》结束了，接下来它的姊妹篇《三生三世枕上书》又要开拍了。上一部讲述的是杨幂出演的白浅与赵又廷出演的夜华。而这一部却讲述了迪丽热巴出演的白凤九与高伟光出演的东华帝君。前一部里面还是两大配角，这一部剧却成了主角。不过她们的故事在《三生三世十里桃花》里面已经讲述了一大半了，再到《三生三世枕上书》里面来再经历一次三生三世，也着实让人为剧情捏了一把汗。不得不说的是，就算是原著作品，《三生三世十里桃花》的剧情构造都是要比《三生三世枕上书》来的精彩的。而且很多情节，都有“借鉴”他人作品的嫌疑。原班人马出发，固然是对喜爱这部作品的观众的尊重，但是对于演员本身却没有多大的好处。就像刚刚公布男女主阵容的时候，迪丽热巴的粉丝就产生了极大的抵触，认为她家公司只知道圈钱，却不知道为迪丽热巴多接一些好的作品，毕竟迪丽热巴同类型的古装剧实在是太多了，需要一部真正的大女主，来巩固一线位置。做人气小花容易，做长久的女演员却是很难的。不过，再看《三生三世十里桃花》这一部无论口碑还是制作都是上良的剧集，不得不说，编剧在很多小细节上面的把控是很到位的。今天就来说一说，凤九与东华之间那些暗搓搓的秀恩爱。《三生三世》里凤九东华最有“夫妻相”，这个动作简直粘贴复制。首先这一组托晒沉思的动作，简直是粘贴复制的存在啊！还有凤九受伤之后的拥抱，东华帝君的臂膀太宽大了，感觉抱着一个小孩一样，很宠溺，很甜有没有？东华为了还凤九一片深情，知道自己无法动情的情况下，估计设计了这凡间一世的情缘，皇帝与贵妃的设置也是配一脸。还有这初次相见的对视，简直是注定了的情缘，这身高差也是没谁了。希望《三生三世枕上书》的剧情能改编的好一些吧！毕竟俊男靓女的组合，再加上制作团队曾经打造过如此华美的《三生三世十里桃花》，只要剧情配得上观众的期待，相信一样能爆红的。凤九东华的三世情缘，又将是怎样一番虐恋情深呢？"
# text="【味集】南宁美食圣地最强攻略！老南宁最爱的味道，都汇集在这些地方"
# # text="《三生三世》里凤九东华最有“夫妻相”，这一模一样的动作是证明"
# text="你是个疯子？这么巧，我也是！"

# snow=SnowNLP(text)
# keyword=snow.keywords(limit=5)
# print(keyword[:2])
# print(snow.words)