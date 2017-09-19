# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 23:20:27 2017

@author: Administrator
"""

import gensim
import os
from os import listdir
from os.path import isfile, join

import pandas as pd
import jieba
from gensim.models import word2vec

mydir = 'C:\\Users\\Administrator\\Desktop\\industry_classification\\MachineLearning\\text\\'

f1 =open(mydir+"鹿鼎记.txt", 'r', encoding='utf-8')
f2 =open(mydir+"result.txt", 'a', encoding='utf-8')  
lines =f1.readlines()  # 读取全部内容  
for line in lines:  
    line.replace('\t', '').replace('\n', '').replace(' ','')  
    seg_list = jieba.cut(line, cut_all=False)
    f2.write(" ".join(seg_list))  
   
f1.close()
f2.close()


sentences = word2vec.Text8Corpus(mydir+'result.txt')  # 加载语料  
model = word2vec.Word2Vec(sentences, size=200)  #训练skip-gram模型，默认window=5  
   
print (model)
# 计算两个词的相似度/相关程度  
try:  
    y1 = model.similarity(u"韦小宝", u"宝剑")  
except KeyError:  
    y1 = 0  
#  
# 计算某个词的相关词列表  
y2 = model.most_similar(u"韦小宝", topn=20)  # 20个最相关的  
 
   
# 寻找对应关系  
y3 =model.most_similar([u'康熙', u'茅十八'], [u'书'], topn=3)  
for item in y3:  
    print (item[0], item[1])  


# 寻找不合群的词  
y4 =model.doesnt_match(u"书 书籍 教材 很".split())  
print (u"不合群的词："), y4  
print ("-----\n")  
   
# 保存模型，以便重用  
model.save(u"书评.model")  
# 对应的加载方式  
# model_2 =word2vec.Word2Vec.load("text8.model")  
   
# 以一种c语言可以解析的形式存储词向量  
#model.save_word2vec_format(u"书评.model.bin", binary=True)  
# 对应的加载方式  
# model_3 =word2vec.Word2Vec.load_word2vec_format("text8.model.bin",binary=True)