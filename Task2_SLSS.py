#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/20 19:09
# @Author  : ELP
# @Site    : 
# @File    : Task2_SLSS.py
# @Software: PyCharm

from gensim.models import keyedvectors
import numpy as np
import time
import logging

'''
train word2vec and etc,all need extra files in the following link:
link: https://pan.baidu.com/s/1kVtDnRh password: rs71
-----------------------------------------------
To run this python file, please:
1.install gensim,numpy,time; load python 3.0
2.add path of each data file
3.add your own path of results conservation
-----------------------------------------------
Time(test): it takes about 60s to run off this py on ubuntu 16
'''

def select_tag2(l1, l2, l3):#chose the top3 tags
    lz=[]
    if l1!=[]:
        lz.append(l1)
    if l2 != []:
        lz.append(l2)
    if l3 != []:
        lz.append(l3)
    for l in lz:
        for i in range(0, len(l)):
            if l[i][1] == 0:
                l[i]=('123',0.0)
    selected_tag = set()
    if len(lz)!=1:
        for i in range(0,len(lz[0])):
            for l in range(0, len(lz)):
                if lz[l][i][0] != '123' and len(selected_tag) <3:
                    selected_tag.add(lz[l][i][0])
    elif len(lz)==1:
        for e in lz[0]:
            if e[0]!='123':
                selected_tag.add(e[0])
    return selected_tag


def task2_pre(all, userfile): # extract the users information from all.txt
    ud_dict, document_list = {}, []
    users = [i.lstrip('\ufeff').rstrip('\n') for i in userfile]
    for line in all:
        userid, Post, Browse, Comment, Vote_up, Vote_down, Favorite, Follow, Followed, Letter, Lettered = line.lstrip(
            '\ufeff').rstrip('\n').split(',')
        if userid in users:
            ud_dict[userid] = {}
            ud_dict[userid][1] = []
            ud_dict[userid][2] = []
            ud_dict[userid][3] = []
            templist1 = [Post]
            templist2 = [Browse, Comment, Vote_up, Vote_down, Favorite]
            templist3 =[Followed, Lettered]
            for i in templist1:
                if i != ' ':
                    docs = i.split('|')
                    document_list.extend(docs)
            ud_dict[userid][1] = docs
            for i in templist2:
                if i != ' ':
                    docs = i.split('|')
                    document_list.extend(docs)
            ud_dict[userid][2] = docs
            for i in templist3:
                if i != ' ':
                    docs = i.split('|')
                    document_list.extend(docs)
            ud_dict[userid][3] = docs
    document_list = set(document_list)
    print(len(ud_dict))
    print(len(document_list))
    return ud_dict, document_list

def doc2topic(path): #Load prediction file data and extract all tags unrepeatablely
    f1=open(path,'r',encoding='utf-8')
    docdict={}
    count=0
    for e in f1:
        did,topic1=e.lstrip('\ufeff').rstrip('\n').split(' ')
        if did in docdict.keys():
            count+=1
        else:
            docdict[did]=topic1
    print(count)
    return docdict

def task2(dict, DocVec, tagdict, res_path):
    fout = open(res_path, 'w', encoding='utf-8')
    for uid in dict.keys():
        allres={}
        res1 = []
        res2 = []
        res3 = []
        for e in dict[uid]:
            if e == 1:
                lis = dict[uid][e]
                if lis != []:
                    score={}
                    for ele in tagdict.keys():
                        score[ele]=0
                    for doc in lis:
                        try:
                            tag=DocVec[doc]
                            score[tag]+=1
                        except:
                            print(doc)
                    ar=[]
                    for element in score.keys():
                        ar.append((element,score[element]))
                        if element not in allres.keys():
                            allres[element]=score[element]
                        else:
                            allres[element]+=score[element]
                    aa = sorted(ar, key=lambda x: x[1], reverse=True)
                    for elem in range(0,3):
                        res1.append(aa[elem])
            if e == 2:
                lis = dict[uid][e]
                if lis != []:
                    score={}
                    for ele in tagdict.keys():
                        score[ele]=0
                    for doc in lis:
                        try:
                            tag=DocVec[doc]
                            score[tag]+=1
                        except:
                            print(doc)
                    ar=[]
                    for element in score.keys():
                        ar.append((element,score[element]))
                        if element not in allres.keys():
                            allres[element]=score[element]
                        else:
                            allres[element]+=score[element]
                    aa = sorted(ar, key=lambda x: x[1], reverse=True)
                    for elem in range(0, 3):
                        res2.append(aa[elem])
            if e == 3:
                lis = dict[uid][e]
                if lis != []:
                    score={}
                    for ele in tagdict.keys():
                        score[ele]=0
                    for doc in lis:
                        try:
                            tag=DocVec[doc]
                            score[tag]+=1
                        except:
                            print(doc)
                    ar=[]
                    for element in score.keys():
                        ar.append((element,score[element]))
                        if element not in allres.keys():
                            allres[element]=score[element]
                        else:
                            allres[element]+=score[element]
                    aa = sorted(ar, key=lambda x: x[1], reverse=True)
                    for elem in range(0, 3):
                        res3.append(aa[elem])
        result = uid
        for each in select_tag2(res1, res2, res3):
            result=result+','+str(each)
        result=result.strip()
        fout.write(result + '\n')

start = time.time()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
word_vectors = keyedvectors.KeyedVectors.load('data/med300-0-wv')
f1=open('data/test/submit.csv','r',encoding='utf-8')
f2=open('data/test/Tag','r',encoding='utf-8')
fo=open('data/restest.csv','w',encoding='utf-8')
dic1={}
for ele in f2:
    tag,tagid_=ele.rstrip('\n').split(',')
    dic1[tagid_]=tag
dic={}
for e in f1:
    did,tagid=e.rstrip('\n').split(' ')
    s=did+' '+str(dic1[tagid])
    fo.write(s+'\n')
path_='data/restest.csv'
DocVec=doc2topic(path_)
print(len(DocVec.keys()))
tagfile = open('data/test/SMPCUP2017_LabelSpace_Task2_Seg.txt', 'r', encoding='utf-8')
tagdict = {}
for line in tagfile:
    tag = line.lstrip('\ufeff').rstrip(' \n')
    n = len(tag.split(' '))
    if n == 1:
        tagdict[tag] = word_vectors.word_vec(tag)
    else:
        v = np.zeros(300)
        for word in tag.split(' '):
            v += word_vectors.word_vec(word)
        v = v / n
        tagdict[tag.replace(' ', '')] = v
tagfile.close()
f1 = open('data/test/SMPCUP2017_TestSet_Task2.txt', 'r', encoding='utf-8')
all=open('data/all.txt','r',encoding='utf-8')

dict1, lis = task2_pre(all,f1)
res_path = 'res/test'
task2(dict1, DocVec, tagdict, res_path)

taglist=[]
tagf=open('data/test/SMPCUP2017_LabelSpace_Task2_Seg.txt', 'r', encoding='utf-8')
for tag in tagf:
    a=tag.lstrip('\ufeff').rstrip(' \n')
    taglist.append(a.replace(' ', ''))
print(taglist)
f1=open('res/test','r',encoding='utf-8')
f2=open('res/task2_SLSS.csv','w',encoding='utf-8')
f2.write('%s,%s,%s,%s\n'%('userid','interest1','interest2','interest3'))

dictionary1={}
dictionary2={}
for tag in taglist:
    dictionary1[tag]=0
    dictionary2[tag]=0

for l1 in f1:

    try:
        user, tags = l1.lstrip('\ufeff').rstrip('\n').rstrip(',').replace("'", "").replace('[', '').replace(']',
                                                                                                            '').split(
            ',', 1)
        tags = set(tags.split(','))
    except:
        user= l1.lstrip('\ufeff').rstrip('\n').rstrip(',')
        tags=set()

    for tag in tags:
            dictionary1[tag]+=1
    for tag in ['移动开发','软件工程','web开发']:
        if len(tags)<3:
            tags.add(tag)
    for tag in tags:
        dictionary2[tag]+=1
    f2.write('%s,%s,%s,%s\n'%(user,list(tags)[0],list(tags)[1],list(tags)[2]))

f2.close()
for tag in taglist:
    print(dictionary1[tag],'\t',dictionary2[tag],'\t',tag)

end = time.time()
print(str(end - start))