#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 11:51:31 2018

@author: marcello
"""
from time import gmtime, strftime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
def save_obj(obj, file_name ):
    with open('out/'+ file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
import sys
def load_obj(name ):
    with open('out/' + name + '.pkl', 'rb') as f:
        if sys.version_info.major > 2:
            d = pickle.load(f, encoding='latin1')
        else:
            d = pickle.load(f)
    return d

ls = ['nT', 'minSLeaf', 'mF', 'mD']
ttls = ['Number of Trees', 'Minimum Number of Samples', 'Maximum Features', 'Maximum Depth of Tree']
dirc = os.listdir('out')
dirc = [s for s in dirc if s.endswith('.pkl')]


def pltNm(str1, str2, dirc, cat, ttl, x1 = 0, x2 = 0.1, y1 = 0, y2 = 0.05):
    plt.figure()
    nmD = [s[:-4] for s in dirc if s.startswith(cat)]
    ints = np.array([int(s[len(cat):]) for s in nmD])
    nmD = np.array(nmD)[ints.argsort()].tolist()
    for n in nmD:
        print(n)
        dic = load_obj(n)
        Tar=np.array(dic[str1])
        Far=np.array(dic[str2])
        plt.scatter(Far,1-Tar, s=1)
        plt.xlim(x1, x2)
        plt.ylim(y1, y2)
    plt.legend([str(s) for s in ints])
    plt.title(ttl)
    plt.xlabel(str2)
    plt.ylabel(str1)
    plt.savefig('out/figs'+cat+str1+str2)
for i in range(len(ls)):
    pltNm('TPR', 'FPR', dirc, ls[i], 'ROC over '+ttls[i])
    pltNm('Precision', 'Recall', dirc, ls[i], 'ROC over '+ttls[i])

def cmp(str1, str2, nmD, cat, ttl, x1 = 0, x2 = 0.1, y1 = 0, y2 = 0.05):
    plt.figure()
    for n in nmD:
        print(n)
        dic = load_obj(n)
        Tar=np.array(dic[str1])
        Far=np.array(dic[str2])
        plt.scatter(Far,1-Tar, s=1)
        plt.xlim(x1, x2)
        plt.ylim(y1, y2)
    plt.legend(nmD)
    plt.title(ttl)
    plt.xlabel(str2)
    plt.ylabel(str1)
    plt.savefig('out/figs'+cat+str1+str2)
cmp('TPR', 'FPR', ['nT1000','noPCA'], 'noPCA', 'ROC over '+'noPCA')
cmp('Precision', 'Recall', ['nT1000','noPCA'], 'noPCA', 'ROC over '+'noPCA')

newDic={}
dics = []
newDic['AUCPR']=[]
newDic['AUCROC']=[]
newDic['trnA']=[]
newDic['tstA']=[]
for f in dirc:
    dic = load_obj(f[:-4])
    dics.append(dic)
    newDic['AUCPR'].append(dic['AUCPR'])
    newDic['AUCROC'].append(dic['AUCROC'])
    if 'trnA' in dic:
        newDic['trnA'].append(dic['trnA'])
    else:
        newDic['trnA'].append([])
    if 'tstA' in dic:
        newDic['tstA'].append(dic['tstA'])
    else:
        newDic['tstA'].append([])
PR_Max = max(newDic['AUCPR'])
ROC_Max = max(newDic['AUCROC'])
ROC_In = np.argmax(np.array(newDic['AUCROC']))
Trn_Max = max(newDic['trnA'])
Tst_Max = max(newDic['tstA'])
l = newDic['tstA']

for i in range(len(l)):
    if not l[i]:
        l[i]=0
Tst_In = np.argmax(np.array(l))

f = dirc[ROC_In]
dic = load_obj(f[:-4])
bestRFROC = [dic['FPR'],dic['TPR']]



plt.figure()
plt.hold(True)
plt.scatter(np.array(bestRFROC[0]),1-np.array(bestRFROC[1]))
plt.scatter(np.array(bestNNROC[0]),1-np.array(bestNNROC[1]))
plt.xlim(0.0, 0.1)
plt.ylim(0, 0.05)
plt.title("ROC")