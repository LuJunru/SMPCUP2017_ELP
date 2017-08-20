#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/8/15 16:18
# @Author  : ELP
# @Site    : 
# @File    : Task3_PGNS.py
# @Software: PyCharm

import sys
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from numpy import genfromtxt
from sklearn.cross_validation import KFold
from sklearn.svm import NuSVR
reload(sys)
sys.setdefaultencoding('utf-8')
import time

'''
We use four features to run the model, they are:
1.numbers of actions calculated by month--7*12 columns
  (actions: browsing, commentary, collection, forwarding, dating, attention, private messages)
2.arithmetic mean of 1--7 colums
3.log10 of 1--7*12 columns
4.growth rate of every kind of actions, which were processed previously like 1--7*11 columns
-----------------------------------------------
To run this python file, please:
1.install numpy,sklearn; load python 2.7
'''

start = time.clock()

#stacking, the second layer of NuSVR has been integrated in
def stacking(base_models, X, Y, T):
    models = base_models
    folds = list(KFold(len(Y), n_folds=10, random_state=0))
    S_train = np.zeros((X.shape[0], len(models)))
    S_test = np.zeros((T.shape[0], len(models)))
    for i, bm in enumerate(models):
        clf = bm[1]
        S_test_i = np.zeros((T.shape[0], len(folds)))
        for j, (train_idx, test_idx) in enumerate(folds):
            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_holdout = X[test_idx]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_holdout)[:]
            S_train[test_idx, i] = y_pred
            S_test_i[:, j] = clf.predict(T)[:]
        S_test[:, i] = S_test_i.mean(1)
    nuss=NuSVR(kernel='rbf')
    nuss.fit(S_train, Y)
    yp = nuss.predict(S_test)[:]
    return yp

# load train data, the growthrate and log value of train data has been preserved in advance
datapath1 = r'data/train/train_growthrate.csv'
deliverydata1 = genfromtxt(datapath1, delimiter=',', skip_header=False)
train_x1 = deliverydata1[:, 1:]
datapath2 = r'data/train/train_orginal_log.csv'
deliverydata2 = genfromtxt(datapath2, delimiter=',', skip_header=False)
# generate arithmetic mean of train data
x8 = np.mean(deliverydata2[:, 1:13], axis=1)
x9 = np.mean(deliverydata2[:, 13:25], axis=1)
x10 = np.mean(deliverydata2[:, 25:37], axis=1)
x11 = np.mean(deliverydata2[:, 38:49], axis=1)
x12 = np.mean(deliverydata2[:, 49:61], axis=1)
x13 = np.mean(deliverydata2[:, 61:73], axis=1)
x14 = np.mean(deliverydata2[:, 73:85], axis=1)

# after correlation analysis, column 38 and 122 were deleted for their low correlations with rest of other columns
train_x2 = np.column_stack((deliverydata2[:, 1:37], deliverydata2[:, 38:121], deliverydata2[:, 122:-1],
                            train_x1, x8, x9, x10, x11, x12, x13, x14))
train_y2 = deliverydata2[:, -1]
# seperate the train data by 10-fold method
x_tr, x_te, y_tr, y_te = train_test_split(train_x2, train_y2, test_size=0.1, random_state=0)

#load test data
datapath_test1 = r'data/test/test_growthrate.csv'
deliverydata_test1 = genfromtxt(datapath_test1, delimiter=',', skip_header=False)
datapath_test2 = r'data/test/test_original.csv'
deliverydata_test2 = genfromtxt(datapath_test2, delimiter=',', skip_header=False)
#test id
test_id=[]
for line in open('data/test/SMPCUP2017_TestSet_Task3.txt','r'):
    test_id.append(line.strip())

# models for first layer
gbr = GradientBoostingRegressor(n_estimators=1800, learning_rate=0.08, max_depth=8, min_samples_split=13, alpha=0.91)
pa = PassiveAggressiveRegressor(C=0.9, n_iter=1)

j=0
final_L = np.zeros((16085,))
k = 0.0
'''
Actually, the prediction of our PGNS is variable. You can change the number of epochs from "20" to a bigger number if you
want to get a much more stable prediction.
It takes about 160s to run off one epoch for test data on Mac OS Sierra 10.12.6(2.4 GHz Intel Core i5; 8 GB 1600 MHz DDR3)
'''
while j<=20:

    #a socre on train prediction with 10-fold method
    L = stacking([['Pa', pa], ['Gbr', gbr]], x_tr, y_tr, x_te).tolist()
    i = 0
    final_score = 0.0
    y_te_list = y_te.tolist()
    while i < len(y_te_list):
        if abs(L[i]) == 0 and y_te_list[i] == 0:
            score = 0.0
        else:
            score = abs(abs(L[i]) - y_te_list[i]) / max(abs(L[i]), y_te_list[i])
        final_score += score
        i += 1
    print 1.0 - final_score / len(y_te_list)

    '''
    It was shown on train and valid data that 0.765 is an almost highest boundary on which our model can make a most stable
    and accurate prediction of valid and test data.
    Thus, groups of weights that predict socres higher than 0.765 above will be used to get a prediction on test data and
    an average of these predictions will be preserved finally.
    '''
    if (1.0 - final_score / len(y_te_list))>0.765:

        test_x1 = deliverydata_test1[:, 1:]
        test_action=np.column_stack((deliverydata_test2[:,1:37],deliverydata_test2[:,38:]))
        test_log=np.log10(np.add(test_action,(np.ones(test_action.shape)*1)))
        xx8 = np.mean(deliverydata_test2[:, 1:13], axis=1)
        xx9 = np.mean(deliverydata_test2[:, 13:25], axis=1)
        xx10 = np.mean(deliverydata_test2[:, 25:37], axis=1)
        xx11 = np.mean(deliverydata_test2[:, 38:49], axis=1)
        xx12 = np.mean(deliverydata_test2[:, 49:61], axis=1)
        xx13 = np.mean(deliverydata_test2[:, 61:73], axis=1)
        xx14 = np.mean(deliverydata_test2[:, 73:85], axis=1)

        text_x2 = np.column_stack((test_action,test_log,test_x1,xx8,xx9,xx10,xx11,xx12,xx13,xx14))

        #prediction
        L1 = stacking([['pa', pa], ['gbr', gbr]], train_x2, train_y2, text_x2)
        final_L = np.add(final_L,L1)
        k+=1.0

    j+=1

#store results in a CSV file
csvfile=file("res/task3_PGNS.csv","w")
writer=csv.writer(csvfile)
writer.writerow(['userid','growthvalue'])

L2=final_L.transpose().tolist()
L3=[r/k for r in L2]

m=0
for l in L3:
    n=[]
    n.append(test_id[m])
    n.append(round(l,3))
    writer.writerow(n)
    m+=1

csvfile.close()

end = time.clock()
print '\n'
print "read: %f s" % (end - start)