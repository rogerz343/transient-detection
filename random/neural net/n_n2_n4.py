#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:13:03 2018

@author: marcello
"""

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from helpers import perfMeasures


def ldMat(nm):
    mat_contents = sio.loadmat('../CIS 520 Feature Data/'+nm+'.mat')
    return mat_contents[nm]

x_test = ldMat('x_test')
y_test = ldMat('y_test')


x_train = ldMat('x_train')
y_train = ldMat('y_train')

y_train  = y_train.reshape((len(y_train),))
y_test  = y_test.reshape((len(y_test),))

x_tot = np.vstack((x_test, x_train))
y_tot = np.concatenate([y_test, y_train])

trP = 0.67

n = len(y_tot)
temp = np.arange(n)
np.random.shuffle(temp)
th = int(n*trP)

x_train, x_test = x_tot[temp<=th], x_tot[temp>th]
y_train, y_test = y_tot[temp<=th], y_tot[temp>th]


def randomShuffle(data_set, label_set):
    temp = np.arange(data_set.shape[0])
    np.random.shuffle(temp)
    data_set_cur = data_set[temp]
    label_set_cur = label_set[temp]
    return data_set_cur, label_set_cur
 
 
def obtainMiniBatch(data_set_cur, label_set_cur, j, step):
    n = data_set_cur.shape[0] / step
    s = int(j * n)
    if (s != 0):
        s = s + 1
    e = int((j + 1) * n)
    if (e < s):
        e = s
    if (e >= data_set_cur.shape[0]):
        e = (data_set_cur.shape[0])
    data_bt = data_set_cur[s : e, :]
    label_bt = label_set_cur[s : e]
    return data_bt, label_bt

def norma(x_train, y_train):
    data_set = x_train
    label_set = y_train

    l0 = -label_set + 1
    l1 = label_set
    label_set = np.transpose(np.vstack((l0, l1)))

    label_set = label_set.astype(np.float32)

    mnSub = (data_set - np.mean(data_set, 0))
    vr = np.sqrt(np.var(data_set, 0))
    data_set = mnSub/vr
    return data_set, label_set

data_set, label_set = norma(x_train, y_train)




# Set hyperparameters
learning_rate = 0.01
epochs = 100
batch_size = 5


#######Generalize below this


m = data_set.shape[0]

sz = data_set.shape[1]

layersizes = [sz, int(sz/2), int(sz/4), 2]
# Training data/label placeholders
x = tf.placeholder(tf.float32, [None, layersizes[0]])
y = tf.placeholder(tf.float32, [None,layersizes[-1]])

# Weights to first layer
W1 = tf.Variable(tf.random_normal([layersizes[0], layersizes[1]], stddev=0.1), name='W1')
b1 = tf.Variable(tf.random_normal([layersizes[1]]), name='b1')

# Weights to second layer
W2 = tf.Variable(tf.random_normal([layersizes[1], layersizes[2]], stddev=0.1), name='W2')
b2 = tf.Variable(tf.random_normal([layersizes[2]]), name='b2')

# Weights to third layer
W3 = tf.Variable(tf.random_normal([layersizes[2], layersizes[3]], stddev=0.1), name='W3')
b3 = tf.Variable(tf.random_normal([layersizes[3]]), name='b3')

# Output of hidden layers
hidden1 = tf.add(tf.matmul(x, W1), b1)
hidden1 = tf.nn.relu(hidden1)
hidden2 = tf.add(tf.matmul(hidden1, W2), b2)
hidden2 = tf.nn.relu(hidden2)

# Output of output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden2, W3), b3))

# Clip output to avoid log(0) error
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

# Cross entropy loss
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Initialization operator
init_op = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize training Session
with tf.Session() as sess:
   sess.run(init_op)
   batch = int(m / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        data_set_cur, label_set_cur = randomShuffle(data_set, label_set)
        for i in range(batch):
            batch_x, batch_y = obtainMiniBatch(data_set_cur, label_set_cur, i, batch)
            batch_x = batch_x.reshape((-1, sz))
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / batch
        print("Epoch", (epoch + 1), ": cross-entropy cost =", "{:.3f}".format(avg_cost))
   trnA = sess.run(accuracy, feed_dict={x: data_set, y: label_set})
   print("Accuracy:", trnA)
   
   data_set, label_set = norma(x_test, y_test)
   tstA = sess.run(accuracy, feed_dict={x: data_set, y: label_set})
   print("Accuracy:", tstA)
   pred = y_.eval(feed_dict={x: data_set}, session=sess)

   


p1 = pred[:,1]
lbl = np.argmax(label_set, axis=1)
n = 1000

import sys    
import os    
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
print(file_name)
dic = perfMeasures(p1, n, lbl, nm = file_name)
dic['trnA']=trnA
dic['tstA']=tstA
import pickle
def save_obj(obj, file_name ):
    with open('out/'+ file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
save_obj(dic, file_name )