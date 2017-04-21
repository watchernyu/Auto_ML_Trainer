#amended from tensorflow official tutorial

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from data_process_helper import *

import tensorflow as tf
import numpy as np
FLAGS = None
DIVIDE_FILE = False #only run this once to prepare data

def readXy(filename):
    dataFile = open(filename)
    lines = dataFile.readlines()
    data = []
    for line in lines:
        tokens = line.split(",")
        entry = [float(num) for num in tokens]
        data.append(entry)
    dataNP = np.matrix(data)
    print(filename,"data shape:",dataNP.shape)

    Y = np.ravel(dataNP[:,0])
    Y = Y.astype(int)
    X = dataNP[:,1:]

    xMean,xStd = getMeanStd(X)
    X = normalizeData(X,xMean,xStd)

    return X,Y

def main():
    nOfLabel = 2
    nOfHidden = 100
    nOfFeatures = 22

    sess = tf.InteractiveSession()
    print("##############program start#############")

    X_train , Y_train = readXy("pd_train.csv")
    Y_train = yToMatrix(Y_train,nOfLabel)#convert y array to one-hot matrix

    X_test , Y_test = readXy("pd_test.csv")
    Y_test = yToMatrix(Y_test,nOfLabel)#convert y array to one-hot matrix

    # Create the model
    x = tf.placeholder(tf.float32, [None, nOfFeatures]) #4 features, 10 hidden neurons

    W1 = tf.Variable(tf.random_normal([nOfFeatures,nOfHidden],stddev=0.0001))
    b1 = tf.Variable(tf.zeros([nOfHidden]))
    W2 = tf.Variable(tf.random_normal([nOfHidden,nOfLabel],stddev=0.0001))
    b2 = tf.Variable(tf.zeros([nOfLabel]))

    a2 = tf.sigmoid(tf.matmul(x,W1)+b1)
    y = tf.matmul(a2,W2)+b2

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, nOfLabel])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdadeltaOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.global_variables_initializer().run()
    # Train
    for i_iter in range(3000):

        sess.run(train_step, feed_dict={x: X_train, y_: Y_train})
        print("episode",i_iter,"train accuracy:",sess.run(accuracy, feed_dict={x:X_train ,y_: Y_train}))

    # Test trained model
    print ()
    print("training finished.")
    print("test accuracy:",sess.run(accuracy, feed_dict={x: X_test,
                                        y_: Y_test}))

    #print (sess.run(tf.argmax(y,1),feed_dict={x: irisX,
    #                                    y_: irisY}))#this will basically give the final prediction...

main()
